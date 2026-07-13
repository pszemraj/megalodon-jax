# Copyright 2025 Peter Szemraj.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference utilities: cache helpers, sampling, and generation."""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM
from megalodon_jax.types import AttentionCache, LayerCache, ModelCache, NormState

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def init_cache(
    config: MegalodonConfig,
) -> ModelCache:
    """Create a sparse zero-history cache for autoregressive generation.

    :param MegalodonConfig config: Model configuration.
    :return ModelCache: Initialized cache with per-layer state.
    """

    layer_caches = tuple(None for _ in range(config.num_layers))
    return ModelCache(layer_caches=layer_caches, final_norm=None)


def index_cache(cache: ModelCache, indices: Int[Array, "new_batch"]) -> ModelCache:
    """Select batch elements from a ModelCache.

    Note: cache position/count fields are shared scalars; this is only valid
    when all batch elements share the same history length.

    :param ModelCache cache: Cache to slice.
    :param Int[Array, "new_batch"] indices: Batch indices to select.
    :return ModelCache: Sliced cache.
    :raises TypeError: If indices do not have an integer dtype.
    :raises ValueError: If indices are not rank one or are out of bounds.
    """

    source_dtype = getattr(indices, "dtype", None)
    if source_dtype is not None and np.dtype(source_dtype) != np.dtype(np.int32):
        raise TypeError(f"indices must have dtype int32, got {source_dtype}")

    indices = jnp.asarray(indices)
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank one, got shape {indices.shape}")
    if indices.dtype != jnp.int32:
        raise TypeError(f"indices must have dtype int32, got {indices.dtype}")

    batch_size: int | None = None

    def bind_batch(name: str, x: Array) -> None:
        """Bind and cross-check the batch width of one cache array.

        :param str name: Cache component name used in validation errors.
        :param Array x: Cache array with a leading batch dimension.
        """
        nonlocal batch_size
        if x.ndim == 0:
            raise ValueError(f"{name} must have a batch dimension")
        if batch_size is None:
            batch_size = x.shape[0]
        elif x.shape[0] != batch_size:
            raise ValueError(f"cache batch mismatch at {name}: {x.shape[0]} != {batch_size}")

    for layer_index, layer_cache in enumerate(cache.layer_caches):
        if layer_cache is None:
            continue
        if layer_cache.attn is not None:
            bind_batch(f"layers.{layer_index}.attn.k", layer_cache.attn.k)
            bind_batch(f"layers.{layer_index}.attn.v", layer_cache.attn.v)
        if layer_cache.norm is not None:
            bind_batch(f"layers.{layer_index}.norm.count", layer_cache.norm.count)
            bind_batch(f"layers.{layer_index}.norm.mean", layer_cache.norm.mean)
            bind_batch(f"layers.{layer_index}.norm.var", layer_cache.norm.var)
        if layer_cache.ema is not None:
            bind_batch(f"layers.{layer_index}.ema.h", layer_cache.ema.h)
    if cache.final_norm is not None:
        bind_batch("final_norm.count", cache.final_norm.count)
        bind_batch("final_norm.mean", cache.final_norm.mean)
        bind_batch("final_norm.var", cache.final_norm.var)

    if batch_size is None:
        if indices.size != 0:
            raise ValueError("cannot index a sparse cache without allocated batch state")
    else:
        indices = eqx.error_if(
            indices,
            jnp.any((indices < 0) | (indices >= batch_size)),
            f"cache indices must be in [0, {batch_size})",
        )

    def index_array(x: Array | None) -> Array | None:
        """Index a cache array along the batch dimension.

        :param Array | None x: Cache array to index.
        :return Array | None: Indexed array or None.
        """
        if x is None:
            return None
        return x[indices]

    def index_layer(layer_cache: LayerCache | None) -> LayerCache | None:
        """Index a LayerCache along the batch dimension.

        :param LayerCache | None layer_cache: Layer cache to index.
        :return LayerCache | None: Indexed layer cache.
        """
        if layer_cache is None:
            return None

        attn = layer_cache.attn
        if attn is not None:
            attn = AttentionCache(
                k=index_array(attn.k),
                v=index_array(attn.v),
                count=attn.count,
            )

        norm = layer_cache.norm
        if norm is not None:
            norm = NormState(
                count=index_array(norm.count),
                mean=index_array(norm.mean),
                var=index_array(norm.var),
            )

        ema = layer_cache.ema
        if ema is not None:
            ema = type(ema)(h=index_array(ema.h))

        return LayerCache(
            attn=attn,
            norm=norm,
            ema=ema,
            position=layer_cache.position,
        )

    indexed_layers = tuple(index_layer(lc) for lc in cache.layer_caches)
    final_norm = cache.final_norm
    if final_norm is not None:
        final_norm = NormState(
            count=index_array(final_norm.count),
            mean=index_array(final_norm.mean),
            var=index_array(final_norm.var),
        )

    return ModelCache(layer_caches=indexed_layers, final_norm=final_norm)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def greedy_token(logits: Float[Array, "batch vocab"]) -> Int[Array, "batch"]:
    """Deterministic argmax sampling.

    :param Float[Array, "batch vocab"] logits: Logits for sampling.
    :return Int[Array, "batch"]: Selected token IDs.
    """

    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _top_k(logits: Array, k: int) -> tuple[Array, Array]:
    """Run ``lax.top_k`` after clamping k to the vocabulary width.

    :param Array logits: Logits whose final axis is the vocabulary dimension.
    :param int k: Requested number of leading values and indices.
    :return tuple[Array, Array]: Top values and their indices, ordered by value.
    """
    return jax.lax.top_k(logits, min(k, logits.shape[-1]))


def _apply_top_k(
    logits: Float[Array, "batch vocab"], top_k: int | None
) -> Float[Array, "batch vocab"]:
    """Apply top-k filtering to logits.

    :param Float[Array, "batch vocab"] logits: Logits to filter.
    :param int | None top_k: Top-k value.
    :return Float[Array, "batch vocab"]: Filtered logits.
    """
    if top_k is None or top_k <= 0:
        return logits
    values, indices = _top_k(logits, top_k)
    filtered = jnp.full_like(logits, -jnp.inf)
    batch = jnp.arange(logits.shape[0])[:, None]
    return filtered.at[batch, indices].set(values)


def _apply_top_p(
    logits: Float[Array, "batch vocab"],
    top_p: float | None,
    *,
    top_k: int | None = None,
) -> Float[Array, "batch vocab"]:
    """Apply nucleus (top-p) filtering to logits.

    :param Float[Array, "batch vocab"] logits: Logits to filter.
    :param float | None top_p: Nucleus probability threshold.
    :param int | None top_k: Optional top-k prefilter.
    :return Float[Array, "batch vocab"]: Filtered logits.
    """
    if top_p is None or top_p >= 1.0:
        return logits

    if top_k is not None and top_k > 0:
        sorted_logits, sorted_indices = _top_k(logits, top_k)
    else:
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative = jnp.cumsum(sorted_probs, axis=-1)

    # Mask tokens past the nucleus boundary; keep first above threshold
    sorted_mask = cumulative > top_p
    sorted_mask = jnp.concatenate(
        [jnp.zeros_like(sorted_mask[:, :1]), sorted_mask[:, :-1]], axis=-1
    )

    if top_k is not None and top_k > 0:
        masked_logits = jnp.where(sorted_mask, -jnp.inf, sorted_logits)
        out = jnp.full_like(logits, -jnp.inf)
        batch_idx = jnp.arange(logits.shape[0])[:, None]
        return out.at[batch_idx, sorted_indices].set(masked_logits)

    # Map back to original order
    scatter_order = jnp.argsort(sorted_indices, axis=-1)
    mask = jnp.take_along_axis(sorted_mask, scatter_order, axis=-1)
    return jnp.where(mask, -jnp.inf, logits)


def sample_token(
    logits: Float[Array, "batch vocab"],
    key: PRNGKeyArray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Int[Array, "batch"]:
    """Sample next token with temperature, top-k, and top-p.

    :param Float[Array, "batch vocab"] logits: Logits for sampling.
    :param PRNGKeyArray key: PRNG key for sampling.
    :param float temperature: Sampling temperature.
    :param int | None top_k: Top-k filter.
    :param float | None top_p: Top-p filter.
    :raises ValueError: If temperature/top-k/top-p inputs are invalid.
    :return Int[Array, "batch"]: Sampled token IDs.
    """

    _validate_sampling(temperature, top_k, top_p, logits.shape[-1])

    if temperature == 0.0:
        return greedy_token(logits)

    work = logits.astype(jnp.float32)
    if temperature != 1.0:
        work = work / jnp.asarray(temperature, dtype=work.dtype)

    if top_p is not None and top_p < 1.0:
        work = _apply_top_p(work, top_p, top_k=top_k)
    else:
        work = _apply_top_k(work, top_k)

    # Categorical accepts logits directly; keep float32 for numeric stability.
    return jax.random.categorical(key, work, axis=-1).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _sample_fn(
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> Callable[[Float[Array, "batch vocab"], PRNGKeyArray | None], Int[Array, "batch"]]:
    """Build a sampler callable configured with temperature/top-k/top-p.

    :param float temperature: Sampling temperature.
    :param int | None top_k: Top-k filter.
    :param float | None top_p: Top-p filter.
    :return Callable[[Float[Array, "batch vocab"], PRNGKeyArray | None], Int[Array, "batch"]]: Sampler.
    """
    if temperature == 0.0:
        return lambda logits, _: greedy_token(logits)
    return functools.partial(sample_token, temperature=temperature, top_k=top_k, top_p=top_p)


def _validate_sampling(
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    output_size: int,
) -> None:
    """Validate sampling controls before selecting greedy or stochastic execution.

    :param float temperature: Sampling temperature, including zero for greedy decoding.
    :param int | None top_k: Maximum number of candidate tokens, or None.
    :param float | None top_p: Nucleus probability threshold, or None.
    :param int output_size: Width of the model output vocabulary.
    :raises ValueError: If a control is non-finite, out of range, or has an invalid type.
    """
    if isinstance(temperature, (bool, np.bool_)):
        raise ValueError("temperature must be a non-boolean finite number")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"temperature must be finite and >= 0, got {temperature}")
    if top_k is not None:
        if isinstance(top_k, bool):
            raise ValueError(f"top_k must be an integer, got {top_k!r}")
        try:
            top_k_value = operator.index(top_k)
        except TypeError as error:
            raise ValueError(f"top_k must be an integer, got {top_k!r}") from error
        if not 0 <= top_k_value <= output_size:
            raise ValueError(f"top_k must be in [0, {output_size}], got {top_k}")
    if top_p is not None:
        if isinstance(top_p, (bool, np.bool_)):
            raise ValueError("top_p must be a non-boolean finite number")
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be finite and in (0, 1], got {top_p}")


def _validate_max_new_tokens(max_new_tokens: object) -> int:
    """Normalize a requested generation length through the integer protocol.

    :param object max_new_tokens: Requested number of generated tokens.
    :raises ValueError: If the value is not a positive, non-boolean integer.
    :return int: Normalized positive generation length.
    """
    if isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be a positive integer")
    try:
        value = operator.index(max_new_tokens)
    except TypeError as error:
        raise ValueError("max_new_tokens must be a positive integer") from error
    if value <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    return value


def _validate_token_id_override(
    name: str,
    token_id: object | None,
    vocab_size: int,
) -> int | None:
    """Normalize an optional special-token override and enforce vocabulary bounds.

    :param str name: Argument name used in validation errors.
    :param object | None token_id: Optional token ID override.
    :param int vocab_size: Exclusive upper vocabulary bound.
    :raises ValueError: If the override is not a non-boolean integer in range.
    :return int | None: Normalized token ID or ``None``.
    """
    if token_id is None:
        return None
    if isinstance(token_id, bool):
        raise ValueError(f"{name} must be an integer token ID")
    try:
        value = operator.index(token_id)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer token ID") from error
    if not 0 <= value < vocab_size:
        raise ValueError(f"{name} must be in [0, {vocab_size}), got {value}")
    return value


def _generate_core(
    model: MegalodonForCausalLM,
    prompt_ids: Int[Array, "batch prompt_len"],
    max_new_tokens: int,
    key: PRNGKeyArray | None = None,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    attention_mask: Bool[Array, "batch prompt_len"] | None = None,
    cache: ModelCache | None = None,
    return_cache: bool = False,
) -> tuple[Int[Array, "batch total_len"], ModelCache | None, PRNGKeyArray | None]:
    """Autoregressive generation using a fixed-shape scan.

    If return_cache is True, the returned cache includes the prompt and every
    generated token. If sampling is enabled (temperature > 0), the returned
    key is the next key to use for subsequent sampling calls.

    :param MegalodonForCausalLM model: Model to generate from.
    :param Int[Array, "batch prompt_len"] prompt_ids: Prompt token IDs.
    :param int max_new_tokens: Number of new tokens to generate.
    :param PRNGKeyArray | None key: PRNG key for sampling.
    :param float temperature: Sampling temperature.
    :param int | None top_k: Top-k filter.
    :param float | None top_p: Top-p filter.
    :param int | None bos_token_id: Optional BOS token ID.
    :param int | None eos_token_id: Optional EOS token ID.
    :param Bool[Array, "batch prompt_len"] | None attention_mask: Optional mask.
    :param ModelCache | None cache: Optional cache state.
    :param bool return_cache: Whether to return cache.
    :raises ValueError: If inputs are invalid or missing required RNG.
    :return tuple[Int[Array, "batch total_len"], ModelCache | None, PRNGKeyArray | None]: Output IDs, cache, key.
    """

    if model.config.effective_output_size != model.config.vocab_size:
        raise ValueError("generate requires effective_output_size to equal vocab_size")
    _validate_sampling(
        temperature,
        top_k,
        top_p,
        model.config.effective_output_size,
    )
    needs_rng = temperature != 0.0
    if needs_rng and key is None:
        raise ValueError("key is required when temperature > 0.")

    sample = _sample_fn(temperature, top_k, top_p)
    B, prompt_len = prompt_ids.shape

    if prompt_len == 0:
        if cache is not None:
            raise ValueError(
                "empty-prompt continuation with a cache is unsupported; "
                "provide the final context token or cached logits"
            )
        if bos_token_id is None:
            bos_token_id = getattr(model.config, "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError(
                "prompt_ids is empty; provide bos_token_id or pass at least one token."
            )
        prompt_ids = jnp.full((B, 1), bos_token_id, dtype=prompt_ids.dtype)
        # The synthetic BOS is valid and unpadded. Cached calls represent the
        # all-valid case with no mask metadata.
        attention_mask = None
        prompt_len = 1

    needs_cache = return_cache or max_new_tokens > 1

    # Prefill
    logits, cache = model(
        prompt_ids,
        attention_mask=attention_mask,
        cache=cache,
        return_cache=needs_cache,
        deterministic=True,
    )

    if attention_mask is not None:
        mask = attention_mask.astype(jnp.bool_)
        positions = jnp.arange(prompt_len, dtype=jnp.int32)
        masked_positions = jnp.where(mask, positions, -1)
        last_idx = masked_positions.max(axis=1)
        has_empty = jnp.any(last_idx < 0)
        last_idx = eqx.error_if(
            last_idx,
            has_empty,
            "attention_mask must contain at least one True per batch element.",
        )
        gather_idx = jnp.broadcast_to(last_idx[:, None, None], (B, 1, logits.shape[-1]))
        last_logits = jnp.take_along_axis(logits, gather_idx, axis=1)[:, 0, :]
    else:
        last_logits = logits[:, -1, :]

    if needs_rng:
        key, subkey = jax.random.split(key)
    else:
        subkey = None
    first_token = sample(last_logits, subkey)

    finished = jnp.zeros((B,), dtype=jnp.bool_)
    if eos_token_id is not None:
        finished = first_token == eos_token_id

    if needs_rng:

        def scan_step(
            carry: tuple[ModelCache, Int[Array, "batch"], PRNGKeyArray, Bool[Array, "batch"]],
            _: None,
        ) -> tuple[
            tuple[ModelCache, Int[Array, "batch"], PRNGKeyArray, Bool[Array, "batch"]],
            Int[Array, "batch"],
        ]:
            """Scan step for autoregressive decoding with RNG.

            :param tuple[ModelCache, Int[Array, "batch"], PRNGKeyArray, Bool[Array, "batch"]] carry: Scan carry.
            :param None _: Unused scan input.
            :return tuple[tuple[ModelCache, Int[Array, "batch"], PRNGKeyArray, Bool[Array, "batch"]], Int[Array, "batch"]]: Carry and token.
            """
            cached, last_token, rng, done = carry

            logits_step, new_cache = model(
                last_token[:, None],
                cache=cached,
                return_cache=True,
                deterministic=True,
            )

            rng, sub = jax.random.split(rng)
            next_token = sample(logits_step[:, 0, :], sub)

            if eos_token_id is not None:
                newly_done = next_token == eos_token_id
                done = done | newly_done
                next_token = jnp.where(done, eos_token_id, next_token)

            return (new_cache, next_token, rng, done), next_token

    else:

        def scan_step(
            carry: tuple[ModelCache, Int[Array, "batch"], Bool[Array, "batch"]],
            _: None,
        ) -> tuple[
            tuple[ModelCache, Int[Array, "batch"], Bool[Array, "batch"]],
            Int[Array, "batch"],
        ]:
            """Scan step for autoregressive decoding without RNG.

            :param tuple[ModelCache, Int[Array, "batch"], Bool[Array, "batch"]] carry: Scan carry.
            :param None _: Unused scan input.
            :return tuple[tuple[ModelCache, Int[Array, "batch"], Bool[Array, "batch"]], Int[Array, "batch"]]: Carry and token.
            """
            cached, last_token, done = carry

            logits_step, new_cache = model(
                last_token[:, None],
                cache=cached,
                return_cache=True,
                deterministic=True,
            )

            next_token = sample(logits_step[:, 0, :], None)

            if eos_token_id is not None:
                newly_done = next_token == eos_token_id
                done = done | newly_done
                next_token = jnp.where(done, eos_token_id, next_token)

            return (new_cache, next_token, done), next_token

    if max_new_tokens == 1:
        if return_cache:
            _, cache = model(
                first_token[:, None],
                cache=cache,
                return_cache=True,
                deterministic=True,
            )
        final_cache = cache if return_cache else None
        generated = first_token[:, None]
        final_key = key
    else:
        if needs_rng:
            init_carry = (cache, first_token, key, finished)
            final_carry, tokens_scan = jax.lax.scan(
                scan_step,
                init_carry,
                None,
                length=max_new_tokens - 1,
            )
            scan_cache = final_carry[0]
            final_key = final_carry[2]
        else:
            init_carry = (cache, first_token, finished)
            final_carry, tokens_scan = jax.lax.scan(
                scan_step,
                init_carry,
                None,
                length=max_new_tokens - 1,
            )
            scan_cache = final_carry[0]
            final_key = key

        generated = jnp.concatenate([first_token[:, None], tokens_scan.T], axis=1)
        if return_cache:
            last_token = generated[:, -1]
            _, final_cache = model(
                last_token[:, None],
                cache=scan_cache,
                return_cache=True,
                deterministic=True,
            )
        else:
            final_cache = None

    result = jnp.concatenate([prompt_ids, generated], axis=1)
    return result, final_cache, final_key


def generate(
    model: MegalodonForCausalLM,
    prompt_ids: Int[Array, "batch prompt_len"],
    max_new_tokens: int,
    key: PRNGKeyArray | None = None,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    attention_mask: Bool[Array, "batch prompt_len"] | None = None,
    cache: ModelCache | None = None,
    return_cache: bool = False,
) -> tuple[Int[Array, "batch total_len"], ModelCache | None, PRNGKeyArray | None]:
    """Autoregressive generation with padding-aware constraints.

    :param MegalodonForCausalLM model: Model to generate from.
    :param Int[Array, "batch prompt_len"] prompt_ids: Prompt token IDs.
    :param int max_new_tokens: Number of new tokens to generate.
    :param PRNGKeyArray | None key: PRNG key for sampling.
    :param float temperature: Sampling temperature.
    :param int | None top_k: Top-k filter.
    :param float | None top_p: Top-p filter.
    :param int | None bos_token_id: Optional BOS token ID.
    :param int | None eos_token_id: Optional EOS token ID.
    :param Bool[Array, "batch prompt_len"] | None attention_mask: Optional mask.
    :param ModelCache | None cache: Optional cache.
    :param bool return_cache: Whether to return cache.
    :raises ValueError: If inputs are invalid or padding constraints are violated.
    :return tuple[Int[Array, "batch total_len"], ModelCache | None, PRNGKeyArray | None]: Output IDs, cache, key.
    """

    max_new_tokens = _validate_max_new_tokens(max_new_tokens)
    bos_token_id = _validate_token_id_override(
        "bos_token_id",
        bos_token_id,
        model.config.vocab_size,
    )
    eos_token_id = _validate_token_id_override(
        "eos_token_id",
        eos_token_id,
        model.config.vocab_size,
    )

    if attention_mask is None:
        return _generate_core(
            model,
            prompt_ids,
            max_new_tokens,
            key,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            attention_mask=None,
            cache=cache,
            return_cache=return_cache,
        )

    attention_mask = jnp.asarray(attention_mask)
    if attention_mask.shape != prompt_ids.shape:
        raise ValueError(
            "attention_mask shape must match prompt_ids shape, got "
            f"{attention_mask.shape} and {prompt_ids.shape}"
        )
    has_padding = not bool(jax.device_get(jnp.all(attention_mask)))
    needs_cache = cache is not None or return_cache or max_new_tokens > 1
    if has_padding and needs_cache:
        raise ValueError(
            "Cannot use cache with padded attention_mask. "
            "Provide unpadded prompts for cached generation."
        )
    if not has_padding:
        attention_mask = None

    return _generate_core(
        model,
        prompt_ids,
        max_new_tokens,
        key,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        attention_mask=attention_mask,
        cache=cache,
        return_cache=return_cache,
    )
