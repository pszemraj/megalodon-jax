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
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM
from megalodon_jax.types import AttentionCache, EMAState, LayerCache, ModelCache, NormState

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def init_cache(
    config: MegalodonConfig,
    batch_size: int,
    dtype: jnp.dtype = jnp.float32,
    *,
    allocate_kv: bool = False,
    allocate_norm: bool = False,
    allocate_ema: bool = False,
) -> ModelCache:
    """Create an empty ModelCache for autoregressive generation.

    :param MegalodonConfig config: Model configuration.
    :param int batch_size: Batch size for cached tensors.
    :param jnp.dtype dtype: Floating dtype for cached arrays.
    :param bool allocate_kv: Whether to pre-allocate KV buffers.
    :param bool allocate_norm: Whether to pre-allocate TimestepNorm state.
    :param bool allocate_ema: Whether to pre-allocate ComplexEMA state.
    :return ModelCache: Initialized cache with per-layer state.
    """

    cache_len = config.effective_max_cache_len
    num_heads = config.num_heads
    head_dim = config.head_dim
    value_head_dim = config.value_head_dim

    def make_attn_cache() -> AttentionCache | None:
        """Build an AttentionCache when KV preallocation is requested.

        :return AttentionCache | None: Allocated cache or None.
        """
        if not allocate_kv:
            return None
        k = jnp.zeros((batch_size, cache_len, num_heads, head_dim), dtype=dtype)
        v = jnp.zeros((batch_size, cache_len, num_heads, value_head_dim), dtype=dtype)
        return AttentionCache(k=k, v=v, count=jnp.array(0, dtype=jnp.int32))

    def make_norm_state() -> NormState:
        """Create an initialized NormState for a batch.

        :return NormState: Initialized norm state.
        """
        return NormState(
            count=jnp.zeros((batch_size,), dtype=jnp.int32),
            mean=jnp.zeros((batch_size, config.norm_num_groups), dtype=jnp.float32),
            var=jnp.ones((batch_size, config.norm_num_groups), dtype=jnp.float32),
        )

    def make_ema_state() -> EMAState:
        """Create an initialized EMAState for a batch.

        :return EMAState: Initialized EMA state.
        """
        h = jnp.zeros((batch_size, config.model_dim, config.cema_ndim), dtype=jnp.complex64)
        return EMAState(h=h)

    layer_caches = []
    for _ in range(config.num_layers):
        layer_caches.append(
            LayerCache(
                attn=make_attn_cache(),
                norm=make_norm_state() if allocate_norm else None,
                ema=make_ema_state() if allocate_ema else None,
                position=jnp.array(0, dtype=jnp.int32),
            )
        )

    final_norm = make_norm_state()

    return ModelCache(layer_caches=tuple(layer_caches), final_norm=final_norm)


def trim_cache(cache: ModelCache, max_len: int) -> ModelCache:
    """Trim KV cache entries to the most recent ``max_len`` tokens.

    The cache is stored as a ring buffer; trimming preserves absolute positions
    by reindexing into a smaller ring when needed.

    :param ModelCache cache: Cache to trim.
    :param int max_len: Maximum number of tokens to retain.
    :return ModelCache: Trimmed cache.
    """

    def trim_layer(layer_cache: LayerCache | None) -> LayerCache | None:
        """Trim a layer cache to the newest max_len entries.

        :param LayerCache | None layer_cache: Layer cache to trim.
        :return LayerCache | None: Trimmed layer cache.
        """
        if layer_cache is None or layer_cache.attn is None:
            return layer_cache
        attn = layer_cache.attn
        cache_size = attn.k.shape[1]
        if cache_size <= max_len:
            return layer_cache
        valid_len = jnp.minimum(attn.count, cache_size)
        keep_len = jnp.minimum(valid_len, max_len)
        idx = jnp.arange(max_len, dtype=jnp.int32)
        keep_mask = idx < keep_len
        start_pos = attn.count - keep_len
        old_idx = jnp.mod(start_pos + idx, cache_size)
        new_idx = jnp.mod(start_pos + idx, max_len)

        k_keep = jnp.take(attn.k, old_idx, axis=1)
        v_keep = jnp.take(attn.v, old_idx, axis=1)
        mask = keep_mask[None, :, None, None]
        k_keep = jnp.where(mask, k_keep, jnp.zeros((), dtype=attn.k.dtype))
        v_keep = jnp.where(mask, v_keep, jnp.zeros((), dtype=attn.v.dtype))

        B, _, H, Dh = attn.k.shape
        _, _, _, Dv = attn.v.shape
        new_k = jnp.zeros((B, max_len, H, Dh), dtype=attn.k.dtype)
        new_v = jnp.zeros((B, max_len, H, Dv), dtype=attn.v.dtype)
        new_k = new_k.at[:, new_idx].set(k_keep)
        new_v = new_v.at[:, new_idx].set(v_keep)

        trimmed = AttentionCache(k=new_k, v=new_v, count=attn.count)
        return LayerCache(
            attn=trimmed,
            norm=layer_cache.norm,
            ema=layer_cache.ema,
            position=layer_cache.position,
        )

    trimmed_layers = tuple(trim_layer(lc) for lc in cache.layer_caches)
    return ModelCache(layer_caches=trimmed_layers, final_norm=cache.final_norm)


def index_cache(cache: ModelCache, indices: Int[Array, "new_batch"]) -> ModelCache:
    """Select batch elements from a ModelCache.

    Note: cache position/count fields are shared scalars; this is only valid
    when all batch elements share the same history length.

    :param ModelCache cache: Cache to slice.
    :param Int[Array, "new_batch"] indices: Batch indices to select.
    :return ModelCache: Sliced cache.
    """

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
    k = int(min(top_k, logits.shape[-1]))
    values, _ = jax.lax.top_k(logits, k)
    thresh = values[:, -1][:, None]
    return jnp.where(logits < thresh, -jnp.inf, logits)


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
        k = int(min(top_k, logits.shape[-1]))
        sorted_logits, sorted_indices = jax.lax.top_k(logits, k)
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

    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if temperature != 0.0 and top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

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

    If return_cache is True and max_new_tokens == 1, the cache is advanced to
    include the generated token. If sampling is enabled (temperature > 0),
    the returned key is the next key to use for subsequent sampling calls.

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

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if max_new_tokens == 0:
        raise ValueError(
            "max_new_tokens must be > 0 to generate; Transformers treats this as an invalid length."
        )

    needs_rng = temperature != 0.0
    if needs_rng and key is None:
        raise ValueError("key is required when temperature > 0.")

    sample = _sample_fn(temperature, top_k, top_p)
    B, prompt_len = prompt_ids.shape

    if prompt_len == 0:
        if bos_token_id is None:
            bos_token_id = getattr(model.config, "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError(
                "prompt_ids is empty; provide bos_token_id or pass at least one token."
            )
        prompt_ids = jnp.full((B, 1), bos_token_id, dtype=prompt_ids.dtype)
        attention_mask = jnp.ones((B, 1), dtype=jnp.bool_)
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
        first_token = jnp.where(finished, eos_token_id, first_token)

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

    has_padding = not bool(jax.device_get(jnp.all(attention_mask)))
    needs_cache = cache is not None or return_cache or max_new_tokens > 1
    if has_padding and needs_cache:
        raise ValueError(
            "Cannot use cache with padded attention_mask. "
            "Provide unpadded prompts for cached generation."
        )

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


# Convenience JIT wrapper for generation with static knobs
generate_jit = eqx.filter_jit(_generate_core)
