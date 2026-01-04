"""Inference utilities: cache helpers, sampling, and generation."""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM
from megalodon_jax.types import AttentionCache, LayerCache, ModelCache, NormState

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def init_cache(
    config: MegalodonConfig,
    batch_size: int,
    dtype: jnp.dtype = jnp.float32,
    *,
    allocate_kv: bool = False,
) -> ModelCache:
    """Create an empty ModelCache for autoregressive generation.

    Args:
        config: Model configuration.
        batch_size: Batch size for cached tensors.
        dtype: Floating dtype for cached arrays.
        allocate_kv: If True, pre-allocate KV buffers; otherwise, leave None so the
            model allocates on first use (saves memory but changes shapes after
            the first call).

    Returns:
        ModelCache with per-layer caches and final norm state.
    """

    cache_len = config.effective_max_cache_len
    num_heads = config.num_heads
    head_dim = config.head_dim
    value_head_dim = config.value_head_dim

    def make_attn_cache() -> AttentionCache | None:
        if not allocate_kv:
            return None
        k = jnp.zeros((batch_size, cache_len, num_heads, head_dim), dtype=dtype)
        v = jnp.zeros((batch_size, cache_len, num_heads, value_head_dim), dtype=dtype)
        return AttentionCache(k=k, v=v, count=jnp.array(0, dtype=jnp.int32))

    layer_caches = []
    for _ in range(config.num_layers):
        layer_caches.append(
            LayerCache(
                attn=make_attn_cache(),
                norm=None,
                ema=None,
                position=jnp.array(0, dtype=jnp.int32),
            )
        )

    final_norm = NormState(
        count=jnp.zeros((batch_size,), dtype=jnp.int32),
        mean=jnp.zeros((batch_size, config.norm_num_groups), dtype=dtype),
        var=jnp.ones((batch_size, config.norm_num_groups), dtype=dtype),
    )

    return ModelCache(layer_caches=tuple(layer_caches), final_norm=final_norm)


def trim_cache(cache: ModelCache, max_len: int) -> ModelCache:
    """Trim KV cache entries to the most recent ``max_len`` tokens."""

    def trim_layer(layer_cache: LayerCache | None) -> LayerCache | None:
        if layer_cache is None or layer_cache.attn is None:
            return layer_cache
        attn = layer_cache.attn
        if attn.k.shape[1] <= max_len:
            return layer_cache
        trimmed = AttentionCache(
            k=attn.k[:, -max_len:],
            v=attn.v[:, -max_len:],
            count=attn.count,
        )
        return LayerCache(
            attn=trimmed,
            norm=layer_cache.norm,
            ema=layer_cache.ema,
            position=layer_cache.position,
        )

    trimmed_layers = tuple(trim_layer(lc) for lc in cache.layer_caches)
    return ModelCache(layer_caches=trimmed_layers, final_norm=cache.final_norm)


def index_cache(cache: ModelCache, indices: Int[Array, new_batch]) -> ModelCache:
    """Select batch elements from a ModelCache (useful for beam search)."""

    def index_array(x: Array | None) -> Array | None:
        if x is None:
            return None
        return x[indices]

    def index_layer(layer_cache: LayerCache | None) -> LayerCache | None:
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


def greedy_token(logits: Float[Array, "batch vocab"]) -> Int[Array, batch]:
    """Deterministic argmax sampling."""

    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _apply_top_k(
    logits: Float[Array, "batch vocab"], top_k: int | None
) -> Float[Array, "batch vocab"]:
    if top_k is None or top_k <= 0:
        return logits
    k = min(top_k, logits.shape[-1])
    # Threshold: k-th largest value per batch
    thresh = jnp.sort(logits, axis=-1)[:, -k][:, None]
    masked = jnp.where(logits < thresh, -jnp.inf, logits)
    return masked


def _apply_top_p(
    logits: Float[Array, "batch vocab"], top_p: float | None
) -> Float[Array, "batch vocab"]:
    if top_p is None or top_p >= 1.0:
        return logits

    # Sort descending
    sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative = jnp.cumsum(sorted_probs, axis=-1)

    # Mask tokens past the nucleus boundary; keep first above threshold
    sorted_mask = cumulative > top_p
    sorted_mask = jnp.concatenate(
        [jnp.zeros_like(sorted_mask[:, :1]), sorted_mask[:, :-1]], axis=-1
    )

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
) -> Int[Array, batch]:
    """Sample next token with temperature, top-k, and top-p."""

    if temperature == 0.0:
        return greedy_token(logits)

    work = logits.astype(jnp.float32)
    if temperature != 1.0:
        work = work / temperature

    work = _apply_top_k(work, top_k)
    work = _apply_top_p(work, top_p)

    # Softmax in float32, guard against all -inf after filtering
    probs = jax.nn.softmax(work, axis=-1)
    log_probs = jnp.log(probs + 1e-9)
    return jax.random.categorical(key, log_probs, axis=-1).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _sample_fn(
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> Callable[[Float[Array, "batch vocab"], PRNGKeyArray], Int[Array, batch]]:
    if temperature == 0.0:
        return lambda logits, _: greedy_token(logits)
    return functools.partial(sample_token, temperature=temperature, top_k=top_k, top_p=top_p)


def generate(
    model: MegalodonForCausalLM,
    prompt_ids: Int[Array, "batch prompt_len"],
    max_new_tokens: int,
    key: PRNGKeyArray,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    attention_mask: Bool[Array, "batch prompt_len"] | None = None,
    cache: ModelCache | None = None,
    return_cache: bool = False,
) -> tuple[Int[Array, "batch total_len"], ModelCache | None]:
    """Autoregressive generation using a fixed-shape scan."""

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    # Early exit
    if max_new_tokens == 0:
        return prompt_ids, cache

    sample = _sample_fn(temperature, top_k, top_p)
    B, prompt_len = prompt_ids.shape

    # Prefill
    logits, cache = model(
        prompt_ids,
        attention_mask=attention_mask,
        cache=cache,
        return_cache=True,
        deterministic=True,
    )

    key, subkey = jax.random.split(key)
    first_token = sample(logits[:, -1, :], subkey)

    finished = jnp.zeros((B,), dtype=jnp.bool_)
    if eos_token_id is not None:
        finished = first_token == eos_token_id
        first_token = jnp.where(finished, eos_token_id, first_token)

    def scan_step(
        carry: tuple[ModelCache, Int[Array, batch], PRNGKeyArray, Bool[Array, batch]],
        _,
    ) -> tuple[
        tuple[ModelCache, Int[Array, batch], PRNGKeyArray, Bool[Array, batch]],
        Int[Array, batch],
    ]:
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

    if max_new_tokens == 1:
        final_cache = cache if return_cache else None
        generated = first_token[:, None]
    else:
        init_carry = (cache, first_token, key, finished)
        final_carry, tokens_scan = jax.lax.scan(
            scan_step,
            init_carry,
            None,
            length=max_new_tokens - 1,
        )
        final_cache = final_carry[0] if return_cache else None
        generated = jnp.concatenate([first_token[:, None], tokens_scan.T], axis=1)

    result = jnp.concatenate([prompt_ids, generated], axis=1)
    return result, final_cache


# Convenience JIT wrapper for generation with static knobs
generate_jit = jax.jit(
    generate,
    static_argnames=(
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "eos_token_id",
        "return_cache",
    ),
)
