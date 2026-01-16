"""Type definitions for Megalodon cache and state.

CRITICAL: All position/count fields are JAX arrays (not Python ints).
Python ints inside jax.lax.scan become static and cause recompilation on each new value.

All cache/state dataclasses are registered as JAX pytrees to work with jit/scan.
"""

from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int

T = TypeVar("T")


def _register_pytree(cls: type[T]) -> type[T]:
    """Register a dataclass as a JAX pytree node.

    This decorator enables JAX transformations (jit, vmap, scan) to work with
    the decorated dataclass by defining how to flatten and unflatten it.

    :param type[T] cls: Dataclass type to register.
    :return type[T]: The class registered as a pytree node.
    """

    def flatten(obj: T) -> tuple[tuple[Any, ...], None]:
        """Flatten dataclass to (children, aux_data).

        :param T obj: Dataclass instance to flatten.
        :return tuple[tuple[Any, ...], None]: Field values and empty aux data.
        """
        children = tuple(getattr(obj, f.name) for f in fields(obj))
        return children, None

    def unflatten(aux_data: None, children: tuple[Any, ...]) -> T:
        """Reconstruct dataclass from children.

        :param None aux_data: Unused auxiliary data (always None).
        :param tuple[Any, ...] children: Field values in declaration order.
        :return T: Reconstructed dataclass instance.
        """
        return cls(*children)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


@_register_pytree
@dataclass
class AttentionCache:
    """Cache for streaming attention.

    Stores key/value tensors in fixed-capacity buffers.
    Use `count` for the total tokens processed (absolute position after last token).
    Buffer capacity is `k.shape[1]`; validity is inferred from `count` and ring
    indexing rather than an explicit mask. The buffer is treated as a circular
    ring indexed by absolute position modulo the cache capacity.

    Note: Unlike the PyTorch reference, this cache does not expose length/start_index
    properties because they would be misleading. In JAX, caches use fixed-size buffers
    with internal validity masking, so buffer capacity != valid cached length.
    """

    k: Float[Array, "batch seq heads head_dim"]
    v: Float[Array, "batch seq heads value_head_dim"]
    count: Int[Array, ""]  # JAX scalar - total tokens seen


@_register_pytree
@dataclass
class NormState:
    """Running statistics for TimestepNorm (Welford's algorithm).

    Tracks cumulative count, mean, and variance per group for streaming
    normalization that only uses past context. Mean/variance are stored in
    float32 for numerical stability, regardless of activation dtype.
    """

    count: Int[Array, "batch"]  # tokens seen per batch element
    mean: Float[Array, "batch groups"]
    var: Float[Array, "batch groups"]  # running variance estimate


@_register_pytree
@dataclass
class EMAState:
    """Complex exponential moving average hidden state.

    The EMA module maintains a complex hidden state that accumulates
    information across the sequence via recurrence: h[t] = q*h[t-1] + p*x[t]
    """

    h: Complex[Array, "batch dim ndim"]


def _default_position() -> Int[Array, ""]:
    """Create default position counter as JAX scalar.

    :return Int[Array, ""]: JAX scalar position counter initialized to zero.
    """
    return jnp.array(0, dtype=jnp.int32)


@_register_pytree
@dataclass
class LayerCache:
    """Combined cache for a single transformer layer.

    Groups attention cache, norm state, and EMA state together with
    the absolute position for RoPE computation.
    """

    attn: AttentionCache | None = None
    norm: NormState | None = None
    ema: EMAState | None = None
    position: Int[Array, ""] = field(default_factory=_default_position)


@_register_pytree
@dataclass
class ModelCache:
    """Full model cache: layer caches + final norm state.

    This cache structure holds all streaming state for the model:
    - One LayerCache per decoder layer (attention, norm, EMA state)
    - One NormState for the final TimestepNorm

    Note: layer_caches must be a tuple (not list) for JAX pytree compatibility.
    """

    layer_caches: tuple[LayerCache | None, ...]
    final_norm: NormState | None = None
