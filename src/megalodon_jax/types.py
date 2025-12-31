"""Type definitions for Megalodon cache and state.

CRITICAL: All position/count fields are JAX arrays (not Python ints).
Python ints inside jax.lax.scan become static and cause recompilation on each new value.

All cache/state dataclasses are registered as JAX pytrees to work with jit/scan.
"""

from dataclasses import dataclass, field, fields

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int


def _register_pytree(cls):
    """Register a dataclass as a JAX pytree node."""

    def flatten(obj):
        """Flatten dataclass to (children, aux_data)."""
        children = tuple(getattr(obj, f.name) for f in fields(obj))
        return children, None

    def unflatten(aux_data, children):
        """Reconstruct dataclass from children."""
        return cls(*children)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


@_register_pytree
@dataclass
class AttentionCache:
    """Cache for streaming attention.

    Stores key/value tensors and a count of tokens processed.
    """

    k: Float[Array, "batch seq heads head_dim"]
    v: Float[Array, "batch seq heads value_head_dim"]
    count: Int[Array, ""]  # JAX scalar - total tokens seen

    @property
    def length(self) -> int:
        """Number of cached timesteps."""
        return self.k.shape[1]

    @property
    def start_index(self) -> Int[Array, ""]:
        """Absolute position of first cached token."""
        return self.count - self.k.shape[1]


@_register_pytree
@dataclass
class NormState:
    """Running statistics for TimestepNorm (Welford's algorithm).

    Tracks cumulative count, mean, and variance per group for streaming
    normalization that only uses past context.
    """

    count: Int[Array, "batch"]  # tokens seen per batch element
    mean: Float[Array, "batch groups"]
    var: Float[Array, "batch groups"]


@_register_pytree
@dataclass
class EMAState:
    """Complex exponential moving average hidden state.

    The EMA module maintains a complex hidden state that accumulates
    information across the sequence via recurrence: h[t] = q*h[t-1] + p*x[t]
    """

    h: Complex[Array, "batch dim ndim"]


def _default_position() -> Int[Array, ""]:
    """Create default position counter as JAX scalar."""
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
