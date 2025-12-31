"""Megalodon JAX layers."""

from megalodon_jax.layers.attention import (
    ChunkedAttention,
    MegalodonAttention,
    NormalizedFFN,
    attention_multi_chunk,
    attention_single_chunk,
)
from megalodon_jax.layers.complex_ema import ComplexEMA
from megalodon_jax.layers.norms import RMSNorm
from megalodon_jax.layers.rotary import RotaryEmbedding
from megalodon_jax.layers.timestep_norm import TimestepNorm

__all__ = [
    # Attention
    "attention_single_chunk",
    "attention_multi_chunk",
    "ChunkedAttention",
    "MegalodonAttention",
    "NormalizedFFN",
    # Core layers
    "ComplexEMA",
    "RMSNorm",
    "RotaryEmbedding",
    "TimestepNorm",
]
