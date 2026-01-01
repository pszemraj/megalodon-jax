"""Megalodon JAX - JAX/Equinox implementation of Megalodon for efficient long-context LLM."""

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.types import AttentionCache, EMAState, LayerCache, NormState

__all__ = [
    "MegalodonConfig",
    "AttentionCache",
    "EMAState",
    "LayerCache",
    "NormState",
]
