"""Megalodon JAX - JAX/Equinox implementation of Megalodon for efficient long-context LLM."""

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.convert import load_from_pretrained, load_weights_from_torch
from megalodon_jax.model import (
    MegalodonBlock,
    MegalodonForCausalLM,
    MegalodonModel,
    ModelCache,
)
from megalodon_jax.types import AttentionCache, EMAState, LayerCache, NormState

__all__ = [
    # Config
    "MegalodonConfig",
    # Model classes
    "MegalodonBlock",
    "MegalodonModel",
    "MegalodonForCausalLM",
    # Cache types
    "ModelCache",
    "AttentionCache",
    "EMAState",
    "LayerCache",
    "NormState",
    # Weight conversion
    "load_weights_from_torch",
    "load_from_pretrained",
]
