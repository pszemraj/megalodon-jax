"""Megalodon JAX - JAX/Equinox implementation of Megalodon for efficient long-context LLM."""

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.inference import (
    generate,
    generate_jit,
    greedy_token,
    index_cache,
    init_cache,
    sample_token,
    trim_cache,
)
from megalodon_jax.model import MegalodonBlock, MegalodonForCausalLM, MegalodonModel
from megalodon_jax.types import AttentionCache, EMAState, LayerCache, ModelCache, NormState

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
    # Inference utilities
    "init_cache",
    "trim_cache",
    "index_cache",
    "greedy_token",
    "sample_token",
    "generate",
    "generate_jit",
]

# Weight conversion utilities require torch.
# Import explicitly: from megalodon_jax.convert import load_from_pretrained, save_safetensors
