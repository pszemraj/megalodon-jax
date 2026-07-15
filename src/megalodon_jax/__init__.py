"""Megalodon JAX - JAX/Equinox implementation of Megalodon for efficient long-context LLM."""

from megalodon_jax.checkpoint import (
    load_checkpoint,
    load_generation_state,
    load_inference_cache,
    load_partial_checkpoint,
    save_checkpoint,
    save_generation_state,
    save_inference_cache,
)
from megalodon_jax.config import MegalodonConfig
from megalodon_jax.inference import (
    generate,
    greedy_token,
    index_cache,
    init_cache,
    sample_token,
)
from megalodon_jax.model import MegalodonBlock, MegalodonForCausalLM, MegalodonModel
from megalodon_jax.types import (
    AttentionCache,
    EMAState,
    GenerationState,
    LayerCache,
    ModelCache,
    NormState,
)

__all__ = [
    # Config
    "MegalodonConfig",
    # Model classes
    "MegalodonBlock",
    "MegalodonModel",
    "MegalodonForCausalLM",
    # Cache types
    "ModelCache",
    "GenerationState",
    "AttentionCache",
    "EMAState",
    "LayerCache",
    "NormState",
    # Native persistence
    "save_checkpoint",
    "load_checkpoint",
    "load_partial_checkpoint",
    "save_inference_cache",
    "load_inference_cache",
    "save_generation_state",
    "load_generation_state",
    # Inference utilities
    "init_cache",
    "index_cache",
    "greedy_token",
    "sample_token",
    "generate",
]

# Original-upstream conversion requires torch and is imported explicitly from
# megalodon_jax.convert.
