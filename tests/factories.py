"""Shared constructors for compact model configurations in tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from megalodon_jax import MegalodonConfig


def tiny_config(**overrides: Any) -> MegalodonConfig:
    """Create the canonical tiny configuration with explicit overrides."""
    config = MegalodonConfig(
        vocab_size=64,
        model_dim=64,
        num_layers=1,
        num_heads=2,
        z_dim=32,
        value_dim=64,
        ffn_hidden_dim=128,
        cema_ndim=4,
        chunk_size=8,
        norm_num_groups=8,
    )
    return replace(config, **overrides)


def floating_to_bf16(value: Any) -> Any:
    """Cast floating-point array leaves to bf16 and preserve other leaves."""
    if eqx.is_array(value) and jnp.issubdtype(value.dtype, jnp.floating):
        return value.astype(jnp.bfloat16)
    return value
