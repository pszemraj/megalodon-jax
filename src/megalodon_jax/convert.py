"""Weight conversion utilities for PyTorch ↔ JAX.

This module provides functions to load PyTorch Megalodon weights into JAX models.
"""

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM

# Type alias for state dict
StateDict = dict[str, Any]


def _to_jax(tensor: Any) -> Array:
    """Convert a PyTorch tensor to JAX array."""
    import torch

    if isinstance(tensor, torch.Tensor):
        return jnp.array(tensor.detach().cpu().numpy())
    return jnp.array(tensor)


def load_weights_from_torch(
    model: MegalodonForCausalLM,
    state_dict: StateDict,
) -> MegalodonForCausalLM:
    """Load PyTorch weights into a JAX MegalodonForCausalLM model.

    The function maps PyTorch state dict keys to JAX model paths and uses
    eqx.tree_at to update the model with loaded weights.

    Weight mapping:
        - model.embed.weight → model.model.embed.weight
        - model.layers.{i}.attn.* → model.model.layers[i].attn.*
        - model.layers.{i}.ffn.* → model.model.layers[i].ffn.*
        - model.norm.* → model.model.norm.*
        - lm_head.weight → (skipped, tied to embed)

    Note: PyTorch nn.Linear stores weight as (out_features, in_features),
    which matches Equinox convention. No transpose is needed.

    Args:
        model: Initialized JAX MegalodonForCausalLM model.
        state_dict: PyTorch model state_dict (from model.state_dict()).

    Returns:
        MegalodonForCausalLM with loaded weights.

    Raises:
        KeyError: If expected keys are missing from state_dict.
    """

    # Helper to get weight or raise informative error
    def get_weight(key: str) -> Array:
        if key not in state_dict:
            raise KeyError(f"Missing key in state_dict: {key}")
        return _to_jax(state_dict[key])

    # Helper to check if key exists
    def has_key(key: str) -> bool:
        return key in state_dict

    # Embedding
    model = eqx.tree_at(
        lambda m: m.model.embed.weight,
        model,
        get_weight("model.embed.weight"),
    )

    # Layers
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # ----- Attention Block -----
        attn_prefix = f"{prefix}.attn"

        # TimestepNorm
        if has_key(f"{attn_prefix}.timenorm.weight"):
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].attn.timenorm.weight,
                model,
                get_weight(f"{attn_prefix}.timenorm.weight"),
            )
        if has_key(f"{attn_prefix}.timenorm.bias"):
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].attn.timenorm.bias,
                model,
                get_weight(f"{attn_prefix}.timenorm.bias"),
            )

        # ComplexEMA
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.cema.alpha,
            model,
            get_weight(f"{attn_prefix}.cema.alpha"),
        )
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.cema.delta,
            model,
            get_weight(f"{attn_prefix}.cema.delta"),
        )
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.cema.theta,
            model,
            get_weight(f"{attn_prefix}.cema.theta"),
        )
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.cema.gamma_real,
            model,
            get_weight(f"{attn_prefix}.cema.gamma_real"),
        )
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.cema.gamma_imag,
            model,
            get_weight(f"{attn_prefix}.cema.gamma_imag"),
        )
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.cema.omega,
            model,
            get_weight(f"{attn_prefix}.cema.omega"),
        )

        # RMSNorm
        if has_key(f"{attn_prefix}.rmsnorm.gamma"):
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].attn.rmsnorm.gamma,
                model,
                get_weight(f"{attn_prefix}.rmsnorm.gamma"),
            )

        # Linear projections
        for proj_name in ["wz", "wv", "wr", "wh1", "wh2"]:
            model = eqx.tree_at(
                lambda m, idx=i, pn=proj_name: getattr(m.model.layers[idx].attn, pn).weight,
                model,
                get_weight(f"{attn_prefix}.{proj_name}.weight"),
            )
            if has_key(f"{attn_prefix}.{proj_name}.bias"):
                model = eqx.tree_at(
                    lambda m, idx=i, pn=proj_name: getattr(m.model.layers[idx].attn, pn).bias,
                    model,
                    get_weight(f"{attn_prefix}.{proj_name}.bias"),
                )

        # Per-head affine (gamma, beta)
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.gamma,
            model,
            get_weight(f"{attn_prefix}.gamma"),
        )
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.beta,
            model,
            get_weight(f"{attn_prefix}.beta"),
        )

        # ChunkedAttention (inner) - rotary embedding
        if has_key(f"{attn_prefix}.inner.rope.inv_freq"):
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].attn.inner.rotary.inv_freq,
                model,
                get_weight(f"{attn_prefix}.inner.rope.inv_freq"),
            )

        # ----- FFN Block -----
        ffn_prefix = f"{prefix}.ffn"

        # LayerNorm
        if has_key(f"{ffn_prefix}.norm.weight"):
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].ffn.norm.weight,
                model,
                get_weight(f"{ffn_prefix}.norm.weight"),
            )
        if has_key(f"{ffn_prefix}.norm.bias"):
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].ffn.norm.bias,
                model,
                get_weight(f"{ffn_prefix}.norm.bias"),
            )

        # FC layers
        for fc_name in ["fc1", "fc2"]:
            model = eqx.tree_at(
                lambda m, idx=i, fn=fc_name: getattr(m.model.layers[idx].ffn, fn).weight,
                model,
                get_weight(f"{ffn_prefix}.{fc_name}.weight"),
            )
            if has_key(f"{ffn_prefix}.{fc_name}.bias"):
                model = eqx.tree_at(
                    lambda m, idx=i, fn=fc_name: getattr(m.model.layers[idx].ffn, fn).bias,
                    model,
                    get_weight(f"{ffn_prefix}.{fc_name}.bias"),
                )

        # fc3 for SwiGLU (optional)
        if has_key(f"{ffn_prefix}.fc3.weight"):
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].ffn.fc3.weight,
                model,
                get_weight(f"{ffn_prefix}.fc3.weight"),
            )
            if has_key(f"{ffn_prefix}.fc3.bias"):
                model = eqx.tree_at(
                    lambda m, idx=i: m.model.layers[idx].ffn.fc3.bias,
                    model,
                    get_weight(f"{ffn_prefix}.fc3.bias"),
                )

    # Final norm
    if has_key("model.norm.weight"):
        model = eqx.tree_at(
            lambda m: m.model.norm.weight,
            model,
            get_weight("model.norm.weight"),
        )
    if has_key("model.norm.bias"):
        model = eqx.tree_at(
            lambda m: m.model.norm.bias,
            model,
            get_weight("model.norm.bias"),
        )

    # lm_head.weight is skipped - tied to embedding

    return model


def load_from_pretrained(
    path: str | Path,
    config: MegalodonConfig | None = None,
    dtype: jnp.dtype = jnp.float32,
    *,
    key: Array,
) -> MegalodonForCausalLM:
    """Load a MegalodonForCausalLM from a PyTorch checkpoint.

    Args:
        path: Path to PyTorch checkpoint file (.pt, .bin, or .safetensors).
        config: Model configuration. If None, attempts to load from checkpoint.
        dtype: Target dtype for model parameters.
        key: PRNG key for model initialization.

    Returns:
        MegalodonForCausalLM with loaded weights.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ValueError: If config is None and cannot be inferred.
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load state dict
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

    # Handle nested state dict (e.g., from checkpoint with optimizer state)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    # Config must be provided for now
    if config is None:
        raise ValueError("config must be provided. Automatic config inference not yet implemented.")

    # Initialize JAX model
    model = MegalodonForCausalLM(config, key=key)

    # Load weights
    model = load_weights_from_torch(model, state_dict)

    return model
