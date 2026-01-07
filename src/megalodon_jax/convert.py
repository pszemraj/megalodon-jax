"""Weight conversion utilities for PyTorch ↔ JAX.

This module provides functions to load PyTorch Megalodon weights into JAX models.
"""

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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


def _validate_shape(
    loaded: Array,
    expected: tuple[int, ...],
    name: str,
) -> None:
    """Validate loaded weight shape matches expected shape.

    Args:
        loaded: Array loaded from checkpoint.
        expected: Expected shape based on model config.
        name: Parameter name for error message.

    Raises:
        ValueError: If shapes don't match.
    """
    if loaded.shape != expected:
        raise ValueError(
            f"Shape mismatch for '{name}':\n"
            f"  Checkpoint: {loaded.shape}\n"
            f"  Expected:   {expected}\n"
            f"Check that config matches the checkpoint."
        )


def _jax_to_numpy(x: Any) -> np.ndarray:
    """Host-copy a JAX array (or tensor-like) to numpy.

    In JAX 0.8+, np.asarray() handles device transfer automatically.
    """
    return np.asarray(x)


def convert_jax_to_torch(
    model: MegalodonForCausalLM,
) -> StateDict:
    """Convert a JAX Megalodon model to a PyTorch-style state dict."""
    import torch

    state_dict: StateDict = {}

    def to_torch(arr: Any) -> Any:
        return torch.from_numpy(_jax_to_numpy(arr))

    state_dict["model.embed.weight"] = to_torch(model.model.embed.weight)

    for i, layer in enumerate(model.model.layers):
        attn_prefix = f"model.layers.{i}.attn"
        ffn_prefix = f"model.layers.{i}.ffn"

        # TimestepNorm
        if layer.attn.timenorm.weight is not None:
            state_dict[f"{attn_prefix}.timenorm.weight"] = to_torch(layer.attn.timenorm.weight)
        if layer.attn.timenorm.bias is not None:
            state_dict[f"{attn_prefix}.timenorm.bias"] = to_torch(layer.attn.timenorm.bias)

        # CEMA
        state_dict[f"{attn_prefix}.cema.alpha"] = to_torch(layer.attn.cema.alpha)
        state_dict[f"{attn_prefix}.cema.delta"] = to_torch(layer.attn.cema.delta)
        state_dict[f"{attn_prefix}.cema.theta"] = to_torch(layer.attn.cema.theta)
        state_dict[f"{attn_prefix}.cema.gamma_real"] = to_torch(layer.attn.cema.gamma_real)
        state_dict[f"{attn_prefix}.cema.gamma_imag"] = to_torch(layer.attn.cema.gamma_imag)
        state_dict[f"{attn_prefix}.cema.omega"] = to_torch(layer.attn.cema.omega)

        # RMSNorm
        if layer.attn.rmsnorm.gamma is not None:
            state_dict[f"{attn_prefix}.rmsnorm.gamma"] = to_torch(layer.attn.rmsnorm.gamma)

        # Linear projections
        for proj_name in ["wz", "wv", "wr", "wh1", "wh2"]:
            proj = getattr(layer.attn, proj_name)
            state_dict[f"{attn_prefix}.{proj_name}.weight"] = to_torch(proj.weight)
            if proj.bias is not None:
                state_dict[f"{attn_prefix}.{proj_name}.bias"] = to_torch(proj.bias)

        state_dict[f"{attn_prefix}.gamma"] = to_torch(layer.attn.gamma)
        state_dict[f"{attn_prefix}.beta"] = to_torch(layer.attn.beta)

        # Rotary embedding
        state_dict[f"{attn_prefix}.inner.rope.inv_freq"] = to_torch(
            layer.attn.inner.rotary.inv_freq
        )

        # FFN norms
        if layer.ffn.norm.weight is not None:
            state_dict[f"{ffn_prefix}.norm.weight"] = to_torch(layer.ffn.norm.weight)
        if layer.ffn.norm.bias is not None:
            state_dict[f"{ffn_prefix}.norm.bias"] = to_torch(layer.ffn.norm.bias)

        # FFN linear layers
        for fc_name in ["fc1", "fc2", "fc3"]:
            fc = getattr(layer.ffn, fc_name, None)
            if fc is None:
                continue
            state_dict[f"{ffn_prefix}.{fc_name}.weight"] = to_torch(fc.weight)
            if fc.bias is not None:
                state_dict[f"{ffn_prefix}.{fc_name}.bias"] = to_torch(fc.bias)

    # Final norm
    if model.model.norm.weight is not None:
        state_dict["model.norm.weight"] = to_torch(model.model.norm.weight)
    if model.model.norm.bias is not None:
        state_dict["model.norm.bias"] = to_torch(model.model.norm.bias)

    # LM head - always emit for PyTorch strict loading compatibility
    if not model.tied and model.lm_head is not None:
        # Untied: use separate lm_head weights
        state_dict["lm_head.weight"] = to_torch(model.lm_head.weight)
    elif model.tied:
        # Tied: emit embed weight as lm_head.weight for PyTorch compatibility
        # PyTorch MegalodonForCausalLM always has lm_head param (aliased when tied)
        # Use .clone() to avoid SafeTensors shared memory error when saving
        state_dict["lm_head.weight"] = to_torch(model.model.embed.weight).clone()

    return state_dict


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

    def get_weight(key: str) -> Array:
        """Retrieve weight from state_dict or raise KeyError."""
        if key not in state_dict:
            raise KeyError(f"Missing key in state_dict: {key}")
        return _to_jax(state_dict[key])

    def has_key(key: str) -> bool:
        """Check if key exists in state_dict."""
        return key in state_dict

    cfg = model.config
    norm_affine = cfg.norm_affine
    swiglu = cfg.swiglu

    # Embedding - validate shape before loading
    embed_weight = get_weight("model.embed.weight")
    expected_embed = (cfg.vocab_size, cfg.model_dim)
    _validate_shape(embed_weight, expected_embed, "model.embed.weight")
    model = eqx.tree_at(
        lambda m: m.model.embed.weight,
        model,
        embed_weight,
    )

    # Validate layer count before loading layers
    num_layers = len(model.model.layers)
    ckpt_layers = sum(1 for k in state_dict if k.startswith("model.layers.") and ".cema.alpha" in k)
    if ckpt_layers != num_layers:
        raise ValueError(
            f"Layer count mismatch: checkpoint has {ckpt_layers} layers, "
            f"config expects {num_layers} layers"
        )

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # ----- Attention Block -----
        attn_prefix = f"{prefix}.attn"

        # TimestepNorm
        if norm_affine:
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].attn.timenorm.weight,
                model,
                get_weight(f"{attn_prefix}.timenorm.weight"),
            )
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].attn.timenorm.bias,
                model,
                get_weight(f"{attn_prefix}.timenorm.bias"),
            )
        else:
            if has_key(f"{attn_prefix}.timenorm.weight") or has_key(f"{attn_prefix}.timenorm.bias"):
                raise ValueError(
                    "Checkpoint provides TimestepNorm affine parameters but config.norm_affine=False."
                )

        # ComplexEMA - validate shapes on first layer
        alpha_weight = get_weight(f"{attn_prefix}.cema.alpha")
        if i == 0:
            expected_alpha = (cfg.model_dim, cfg.cema_ndim, 1)
            _validate_shape(alpha_weight, expected_alpha, f"{attn_prefix}.cema.alpha")
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.cema.alpha,
            model,
            alpha_weight,
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
        omega_weight = get_weight(f"{attn_prefix}.cema.omega")
        if i == 0:
            _validate_shape(omega_weight, (cfg.model_dim,), f"{attn_prefix}.cema.omega")
        model = eqx.tree_at(
            lambda m, idx=i: m.model.layers[idx].attn.cema.omega,
            model,
            omega_weight,
        )

        # RMSNorm
        if norm_affine:
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].attn.rmsnorm.gamma,
                model,
                get_weight(f"{attn_prefix}.rmsnorm.gamma"),
            )
        elif has_key(f"{attn_prefix}.rmsnorm.gamma"):
            raise ValueError(
                "Checkpoint provides RMSNorm affine parameters but config.norm_affine=False."
            )

        # Linear projections - validate wz shape on first layer
        for proj_name in ["wz", "wv", "wr", "wh1", "wh2"]:
            proj_weight = get_weight(f"{attn_prefix}.{proj_name}.weight")
            if i == 0 and proj_name == "wz":
                expected_wz = (cfg.z_dim, cfg.model_dim)
                _validate_shape(proj_weight, expected_wz, f"{attn_prefix}.wz.weight")
            model = eqx.tree_at(
                lambda m, idx=i, pn=proj_name: getattr(m.model.layers[idx].attn, pn).weight,
                model,
                proj_weight,
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
        if norm_affine:
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].ffn.norm.weight,
                model,
                get_weight(f"{ffn_prefix}.norm.weight"),
            )
            model = eqx.tree_at(
                lambda m, idx=i: m.model.layers[idx].ffn.norm.bias,
                model,
                get_weight(f"{ffn_prefix}.norm.bias"),
            )
        else:
            if has_key(f"{ffn_prefix}.norm.weight") or has_key(f"{ffn_prefix}.norm.bias"):
                raise ValueError(
                    "Checkpoint provides LayerNorm affine parameters but config.norm_affine=False."
                )

        # FC layers - validate fc1 shape on first layer
        for fc_name in ["fc1", "fc2"]:
            fc_weight = get_weight(f"{ffn_prefix}.{fc_name}.weight")
            if i == 0 and fc_name == "fc1":
                expected_fc1 = (cfg.ffn_hidden_dim, cfg.model_dim)
                _validate_shape(fc_weight, expected_fc1, f"{ffn_prefix}.fc1.weight")
            model = eqx.tree_at(
                lambda m, idx=i, fn=fc_name: getattr(m.model.layers[idx].ffn, fn).weight,
                model,
                fc_weight,
            )
            if has_key(f"{ffn_prefix}.{fc_name}.bias"):
                model = eqx.tree_at(
                    lambda m, idx=i, fn=fc_name: getattr(m.model.layers[idx].ffn, fn).bias,
                    model,
                    get_weight(f"{ffn_prefix}.{fc_name}.bias"),
                )

        # fc3 for SwiGLU (optional)
        if swiglu:
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
        elif has_key(f"{ffn_prefix}.fc3.weight") or has_key(f"{ffn_prefix}.fc3.bias"):
            raise ValueError("Checkpoint includes SwiGLU weights but config.swiglu=False.")

    # Final norm
    if norm_affine:
        model = eqx.tree_at(
            lambda m: m.model.norm.weight,
            model,
            get_weight("model.norm.weight"),
        )
        model = eqx.tree_at(
            lambda m: m.model.norm.bias,
            model,
            get_weight("model.norm.bias"),
        )
    elif has_key("model.norm.weight") or has_key("model.norm.bias"):
        raise ValueError(
            "Checkpoint provides final norm affine parameters but config.norm_affine=False."
        )

    # lm_head.weight - require for untied configs
    if not model.tied and model.lm_head is not None:
        if not has_key("lm_head.weight"):
            raise KeyError("Missing key in state_dict: lm_head.weight")
        lm_head_weight = get_weight("lm_head.weight")
        lm_out = cfg.vocab_size if cfg.output_size == -1 else cfg.output_size
        expected_lm_head = (lm_out, cfg.model_dim)
        _validate_shape(lm_head_weight, expected_lm_head, "lm_head.weight")
        model = eqx.tree_at(
            lambda m: m.lm_head.weight,
            model,
            lm_head_weight,
        )

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

    # Apply dtype cast if requested
    if dtype != jnp.float32:

        def cast_arrays(x: Any) -> Any:
            """Cast floating point arrays to target dtype."""
            if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(dtype)
            return x

        model = jax.tree.map(cast_arrays, model)

    return model


def save_safetensors(
    model: MegalodonForCausalLM,
    path: str | Path,
) -> None:
    """Save a MegalodonForCausalLM to SafeTensors (PyTorch format)."""
    from safetensors.torch import save_file

    state_dict = convert_jax_to_torch(model)
    save_file(state_dict, str(path))
