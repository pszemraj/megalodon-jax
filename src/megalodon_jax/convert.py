# Copyright 2025 Peter Szemraj.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Weight conversion utilities for PyTorch ↔ JAX.

This module requires torch. Install with:
    pip install megalodon-jax[convert]

Usage:
    from megalodon_jax.convert import load_from_pretrained, save_safetensors
"""

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM
from megalodon_jax.precision import ensure_sensitive_param_dtype

StateDict = dict[str, Any]


def _to_jax(tensor: Any) -> Array:
    """Convert a PyTorch tensor to a JAX array.

    :param Any tensor: Torch tensor or array-like input.
    :return Array: JAX array on the default device.
    """
    if isinstance(tensor, torch.Tensor):
        return jnp.array(tensor.detach().cpu().numpy())
    return jnp.array(tensor)


def _validate_shape(
    loaded: Array,
    expected: tuple[int, ...],
    name: str,
) -> None:
    """Validate loaded weight shape matches expected shape.

    :param Array loaded: Array loaded from checkpoint.
    :param tuple[int, ...] expected: Expected shape based on config.
    :param str name: Parameter name for error reporting.
    :raises ValueError: If shapes don't match.
    :return None: None.
    """
    if loaded.shape != expected:
        raise ValueError(
            f"Shape mismatch for '{name}':\n"
            f"  Checkpoint: {loaded.shape}\n"
            f"  Expected:   {expected}\n"
            "Check that config matches the checkpoint."
        )


def _to_torch_tensor(x: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert an array-like object into a torch tensor.

    :param Any x: JAX/NumPy array-like input.
    :param torch.dtype | None dtype: Optional target dtype for the tensor.
    :return torch.Tensor: Torch tensor on CPU.
    """
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags.writeable:
        arr = arr.copy()
    tensor = torch.from_numpy(arr)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def convert_jax_to_torch(
    model: MegalodonForCausalLM,
    *,
    dtype: torch.dtype | None = None,
    include_rope_inv_freq: bool = False,
) -> StateDict:
    """Convert a JAX Megalodon model to a PyTorch-style state dict.

    Precision-sensitive parameters (CEMA params, norms, per-head Q/K affine) are
    exported in fp32 regardless of the requested dtype.

    :param MegalodonForCausalLM model: JAX MegalodonForCausalLM to export.
    :param torch.dtype | None dtype: Optional dtype for floating tensors.
    :param bool include_rope_inv_freq: Whether to include RoPE inv_freq buffers.
    :return StateDict: PyTorch-style state dict for export.
    """

    def convert(arr: Any, *, dtype_override: torch.dtype | None = None) -> torch.Tensor:
        """Convert a leaf array with optional dtype override."""
        use_dtype = dtype_override if dtype_override is not None else dtype
        return _to_torch_tensor(arr, dtype=use_dtype)

    state_dict: StateDict = {}

    state_dict["model.embed.weight"] = convert(model.model.embed.weight)

    for i, layer in enumerate(model.model.layers):
        attn_prefix = f"model.layers.{i}.attn"
        ffn_prefix = f"model.layers.{i}.ffn"

        # TimestepNorm
        if layer.attn.timenorm.weight is not None:
            state_dict[f"{attn_prefix}.timenorm.weight"] = convert(
                layer.attn.timenorm.weight,
                dtype_override=torch.float32,
            )
        if layer.attn.timenorm.bias is not None:
            state_dict[f"{attn_prefix}.timenorm.bias"] = convert(
                layer.attn.timenorm.bias,
                dtype_override=torch.float32,
            )

        # CEMA
        state_dict[f"{attn_prefix}.cema.alpha"] = convert(
            layer.attn.cema.alpha,
            dtype_override=torch.float32,
        )
        state_dict[f"{attn_prefix}.cema.delta"] = convert(
            layer.attn.cema.delta,
            dtype_override=torch.float32,
        )
        state_dict[f"{attn_prefix}.cema.theta"] = convert(
            layer.attn.cema.theta,
            dtype_override=torch.float32,
        )
        state_dict[f"{attn_prefix}.cema.gamma_real"] = convert(
            layer.attn.cema.gamma_real,
            dtype_override=torch.float32,
        )
        state_dict[f"{attn_prefix}.cema.gamma_imag"] = convert(
            layer.attn.cema.gamma_imag,
            dtype_override=torch.float32,
        )
        state_dict[f"{attn_prefix}.cema.omega"] = convert(
            layer.attn.cema.omega,
            dtype_override=torch.float32,
        )

        # RMSNorm
        if layer.attn.rmsnorm.gamma is not None:
            state_dict[f"{attn_prefix}.rmsnorm.gamma"] = convert(
                layer.attn.rmsnorm.gamma,
                dtype_override=torch.float32,
            )

        # Linear projections
        for proj_name in ["wz", "wv", "wr", "wh1", "wh2"]:
            proj = getattr(layer.attn, proj_name)
            state_dict[f"{attn_prefix}.{proj_name}.weight"] = convert(proj.weight)
            if proj.bias is not None:
                state_dict[f"{attn_prefix}.{proj_name}.bias"] = convert(proj.bias)

        state_dict[f"{attn_prefix}.gamma"] = convert(
            layer.attn.gamma,
            dtype_override=torch.float32,
        )
        state_dict[f"{attn_prefix}.beta"] = convert(
            layer.attn.beta,
            dtype_override=torch.float32,
        )

        # Rotary embedding (inv_freq is not a persistent buffer in PyTorch)
        if include_rope_inv_freq:
            state_dict[f"{attn_prefix}.inner.rope.inv_freq"] = convert(
                layer.attn.inner.rotary.inv_freq
            )

        # FFN norms
        if layer.ffn.norm.weight is not None:
            state_dict[f"{ffn_prefix}.norm.weight"] = convert(
                layer.ffn.norm.weight,
                dtype_override=torch.float32,
            )
        if layer.ffn.norm.bias is not None:
            state_dict[f"{ffn_prefix}.norm.bias"] = convert(
                layer.ffn.norm.bias,
                dtype_override=torch.float32,
            )

        # FFN linear layers
        for fc_name in ["fc1", "fc2", "fc3"]:
            fc = getattr(layer.ffn, fc_name, None)
            if fc is None:
                continue
            state_dict[f"{ffn_prefix}.{fc_name}.weight"] = convert(fc.weight)
            if fc.bias is not None:
                state_dict[f"{ffn_prefix}.{fc_name}.bias"] = convert(fc.bias)

    # Final norm
    if model.model.norm.weight is not None:
        state_dict["model.norm.weight"] = convert(
            model.model.norm.weight,
            dtype_override=torch.float32,
        )
    if model.model.norm.bias is not None:
        state_dict["model.norm.bias"] = convert(
            model.model.norm.bias,
            dtype_override=torch.float32,
        )

    # LM head - always emit for PyTorch strict loading compatibility
    if not model.tied and model.lm_head is not None:
        state_dict["lm_head.weight"] = convert(model.lm_head.weight)
    elif model.tied:
        # Clone to avoid SafeTensors shared memory error
        state_dict["lm_head.weight"] = convert(model.model.embed.weight).clone()

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

    :param MegalodonForCausalLM model: Initialized JAX model.
    :param StateDict state_dict: PyTorch state dict.
    :raises KeyError: If expected keys are missing.
    :return MegalodonForCausalLM: Model with loaded weights.
    """

    def get_weight(key: str) -> Array:
        """Retrieve a weight from the state dict or raise KeyError.

        :param str key: State dict key to fetch.
        :raises KeyError: If the key is missing.
        :return Array: JAX array for the parameter.
        """
        if key not in state_dict:
            raise KeyError(f"Missing key in state_dict: {key}")
        return _to_jax(state_dict[key])

    def has_key(key: str) -> bool:
        """Check whether a key exists in the state dict.

        :param str key: State dict key to query.
        :return bool: True if the key exists.
        """
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

    return ensure_sensitive_param_dtype(model)


def load_from_pretrained(
    path: str | Path,
    config: MegalodonConfig,
    dtype: jnp.dtype = jnp.float32,
    *,
    key: Array,
) -> MegalodonForCausalLM:
    """Load a MegalodonForCausalLM from a checkpoint file.

    Supports .safetensors, .pt, and .bin formats.

    :param str | Path path: Path to checkpoint file.
    :param MegalodonConfig config: Model configuration.
    :param jnp.dtype dtype: Target dtype for parameters (sensitive params stay fp32).
    :param Array key: PRNG key for model initialization.
    :raises FileNotFoundError: If checkpoint file doesn't exist.
    :return MegalodonForCausalLM: Model with loaded weights.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load state dict
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path), device="cpu")
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

    # Handle nested state dict (e.g., from checkpoint with optimizer state)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    # Initialize JAX model
    model = MegalodonForCausalLM(config, key=key)

    # Load weights
    model = load_weights_from_torch(model, state_dict)

    # Apply dtype cast if requested
    if dtype != jnp.float32:

        def cast_arrays(x: Any) -> Any:
            """Cast floating-point arrays to the target dtype.

            :param Any x: Array leaf to cast when floating point.
            :return Any: Casted array or original input.
            """
            if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(dtype)
            return x

        model = jax.tree.map(cast_arrays, model)
        model = ensure_sensitive_param_dtype(model)

    return ensure_sensitive_param_dtype(model)


def save_safetensors(
    model: MegalodonForCausalLM,
    path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    include_rope_inv_freq: bool = False,
) -> None:
    """Save a MegalodonForCausalLM to SafeTensors format.

    :param MegalodonForCausalLM model: JAX model to export.
    :param str | Path path: Output path for the .safetensors file.
    :param torch.dtype | None dtype: Optional dtype for exported tensors.
    :param bool include_rope_inv_freq: Whether to include RoPE inv_freq buffers.
    :return None: None.
    """
    from safetensors.torch import save_file

    state_dict = convert_jax_to_torch(
        model,
        dtype=dtype,
        include_rope_inv_freq=include_rope_inv_freq,
    )
    save_file(state_dict, str(path))
