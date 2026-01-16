"""Weight conversion utilities for PyTorch ↔ JAX.

This module provides functions to load PyTorch Megalodon weights into JAX models.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM

# Type alias for state dict
StateDict = dict[str, Any]


if TYPE_CHECKING:
    import torch
else:
    try:
        import torch  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        torch = None  # type: ignore[assignment]


def _require_torch() -> "torch":
    """Return the torch module or raise a helpful error if unavailable."""
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for PyTorch interop. "
            "Install megalodon-jax with the dev extra or add torch manually."
        )
    return torch


def _is_torch_tensor(tensor: Any) -> bool:
    """Check whether a value is a torch.Tensor without hard requiring torch."""
    if torch is None:
        return False
    return isinstance(tensor, torch.Tensor)


def _to_jax(tensor: Any) -> Array:
    """Convert a PyTorch tensor to a JAX array.

    :param Any tensor: Torch tensor or array-like input.
    :return Array: JAX array on the default device.
    """
    if _is_torch_tensor(tensor):
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
            f"Check that config matches the checkpoint."
        )


def _jax_to_numpy(x: Any) -> np.ndarray:
    """Host-copy a JAX array (or tensor-like) to numpy.

    In JAX 0.8+, np.asarray() handles device transfer automatically.

    :param Any x: JAX array or array-like input.
    :return np.ndarray: Host numpy array.
    """
    return np.asarray(x)


def _to_torch_tensor(x: Any, *, dtype: "torch.dtype | None" = None) -> "torch.Tensor":
    """Convert an array-like object into a torch tensor with safe host dtype handling.

    :param Any x: JAX/NumPy array-like input.
    :param torch.dtype | None dtype: Optional target dtype for the tensor.
    :return torch.Tensor: Torch tensor on CPU.
    """
    torch_mod = _require_torch()

    arr = _jax_to_numpy(x)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags.writeable:
        arr = arr.copy()
    tensor = torch_mod.from_numpy(arr)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def _to_numpy_array(x: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert an array-like object into a numpy array with safe dtype handling."""
    arr = _jax_to_numpy(x)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def _export_state_dict(
    model: MegalodonForCausalLM,
    *,
    to_array: Callable[[Any, Any | None], Any],
    dtype: Any | None,
    include_rope_inv_freq: bool,
    gamma_dtype: Any,
    clone_array: Callable[[Any], Any],
) -> StateDict:
    """Build a PyTorch-style state dict using a configurable array converter."""

    def convert(arr: Any, *, dtype_override: Any | None = None) -> Any:
        """Convert a leaf array with optional dtype override."""
        use_dtype = dtype_override if dtype_override is not None else dtype
        return to_array(arr, use_dtype)

    state_dict: StateDict = {}

    state_dict["model.embed.weight"] = convert(model.model.embed.weight)

    for i, layer in enumerate(model.model.layers):
        attn_prefix = f"model.layers.{i}.attn"
        ffn_prefix = f"model.layers.{i}.ffn"

        # TimestepNorm
        if layer.attn.timenorm.weight is not None:
            state_dict[f"{attn_prefix}.timenorm.weight"] = convert(layer.attn.timenorm.weight)
        if layer.attn.timenorm.bias is not None:
            state_dict[f"{attn_prefix}.timenorm.bias"] = convert(layer.attn.timenorm.bias)

        # CEMA
        state_dict[f"{attn_prefix}.cema.alpha"] = convert(layer.attn.cema.alpha)
        state_dict[f"{attn_prefix}.cema.delta"] = convert(layer.attn.cema.delta)
        state_dict[f"{attn_prefix}.cema.theta"] = convert(layer.attn.cema.theta)
        state_dict[f"{attn_prefix}.cema.gamma_real"] = convert(
            layer.attn.cema.gamma_real,
            dtype_override=gamma_dtype,
        )
        state_dict[f"{attn_prefix}.cema.gamma_imag"] = convert(
            layer.attn.cema.gamma_imag,
            dtype_override=gamma_dtype,
        )
        state_dict[f"{attn_prefix}.cema.omega"] = convert(layer.attn.cema.omega)

        # RMSNorm
        if layer.attn.rmsnorm.gamma is not None:
            state_dict[f"{attn_prefix}.rmsnorm.gamma"] = convert(layer.attn.rmsnorm.gamma)

        # Linear projections
        for proj_name in ["wz", "wv", "wr", "wh1", "wh2"]:
            proj = getattr(layer.attn, proj_name)
            state_dict[f"{attn_prefix}.{proj_name}.weight"] = convert(proj.weight)
            if proj.bias is not None:
                state_dict[f"{attn_prefix}.{proj_name}.bias"] = convert(proj.bias)

        state_dict[f"{attn_prefix}.gamma"] = convert(layer.attn.gamma)
        state_dict[f"{attn_prefix}.beta"] = convert(layer.attn.beta)

        # Rotary embedding (inv_freq is not a persistent buffer in PyTorch)
        if include_rope_inv_freq:
            state_dict[f"{attn_prefix}.inner.rope.inv_freq"] = convert(
                layer.attn.inner.rotary.inv_freq
            )

        # FFN norms
        if layer.ffn.norm.weight is not None:
            state_dict[f"{ffn_prefix}.norm.weight"] = convert(layer.ffn.norm.weight)
        if layer.ffn.norm.bias is not None:
            state_dict[f"{ffn_prefix}.norm.bias"] = convert(layer.ffn.norm.bias)

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
        state_dict["model.norm.weight"] = convert(model.model.norm.weight)
    if model.model.norm.bias is not None:
        state_dict["model.norm.bias"] = convert(model.model.norm.bias)

    # LM head - always emit for PyTorch strict loading compatibility
    if not model.tied and model.lm_head is not None:
        # Untied: use separate lm_head weights
        state_dict["lm_head.weight"] = convert(model.lm_head.weight)
    elif model.tied:
        # Tied: emit embed weight as lm_head.weight for PyTorch compatibility
        # Use clone to avoid SafeTensors shared memory error when saving
        state_dict["lm_head.weight"] = clone_array(convert(model.model.embed.weight))

    return state_dict


def convert_jax_to_torch(
    model: MegalodonForCausalLM,
    *,
    dtype: "torch.dtype | None" = None,
    include_rope_inv_freq: bool = False,
) -> StateDict:
    """Convert a JAX Megalodon model to a PyTorch-style state dict.

    Requires torch to be installed.

    :param MegalodonForCausalLM model: JAX MegalodonForCausalLM to export.
    :param torch.dtype | None dtype: Optional dtype for floating tensors.
    :param bool include_rope_inv_freq: Whether to include RoPE inv_freq buffers.
    :return StateDict: PyTorch-style state dict for export.
    """
    torch_mod = _require_torch()

    def to_torch(arr: Any, dtype_override: Any | None) -> Any:
        """Convert a JAX/NumPy array to a torch tensor with the export dtype."""
        return _to_torch_tensor(arr, dtype=dtype_override)

    return _export_state_dict(
        model,
        to_array=to_torch,
        dtype=dtype,
        include_rope_inv_freq=include_rope_inv_freq,
        gamma_dtype=torch_mod.float32,
        clone_array=lambda tensor: tensor.clone(),
    )


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

    return model


def load_from_pretrained(
    path: str | Path,
    config: MegalodonConfig | None = None,
    dtype: jnp.dtype = jnp.float32,
    *,
    key: Array,
) -> MegalodonForCausalLM:
    """Load a MegalodonForCausalLM from a PyTorch checkpoint.

    `.safetensors` files load without torch (via `safetensors.numpy`);
    other formats require torch for `torch.load`.

    :param str | Path path: Path to checkpoint file.
    :param MegalodonConfig | None config: Model configuration.
    :param jnp.dtype dtype: Target dtype for parameters.
    :param Array key: PRNG key for model initialization.
    :raises FileNotFoundError: If checkpoint file doesn't exist.
    :raises ValueError: If config is None and cannot be inferred.
    :return MegalodonForCausalLM: Model with loaded weights.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load state dict
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        state_dict = load_file(path)
    else:
        torch_mod = _require_torch()
        state_dict = torch_mod.load(path, map_location="cpu", weights_only=True)

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
            """Cast floating-point arrays to the target dtype.

            :param Any x: Array leaf to cast when floating point.
            :return Any: Casted array or original input.
            """
            if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(dtype)
            return x

        model = jax.tree.map(cast_arrays, model)

    return model


def save_safetensors(
    model: MegalodonForCausalLM,
    path: str | Path,
    *,
    dtype: "torch.dtype | None" = None,
    include_rope_inv_freq: bool = False,
) -> None:
    """Save a MegalodonForCausalLM to SafeTensors (PyTorch format).

    Uses the torch backend when available. Falls back to `safetensors.numpy`
    when torch is not installed; dtype overrides require torch.

    :param MegalodonForCausalLM model: JAX model to export.
    :param str | Path path: Output path for the .safetensors file.
    :param torch.dtype | None dtype: Optional dtype for exported tensors.
    :param bool include_rope_inv_freq: Whether to include RoPE inv_freq buffers.
    :return None: None.
    """
    if torch is None:
        if dtype is not None:
            raise ModuleNotFoundError(
                "torch is required to use dtype overrides when saving safetensors."
            )
        from safetensors.numpy import save_file

        state_dict = _export_state_dict(
            model,
            to_array=lambda arr, dtype_override: _to_numpy_array(arr, dtype=dtype_override),
            dtype=None,
            include_rope_inv_freq=include_rope_inv_freq,
            gamma_dtype=np.float32,
            clone_array=lambda array: np.array(array, copy=True),
        )
        save_file(state_dict, str(path))
        return

    from safetensors.torch import save_file

    state_dict = convert_jax_to_torch(
        model,
        dtype=dtype,
        include_rope_inv_freq=include_rope_inv_freq,
    )
    save_file(state_dict, str(path))
