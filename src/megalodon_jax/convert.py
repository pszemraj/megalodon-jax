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
"""Strict conversion for the exact released PyTorch Megalodon keyspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array

from megalodon_jax.checkpoint import _apply_parameters, _require_exact_keys
from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM

StateDict = dict[str, torch.Tensor]


def _torch_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert an array-like value to a writable PyTorch tensor.

    Floating-point inputs first pass through NumPy float32 to preserve the
    released checkpoint representation.

    :param Any value: Array-like value to convert.
    :param torch.dtype | None dtype: Optional destination dtype.
    :return torch.Tensor: Writable CPU tensor.
    """
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
    if not array.flags.writeable:
        array = array.copy()
    tensor = torch.from_numpy(array)
    return tensor if dtype is None else tensor.to(dtype)


def _jax_float(value: torch.Tensor, name: str) -> Array:
    """Convert an upstream tensor to a JAX float32 array.

    :param torch.Tensor value: Upstream tensor to convert.
    :param str name: Upstream key used in validation errors.
    :raises TypeError: If ``value`` is not a PyTorch tensor.
    :return Array: JAX float32 array on the default device.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"upstream value {name!r} is not a torch.Tensor")
    return jnp.asarray(value.detach().float().cpu().numpy(), dtype=jnp.float32)


def _rope_frequencies(config: MegalodonConfig) -> np.ndarray:
    """Construct the released adjacent-pair RoPE frequency vector.

    :param MegalodonConfig config: Model configuration defining head size and RoPE base.
    :return np.ndarray: Float32 frequencies with shape ``(head_dim / 2,)``.
    """
    half = config.head_dim // 2
    exponent = np.arange(half, dtype=np.float32) / half
    return np.power(np.float32(config.effective_rope_base), -exponent).astype(np.float32)


def export_upstream_state_dict(
    model: MegalodonForCausalLM,
    *,
    dtype: torch.dtype | None = None,
) -> StateDict:
    """Export an exact world-size-one released-source state dictionary.

    :param MegalodonForCausalLM model: JAX model to export.
    :param torch.dtype | None dtype: Optional dtype for ordinary projection tensors.
        Numerically sensitive parameters remain float32.
    :return StateDict: Released-source keyspace backed by CPU PyTorch tensors.
    """
    config = model.config

    def ordinary(value: Any) -> torch.Tensor:
        """Convert an ordinary parameter using the requested export dtype.

        :param Any value: Parameter value to convert.
        :return torch.Tensor: Converted CPU tensor.
        """
        return _torch_tensor(value, dtype=dtype)

    def fp32(value: Any) -> torch.Tensor:
        """Convert a numerically sensitive parameter to float32.

        :param Any value: Parameter value to convert.
        :return torch.Tensor: Float32 CPU tensor.
        """
        return _torch_tensor(value, dtype=torch.float32)

    state: StateDict = {
        "embed.weight": ordinary(model.model.embed.weight),
        "rope.freqs": torch.from_numpy(_rope_frequencies(config)),
    }
    for index, layer in enumerate(model.model.layers):
        attn = layer.attn
        ap = f"layers.{index}.mega"
        state[f"{ap}.timenorm.prior_count"] = torch.tensor(
            attn.timenorm.prior_count,
            dtype=torch.int64,
        )
        prior_mean = (
            jnp.zeros(attn.timenorm.num_groups, dtype=jnp.float32)
            if attn.timenorm.prior_mean is None
            else attn.timenorm.prior_mean
        )
        prior_logv = (
            jnp.zeros(attn.timenorm.num_groups, dtype=jnp.float32)
            if attn.timenorm.prior_logv is None
            else attn.timenorm.prior_logv
        )
        state[f"{ap}.timenorm.prior_mean"] = fp32(prior_mean)
        state[f"{ap}.timenorm.prior_logv"] = fp32(prior_logv)
        state[f"{ap}.timenorm.weight"] = fp32(attn.timenorm.weight)
        state[f"{ap}.timenorm.bias"] = fp32(attn.timenorm.bias)

        cema = attn.cema
        state[f"{ap}.cema.alpha"] = fp32(cema.alpha)
        state[f"{ap}.cema.delta"] = fp32(cema.delta)
        state[f"{ap}.cema.theta"] = fp32(cema.theta)
        state[f"{ap}.cema.gamma"] = fp32(jnp.stack([cema.gamma_real, cema.gamma_imag], axis=-1))
        state[f"{ap}.cema.omega"] = fp32(cema.omega[:, None])

        if attn.rmsnorm.gamma is not None:
            state[f"{ap}.rmsnorm.weight"] = fp32(attn.rmsnorm.gamma)
        for name in ("wz", "wv", "wr", "wh1", "wh2"):
            projection = getattr(attn, name)
            state[f"{ap}.{name}.weight"] = ordinary(projection.weight)
            if projection.bias is not None:
                state[f"{ap}.{name}.bias"] = ordinary(projection.bias)
        state[f"{ap}.gamma"] = fp32(attn.gamma)
        state[f"{ap}.beta"] = fp32(attn.beta)

        ffn = layer.ffn
        fp = f"layers.{index}.nffn"
        if ffn.norm.weight is not None:
            state[f"{fp}.norm.weight"] = fp32(ffn.norm.weight)
            if ffn.norm.bias is None:
                raise ValueError(f"{fp}.norm has weight but no required bias")
            state[f"{fp}.norm.bias"] = fp32(ffn.norm.bias)
        state[f"{fp}.fc1.weight"] = ordinary(ffn.fc1.weight)
        state[f"{fp}.fc2.weight"] = ordinary(ffn.fc2.weight)
        if ffn.fc3 is not None:
            state[f"{fp}.fc3.weight"] = ordinary(ffn.fc3.weight)
        if ffn.alpha is not None:
            state[f"{fp}.alpha"] = fp32(ffn.alpha)

    final_norm = model.model.norm
    state["output.final_norm.prior_count"] = torch.tensor(
        final_norm.prior_count,
        dtype=torch.int64,
    )
    final_prior_mean = (
        jnp.zeros(final_norm.num_groups, dtype=jnp.float32)
        if final_norm.prior_mean is None
        else final_norm.prior_mean
    )
    final_prior_logv = (
        jnp.zeros(final_norm.num_groups, dtype=jnp.float32)
        if final_norm.prior_logv is None
        else final_norm.prior_logv
    )
    state["output.final_norm.prior_mean"] = fp32(final_prior_mean)
    state["output.final_norm.prior_logv"] = fp32(final_prior_logv)
    state["output.final_norm.weight"] = fp32(final_norm.weight)
    state["output.final_norm.bias"] = fp32(final_norm.bias)
    output_weight = model.model.embed.weight if model.tied else model.lm_head.weight
    state["output.output.weight"] = ordinary(output_weight).clone()
    return state


def _validate_prior(
    state: StateDict,
    prefix: str,
    expected_groups: int,
) -> None:
    """Validate the zero-prior convention used by released top-level models.

    :param StateDict state: Upstream state dictionary.
    :param str prefix: TimestepNorm key prefix.
    :param int expected_groups: Required number of normalization groups.
    :raises ValueError: If the prior count, shapes, or values are incompatible.
    """
    count = state[f"{prefix}.prior_count"]
    if count.numel() != 1 or int(count.item()) != 0:
        raise ValueError(
            f"{prefix} uses learned prior_count; top-level released presets require zero"
        )
    for name in ("prior_mean", "prior_logv"):
        tensor = state[f"{prefix}.{name}"]
        if tuple(tensor.shape) != (expected_groups,):
            raise ValueError(
                f"{prefix}.{name} has shape {tuple(tensor.shape)}, expected {(expected_groups,)}"
            )
        if torch.count_nonzero(tensor).item() != 0:
            raise ValueError(f"{prefix}.{name} must be zero when prior_count is zero")


def load_upstream_state_dict(
    model: MegalodonForCausalLM,
    state_dict: StateDict,
) -> MegalodonForCausalLM:
    """Load an exact released world-size-one state dictionary, strictly.

    :param MegalodonForCausalLM model: Initialized JAX model defining the target schema.
    :param StateDict state_dict: Released-source state dictionary.
    :raises TypeError: If an upstream value is not a PyTorch tensor.
    :raises ValueError: If keys, tensor layouts, priors, or tied weights are incompatible.
    :return MegalodonForCausalLM: Model with converted upstream parameters.
    """
    expected = set(export_upstream_state_dict(model))
    actual = set(state_dict)
    _require_exact_keys(actual, expected, context="strict original-upstream key mismatch")

    config = model.config
    native: dict[str, Array] = {
        "model.embed.weight": _jax_float(state_dict["embed.weight"], "embed.weight"),
    }
    expected_freqs = _rope_frequencies(config)
    actual_freqs = state_dict["rope.freqs"].detach().float().cpu().numpy()
    if actual_freqs.shape != expected_freqs.shape or not np.allclose(
        actual_freqs,
        expected_freqs,
        atol=2e-6,
        rtol=2e-6,
    ):
        raise ValueError("upstream rope.freqs is incompatible with config RoPE coordinates")

    for index, layer in enumerate(model.model.layers):
        ap = f"layers.{index}.mega"
        jp = f"model.layers.{index}.attn"
        _validate_prior(state_dict, f"{ap}.timenorm", layer.attn.timenorm.num_groups)
        native[f"{jp}.timenorm.weight"] = _jax_float(
            state_dict[f"{ap}.timenorm.weight"],
            f"{ap}.timenorm.weight",
        )
        native[f"{jp}.timenorm.bias"] = _jax_float(
            state_dict[f"{ap}.timenorm.bias"],
            f"{ap}.timenorm.bias",
        )
        for name in ("alpha", "delta", "theta"):
            native[f"{jp}.cema.{name}"] = _jax_float(
                state_dict[f"{ap}.cema.{name}"],
                f"{ap}.cema.{name}",
            )
        gamma = _jax_float(state_dict[f"{ap}.cema.gamma"], f"{ap}.cema.gamma")
        expected_gamma = (config.model_dim, config.cema_ndim, 2)
        if gamma.shape != expected_gamma:
            raise ValueError(f"{ap}.cema.gamma has shape {gamma.shape}, expected {expected_gamma}")
        native[f"{jp}.cema.gamma_real"] = gamma[..., 0]
        native[f"{jp}.cema.gamma_imag"] = gamma[..., 1]
        omega = _jax_float(state_dict[f"{ap}.cema.omega"], f"{ap}.cema.omega")
        if omega.shape != (config.model_dim, 1):
            raise ValueError(f"{ap}.cema.omega must have shape {(config.model_dim, 1)}")
        native[f"{jp}.cema.omega"] = omega[:, 0]

        if layer.attn.rmsnorm.gamma is not None:
            native[f"{jp}.rmsnorm.gamma"] = _jax_float(
                state_dict[f"{ap}.rmsnorm.weight"],
                f"{ap}.rmsnorm.weight",
            )
        for name in ("wz", "wv", "wr", "wh1", "wh2"):
            native[f"{jp}.{name}.weight"] = _jax_float(
                state_dict[f"{ap}.{name}.weight"],
                f"{ap}.{name}.weight",
            )
            if getattr(layer.attn, name).bias is not None:
                native[f"{jp}.{name}.bias"] = _jax_float(
                    state_dict[f"{ap}.{name}.bias"],
                    f"{ap}.{name}.bias",
                )
        native[f"{jp}.gamma"] = _jax_float(state_dict[f"{ap}.gamma"], f"{ap}.gamma")
        native[f"{jp}.beta"] = _jax_float(state_dict[f"{ap}.beta"], f"{ap}.beta")

        fp = f"layers.{index}.nffn"
        jfp = f"model.layers.{index}.ffn"
        if layer.ffn.norm.weight is not None:
            native[f"{jfp}.norm.weight"] = _jax_float(
                state_dict[f"{fp}.norm.weight"],
                f"{fp}.norm.weight",
            )
            native[f"{jfp}.norm.bias"] = _jax_float(
                state_dict[f"{fp}.norm.bias"],
                f"{fp}.norm.bias",
            )
        for name in ("fc1", "fc2", "fc3"):
            projection = getattr(layer.ffn, name)
            if projection is not None:
                native[f"{jfp}.{name}.weight"] = _jax_float(
                    state_dict[f"{fp}.{name}.weight"],
                    f"{fp}.{name}.weight",
                )
        if layer.ffn.alpha is not None:
            native[f"{jfp}.alpha"] = _jax_float(
                state_dict[f"{fp}.alpha"],
                f"{fp}.alpha",
            )

    _validate_prior(state_dict, "output.final_norm", model.model.norm.num_groups)
    native["model.norm.weight"] = _jax_float(
        state_dict["output.final_norm.weight"],
        "output.final_norm.weight",
    )
    native["model.norm.bias"] = _jax_float(
        state_dict["output.final_norm.bias"],
        "output.final_norm.bias",
    )
    output = _jax_float(state_dict["output.output.weight"], "output.output.weight")
    if model.tied:
        # Released share_emb aliases one logical parameter. Although state_dict
        # serialization emits two tensors, both originate from that exact value;
        # accepting approximate equality would silently bless an untied artifact.
        if not np.array_equal(np.asarray(output), np.asarray(native["model.embed.weight"])):
            raise ValueError("tied upstream output and embedding weights must be bit-identical")
    else:
        native["lm_head.weight"] = output

    loaded, _ = _apply_parameters(model, native)
    return loaded


def _replicated(shards: list[torch.Tensor], key: str) -> torch.Tensor:
    """Validate and select a parameter replicated across model-parallel ranks.

    :param list[torch.Tensor] shards: Per-rank copies of the parameter.
    :param str key: Upstream key used in validation errors.
    :raises ValueError: If replicated values differ across ranks.
    :return torch.Tensor: The common replicated tensor.
    """
    first = shards[0]
    if any(not torch.equal(first, shard) for shard in shards[1:]):
        raise ValueError(f"replicated upstream shard key differs across ranks: {key}")
    return first


def _merge_axis(key: str) -> int | None:
    """Return the original-source model-parallel consolidation axis.

    Parameters constructed from a dimension divided by the model-parallel
    world size are concatenated. Full-width normalization and layer-scale
    parameters are replicated and must agree across ranks.

    :param str key: Released-source parameter key.
    :raises ValueError: If the key has no known sharding rule.
    :return int | None: Concatenation axis, or ``None`` for replicated values.
    """
    if key == "embed.weight" or key == "output.output.weight":
        return 1
    if key == "rope.freqs" or key.endswith(".prior_count"):
        return None
    # Attention RMSNorm, FFN LayerNorm, and FFN alpha are constructed at full
    # model_dim in the released source, so each model-parallel rank stores an
    # identical copy.
    if ".rmsnorm.weight" in key or ".nffn.norm." in key or key.endswith(".nffn.alpha"):
        return None
    if ".cema." in key:
        return 0
    if key.endswith(".wh2.weight") or key.endswith(".fc2.weight"):
        return 1
    if key.endswith(".mega.gamma") or key.endswith(".mega.beta"):
        return 1
    concat_zero = (
        # TimestepNorm divides features/groups by model-parallel world size.
        ".timenorm." in key
        or key.endswith(".wz.weight")
        or key.endswith(".wz.bias")
        or key.endswith(".wv.weight")
        or key.endswith(".wv.bias")
        or key.endswith(".wr.weight")
        or key.endswith(".wr.bias")
        or key.endswith(".wh1.weight")
        or key.endswith(".wh1.bias")
        or key.endswith(".fc1.weight")
        or key.endswith(".fc3.weight")
        # Final TimestepNorm is partitioned by feature/group just like the
        # per-layer TimestepNorm; prior_count was handled as replicated above.
        or key.startswith("output.final_norm.")
    )
    if concat_zero:
        return 0
    raise ValueError(f"unknown original-upstream model-parallel key: {key}")


def _load_consolidated_directory(path: Path) -> StateDict:
    """Load and merge an upstream consolidated model-parallel checkpoint.

    :param Path path: Directory containing consolidation metadata and rank files.
    :raises FileNotFoundError: If a required consolidated shard is absent.
    :raises TypeError: If a consolidated value is not a tensor.
    :raises ValueError: If metadata or shard structures are incompatible.
    :return StateDict: Merged world-size-one state dictionary.
    """
    config_path = path / "consolidate_config.json"
    if not config_path.is_file():
        raise ValueError(
            "raw FSDP checkpoints are unsupported; run the original upstream "
            "consolidation script first"
        )
    try:
        with config_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid {config_path.name}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{config_path.name} must contain a JSON object")
    if "model_parallel_size" not in metadata:
        raise ValueError(f"{config_path.name} is missing required model_parallel_size")
    world_size = metadata["model_parallel_size"]
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(
            f"{config_path.name} model_parallel_size must be a positive integer, got {world_size!r}"
        )
    files = (
        [path / "consolidated.pth"]
        if world_size == 1
        else [path / f"consolidated.{rank:02d}.pth" for rank in range(world_size)]
    )
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    shards = [torch.load(file, map_location="cpu", weights_only=True) for file in files]
    if any(not isinstance(shard, dict) for shard in shards):
        raise ValueError("consolidated checkpoint shard is not a state dictionary")
    key_sets = [set(shard) for shard in shards]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("consolidated checkpoint shards have different key sets")

    merged: StateDict = {}
    for key in sorted(key_sets[0]):
        values = [shard[key] for shard in shards]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"consolidated value {key!r} is not a tensor")
        axis = _merge_axis(key)
        merged[key] = _replicated(values, key) if axis is None else torch.cat(values, dim=axis)
    return merged


def load_upstream_checkpoint(
    path: str | Path,
    config: MegalodonConfig,
    *,
    key: Array,
) -> MegalodonForCausalLM:
    """Load a strict original-upstream file or consolidated directory.

    :param str | Path path: Upstream checkpoint file or consolidated directory.
    :param MegalodonConfig config: Configuration for the target JAX model.
    :param Array key: PRNG key used to initialize the target model.
    :raises FileNotFoundError: If the source or a required shard is absent.
    :raises TypeError: If an upstream checkpoint value has the wrong type.
    :raises ValueError: If the checkpoint schema is incompatible.
    :return MegalodonForCausalLM: Model loaded with upstream parameters.
    """
    source = Path(path)
    if source.is_dir():
        state = _load_consolidated_directory(source)
    else:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix == ".safetensors":
            raise ValueError(
                "metadata-free or Hugging Face SafeTensors are not original-upstream checkpoints"
            )
        state = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(not isinstance(name, str) for name in state):
            raise ValueError("original-upstream checkpoint is not a flat string-keyed state dict")
    model = MegalodonForCausalLM(config, key=key)
    return load_upstream_state_dict(model, state)


def _removed(name: str, replacement: str) -> NoReturn:
    """Raise the standard error for a removed ambiguous conversion API.

    :param str name: Removed function name.
    :param str replacement: Explicit replacement API.
    :raises RuntimeError: Always, with migration guidance.
    """
    raise RuntimeError(
        f"{name} was removed because it ambiguously targeted a different schema; use {replacement}"
    )


def convert_jax_to_torch(*args: Any, **kwargs: Any) -> NoReturn:
    """Reject the historical Hugging Face-shaped exporter.

    :param Any args: Ignored historical positional arguments.
    :param Any kwargs: Ignored historical keyword arguments.
    :raises RuntimeError: Always; use :func:`export_upstream_state_dict`.
    """
    del args, kwargs
    _removed("convert_jax_to_torch", "export_upstream_state_dict")


def load_weights_from_torch(*args: Any, **kwargs: Any) -> NoReturn:
    """Reject the historical Hugging Face-shaped loader.

    :param Any args: Ignored historical positional arguments.
    :param Any kwargs: Ignored historical keyword arguments.
    :raises RuntimeError: Always; use :func:`load_upstream_state_dict`.
    """
    del args, kwargs
    _removed("load_weights_from_torch", "load_upstream_state_dict")


def load_from_pretrained(*args: Any, **kwargs: Any) -> NoReturn:
    """Reject schema guessing between native and upstream checkpoints.

    :param Any args: Ignored historical positional arguments.
    :param Any kwargs: Ignored historical keyword arguments.
    :raises RuntimeError: Always; use an explicit native or upstream loader.
    """
    del args, kwargs
    _removed("load_from_pretrained", "load_checkpoint or load_upstream_checkpoint")


def save_safetensors(*args: Any, **kwargs: Any) -> NoReturn:
    """Reject the historical metadata-free SafeTensors writer.

    :param Any args: Ignored historical positional arguments.
    :param Any kwargs: Ignored historical keyword arguments.
    :raises RuntimeError: Always; use :func:`save_checkpoint`.
    """
    del args, kwargs
    _removed("save_safetensors", "save_checkpoint")
