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
"""Strict native checkpoint and inference-cache persistence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from safetensors import safe_open
from safetensors.flax import load_file, save_file

from megalodon_jax.cache import CACHE_INVARIANT_MESSAGE, validate_model_cache_host
from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM
from megalodon_jax.types import AttentionCache, EMAState, LayerCache, ModelCache, NormState

MODEL_FORMAT = "megalodon-jax"
MODEL_FORMAT_VERSION = "2"
CACHE_FORMAT = "megalodon-jax-cache"
CACHE_FORMAT_VERSION = "1"
ROPE_LAYOUT = "adjacent_pair"
NORMALIZATION_STORAGE = "plus_one"
BIAS_SCHEMA = "upstream"
INITIALIZER_SCHEMA = "split-boundary-internal-v1"
DTYPE_POLICY = "fp32-params-fp32-or-bf16-compute"
BF16_DTYPE_POLICY = "bf16-ordinary-params-fp32-sensitive"
_DTYPE_FIELDS = (
    "param_dtype",
    "compute_dtype",
    "accum_dtype",
    "attention_softmax_dtype",
    "loss_softmax_dtype",
)


def _dtype_policy(config: MegalodonConfig) -> str:
    """Return the checkpoint policy matching ordinary parameter storage.

    :param MegalodonConfig config: Configuration whose parameter dtype selects the policy.
    :raises ValueError: If the configuration uses an unsupported parameter dtype.
    :return str: Stable checkpoint metadata value.
    """
    if config.param_dtype == jnp.float32:
        return DTYPE_POLICY
    if config.param_dtype == jnp.bfloat16:
        return BF16_DTYPE_POLICY
    raise ValueError(f"unsupported checkpoint param_dtype: {config.param_dtype}")


def _config_dict(config: MegalodonConfig) -> dict[str, Any]:
    """Serialize a model configuration with portable dtype names.

    :param MegalodonConfig config: Configuration to serialize.
    :return dict[str, Any]: Configuration fields suitable for JSON encoding.
    """
    data = dataclasses.asdict(config)
    for field in _DTYPE_FIELDS:
        data[field] = str(jnp.dtype(data[field]))
    for field, value in data.items():
        if isinstance(value, np.integer):
            data[field] = int(value)
        elif isinstance(value, np.floating):
            data[field] = float(value)
    return data


def _config_json(config: MegalodonConfig) -> str:
    """Encode a model configuration as deterministic compact JSON.

    :param MegalodonConfig config: Configuration to encode.
    :return str: Deterministic JSON representation.
    """
    return json.dumps(_config_dict(config), sort_keys=True, separators=(",", ":"))


def _config_from_json(payload: str) -> MegalodonConfig:
    """Decode and validate a checkpoint configuration.

    :param str payload: JSON-encoded configuration.
    :raises ValueError: If the payload is invalid or uses an unsupported schema or dtype.
    :return MegalodonConfig: Validated model configuration.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint config_json is invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("checkpoint config_json must encode an object")
    for field in _DTYPE_FIELDS:
        if field not in data:
            raise ValueError(f"checkpoint config is missing dtype field {field!r}")
        name = data[field]
        if name == "float32":
            data[field] = jnp.float32
        elif name == "bfloat16":
            data[field] = jnp.bfloat16
        else:
            raise ValueError(f"checkpoint uses unsupported dtype {name!r}")
    try:
        return MegalodonConfig(**data)
    except TypeError as exc:
        raise ValueError("checkpoint config does not match the current v2 schema") from exc


def config_fingerprint(config: MegalodonConfig) -> str:
    """Return the stable fingerprint used to bind caches and checkpoints.

    :param MegalodonConfig config: Configuration to fingerprint.
    :return str: SHA-256 digest of the canonical configuration JSON.
    """
    return hashlib.sha256(_config_json(config).encode("utf-8")).hexdigest()


def _manifest(tensors: dict[str, Array]) -> str:
    """Fingerprint tensor names, shapes, and dtypes.

    :param dict[str, Array] tensors: Named tensors in the persisted artifact.
    :return str: SHA-256 digest of the canonical tensor schema.
    """
    rows = [
        (name, tuple(int(size) for size in value.shape), str(value.dtype))
        for name, value in sorted(tensors.items())
    ]
    payload = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_exact_keys(
    actual: Collection[str],
    expected: Collection[str],
    *,
    context: str,
) -> None:
    """Require an exact key set and report both sides of any mismatch.

    :param Collection[str] actual: Keys present in the input.
    :param Collection[str] expected: Required keys.
    :param str context: Description included in a mismatch error.
    :raises ValueError: If keys are missing or unexpected.
    """
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        raise ValueError(f"{context}: missing={missing}, unexpected={unexpected}")


def _require_tensor(
    tensors: dict[str, Array],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    *,
    context: str,
) -> Array:
    """Return a tensor only when its shape and dtype match exactly.

    :param dict[str, Array] tensors: Tensor mapping to inspect.
    :param str name: Tensor key to retrieve.
    :param tuple[int, ...] shape: Required tensor shape.
    :param jnp.dtype dtype: Required tensor dtype.
    :param str context: Description included in validation errors.
    :raises KeyError: If ``name`` is absent.
    :raises ValueError: If the tensor shape or dtype differs.
    :return Array: Validated tensor.
    """
    value = tensors[name]
    if value.shape != shape:
        raise ValueError(f"{context} {name} has shape {value.shape}, expected {shape}")
    expected_dtype = jnp.dtype(dtype)
    if value.dtype != expected_dtype:
        raise ValueError(f"{context} {name} has dtype {value.dtype}, expected {expected_dtype}")
    return value


def _load_manifest_tensors(
    path: Path,
    metadata: dict[str, str],
    *,
    manifest_key: str,
    error: str,
) -> dict[str, Array]:
    """Load tensors and validate the schema manifest recorded in metadata.

    :param Path path: SafeTensors file to load.
    :param dict[str, str] metadata: Artifact metadata.
    :param str manifest_key: Metadata key containing the expected manifest digest.
    :param str error: Error message for a manifest mismatch.
    :raises ValueError: If the loaded tensor manifest does not match the metadata.
    :return dict[str, Array]: Loaded tensor mapping.
    """
    tensors = load_file(str(path))
    if _manifest(tensors) != metadata.get(manifest_key):
        raise ValueError(error)
    return tensors


def _save_atomic_safetensors(
    destination: Path,
    tensors: dict[str, Array],
    metadata: dict[str, str],
    *,
    suffix_error: str,
) -> None:
    """Write a SafeTensors file atomically after validating its suffix.

    :param Path destination: Final SafeTensors path.
    :param dict[str, Array] tensors: Tensors to persist.
    :param dict[str, str] metadata: Metadata to persist.
    :param str suffix_error: Error message for a non-SafeTensors destination.
    :raises ValueError: If ``destination`` does not use the ``.safetensors`` suffix.
    """
    if destination.suffix != ".safetensors":
        raise ValueError(suffix_error)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


Entry = tuple[
    str,
    Array,
    Callable[[MegalodonForCausalLM], Array],
]


def _model_entries(model: MegalodonForCausalLM) -> list[Entry]:
    """Enumerate native checkpoint parameters and their replacement selectors.

    :param MegalodonForCausalLM model: Model whose parameters are enumerated.
    :return list[Entry]: Native name, current value, and Equinox selector triples.
    """
    entries: list[Entry] = []

    def add(name: str, value: Array, where: Callable[[MegalodonForCausalLM], Array]) -> None:
        """Append one native checkpoint entry.

        :param str name: Native checkpoint key.
        :param Array value: Current parameter value.
        :param Callable[[MegalodonForCausalLM], Array] where: Parameter selector.
        """
        entries.append((name, value, where))

    add("model.embed.weight", model.model.embed.weight, lambda root: root.model.embed.weight)
    for index, layer in enumerate(model.model.layers):
        attn = layer.attn
        ap = f"model.layers.{index}.attn"
        add(
            f"{ap}.timenorm.weight",
            attn.timenorm.weight,
            lambda root, i=index: root.model.layers[i].attn.timenorm.weight,
        )
        add(
            f"{ap}.timenorm.bias",
            attn.timenorm.bias,
            lambda root, i=index: root.model.layers[i].attn.timenorm.bias,
        )
        if attn.timenorm.prior_mean is not None:
            if attn.timenorm.prior_logv is None:
                raise ValueError(
                    f"{ap}.timenorm prior_mean requires a matching prior_logv parameter"
                )
            add(
                f"{ap}.timenorm.prior_mean",
                attn.timenorm.prior_mean,
                lambda root, i=index: root.model.layers[i].attn.timenorm.prior_mean,
            )
            add(
                f"{ap}.timenorm.prior_logv",
                attn.timenorm.prior_logv,
                lambda root, i=index: root.model.layers[i].attn.timenorm.prior_logv,
            )

        for name in ("alpha", "delta", "theta", "gamma_real", "gamma_imag", "omega"):
            add(
                f"{ap}.cema.{name}",
                getattr(attn.cema, name),
                lambda root, i=index, n=name: getattr(root.model.layers[i].attn.cema, n),
            )
        if attn.rmsnorm.gamma is not None:
            add(
                f"{ap}.rmsnorm.gamma",
                attn.rmsnorm.gamma,
                lambda root, i=index: root.model.layers[i].attn.rmsnorm.gamma,
            )
        for name in ("wz", "wv", "wr", "wh1", "wh2"):
            projection = getattr(attn, name)
            add(
                f"{ap}.{name}.weight",
                projection.weight,
                lambda root, i=index, n=name: getattr(root.model.layers[i].attn, n).weight,
            )
            if projection.bias is not None:
                add(
                    f"{ap}.{name}.bias",
                    projection.bias,
                    lambda root, i=index, n=name: getattr(root.model.layers[i].attn, n).bias,
                )
        add(
            f"{ap}.gamma",
            attn.gamma,
            lambda root, i=index: root.model.layers[i].attn.gamma,
        )
        add(
            f"{ap}.beta",
            attn.beta,
            lambda root, i=index: root.model.layers[i].attn.beta,
        )

        ffn = layer.ffn
        fp = f"model.layers.{index}.ffn"
        if ffn.norm.weight is not None:
            if ffn.norm.bias is None:
                raise ValueError(f"{fp}.norm weight requires a matching bias parameter")
            add(
                f"{fp}.norm.weight",
                ffn.norm.weight,
                lambda root, i=index: root.model.layers[i].ffn.norm.weight,
            )
            add(
                f"{fp}.norm.bias",
                ffn.norm.bias,
                lambda root, i=index: root.model.layers[i].ffn.norm.bias,
            )
        for name in ("fc1", "fc2", "fc3"):
            projection = getattr(ffn, name)
            if projection is None:
                continue
            add(
                f"{fp}.{name}.weight",
                projection.weight,
                lambda root, i=index, n=name: getattr(root.model.layers[i].ffn, n).weight,
            )
        if ffn.alpha is not None:
            add(
                f"{fp}.alpha",
                ffn.alpha,
                lambda root, i=index: root.model.layers[i].ffn.alpha,
            )

    add("model.norm.weight", model.model.norm.weight, lambda root: root.model.norm.weight)
    add("model.norm.bias", model.model.norm.bias, lambda root: root.model.norm.bias)
    if model.lm_head is not None:
        add("lm_head.weight", model.lm_head.weight, lambda root: root.lm_head.weight)
    return entries


def model_state_dict(model: MegalodonForCausalLM) -> dict[str, Array]:
    """Return the canonical v2 native parameter mapping.

    :param MegalodonForCausalLM model: Model to serialize.
    :return dict[str, Array]: Native checkpoint parameter mapping.
    """
    return {name: value for name, value, _ in _model_entries(model)}


def _metadata(path: Path) -> dict[str, str]:
    """Read SafeTensors metadata from a file.

    :param Path path: SafeTensors file to inspect.
    :raises ValueError: If the file is not readable as SafeTensors.
    :return dict[str, str]: File metadata, or an empty mapping when absent.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with safe_open(str(path), framework="flax") as handle:
            metadata = handle.metadata()
    except Exception as exc:
        raise ValueError(f"{path} is not a readable SafeTensors file") from exc
    return {} if metadata is None else dict(metadata)


def _read_model_file(path: Path) -> tuple[dict[str, Array], dict[str, str]]:
    """Load a native checkpoint after validating its format and metadata.

    :param Path path: Native checkpoint path.
    :raises FileNotFoundError: If the checkpoint does not exist.
    :raises ValueError: If its format, metadata, configuration, or manifest is invalid.
    :return tuple[dict[str, Array], dict[str, str]]: Tensors and validated metadata.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = _metadata(path)
    if (
        metadata.get("format") != MODEL_FORMAT
        or metadata.get("format_version") != MODEL_FORMAT_VERSION
    ):
        raise ValueError(
            "incompatible or legacy JAX checkpoint: expected explicit "
            f"{MODEL_FORMAT} format version {MODEL_FORMAT_VERSION}"
        )
    required = {
        "config_json",
        "config_fingerprint",
        "parameter_manifest_sha256",
        "rope_layout",
        "normalization_storage",
        "bias_schema",
        "initializer_schema",
        "tying",
        "dtype_policy",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"checkpoint metadata is incomplete: missing {missing}")
    if metadata["rope_layout"] != ROPE_LAYOUT:
        raise ValueError("checkpoint RoPE convention is incompatible")
    if metadata["normalization_storage"] != NORMALIZATION_STORAGE:
        raise ValueError("checkpoint normalization storage is incompatible")
    if metadata["bias_schema"] != BIAS_SCHEMA:
        raise ValueError("checkpoint projection-bias schema is incompatible")
    if metadata["initializer_schema"] != INITIALIZER_SCHEMA:
        raise ValueError("checkpoint initializer schema is incompatible")
    checkpoint_config = _config_from_json(metadata["config_json"])
    if metadata["dtype_policy"] != _dtype_policy(checkpoint_config):
        raise ValueError("checkpoint dtype policy is incompatible")
    if config_fingerprint(checkpoint_config) != metadata["config_fingerprint"]:
        raise ValueError("checkpoint config fingerprint is invalid")
    expected_tying = "tied" if checkpoint_config.share_emb else "untied"
    if metadata["tying"] != expected_tying:
        raise ValueError("checkpoint tying metadata disagrees with its configuration")
    tensors = _load_manifest_tensors(
        path,
        metadata,
        manifest_key="parameter_manifest_sha256",
        error="checkpoint parameter manifest does not match its metadata",
    )
    return tensors, metadata


def apply_model_state_dict(
    model: MegalodonForCausalLM,
    tensors: dict[str, Array],
    *,
    include: Collection[str] | None = None,
) -> tuple[MegalodonForCausalLM, dict[str, Any]]:
    """Apply strict checkpoint tensors to a model.

    :param MegalodonForCausalLM model: Model providing parameter shapes and selectors.
    :param dict[str, Array] tensors: Native checkpoint tensor mapping.
    :param Collection[str] | None include: Optional explicit subset to restore.
    :raises ValueError: If keys, shapes, dtypes, or a partial selection are invalid.
    :return tuple[MegalodonForCausalLM, dict[str, Any]]: Updated model and restore report.
    """
    entries = _model_entries(model)
    expected = {name for name, _, _ in entries}
    available = set(tensors)
    selected = expected if include is None else set(include)
    if include is None:
        require_exact_keys(available, expected, context="strict checkpoint key mismatch")
    else:
        unknown = sorted(selected - expected)
        unavailable = sorted(selected - available)
        if unknown or unavailable:
            raise ValueError(
                f"partial restore selection is invalid: unknown={unknown}, unavailable={unavailable}"
            )

    restored: list[str] = []
    initialized: list[str] = []
    for name, template, where in entries:
        if name not in selected:
            initialized.append(name)
            continue
        value = _require_tensor(
            tensors,
            name,
            template.shape,
            template.dtype,
            context="checkpoint tensor",
        )
        model = eqx.tree_at(where, model, value)
        restored.append(name)
    return model, {"restored": restored, "initialized": initialized}


def save_checkpoint(model: MegalodonForCausalLM, path: str | Path) -> None:
    """Save a strict native v2 model checkpoint.

    :param MegalodonForCausalLM model: Model to save.
    :param str | Path path: Destination ending in ``.safetensors``.
    :raises ValueError: If the destination suffix is invalid.
    """
    destination = Path(path)
    tensors = model_state_dict(model)
    config_json = _config_json(model.config)
    metadata = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "config_json": config_json,
        "config_fingerprint": config_fingerprint(model.config),
        "parameter_manifest_sha256": _manifest(tensors),
        "rope_layout": ROPE_LAYOUT,
        "normalization_storage": NORMALIZATION_STORAGE,
        "bias_schema": BIAS_SCHEMA,
        "initializer_schema": INITIALIZER_SCHEMA,
        "tying": "tied" if model.tied else "untied",
        "dtype_policy": _dtype_policy(model.config),
    }
    _save_atomic_safetensors(
        destination,
        tensors,
        metadata,
        suffix_error="native checkpoints must use the .safetensors suffix",
    )


def load_checkpoint(
    path: str | Path,
    *,
    key: Array,
) -> MegalodonForCausalLM:
    """Load a strict native v2 model checkpoint.

    :param str | Path path: Native checkpoint path.
    :param Array key: PRNG key used to construct the target model.
    :raises FileNotFoundError: If the checkpoint does not exist.
    :raises ValueError: If the checkpoint is incompatible or malformed.
    :return MegalodonForCausalLM: Restored model.
    """
    tensors, metadata = _read_model_file(Path(path))
    config = _config_from_json(metadata["config_json"])
    model = MegalodonForCausalLM(config, key=key)
    model, _ = apply_model_state_dict(model, tensors)
    return model


def load_partial_checkpoint(
    path: str | Path,
    config: MegalodonConfig,
    include: Collection[str],
    *,
    key: Array,
) -> tuple[MegalodonForCausalLM, dict[str, Any]]:
    """Restore an explicit parameter allowlist into a freshly initialized model.

    :param str | Path path: Native checkpoint path.
    :param MegalodonConfig config: Configuration for the target model.
    :param Collection[str] include: Native parameter names to restore.
    :param Array key: PRNG key used to initialize parameters not restored.
    :raises FileNotFoundError: If the checkpoint does not exist.
    :raises ValueError: If the checkpoint or restore selection is invalid.
    :return tuple[MegalodonForCausalLM, dict[str, Any]]: Model and restore report.
    """
    tensors, metadata = _read_model_file(Path(path))
    model = MegalodonForCausalLM(config, key=key)
    model, report = apply_model_state_dict(model, tensors, include=include)
    source_fingerprint = metadata["config_fingerprint"]
    target_fingerprint = config_fingerprint(config)
    report.update(
        {
            "source_config_fingerprint": source_fingerprint,
            "target_config_fingerprint": target_fingerprint,
            "exact_config_match": source_fingerprint == target_fingerprint,
        }
    )
    return model, report


def _cache_tensors(cache: ModelCache) -> tuple[dict[str, Array], list[str]]:
    """Flatten a model cache into persisted tensors and presence records.

    :param ModelCache cache: Inference cache to flatten.
    :raises ValueError: If a layer position differs from its attention count.
    :return tuple[dict[str, Array], list[str]]: Tensor mapping and component presence list.
    """
    tensors: dict[str, Array] = {}
    present: list[str] = []
    for index, layer in enumerate(cache.layer_caches):
        if layer is None:
            continue
        prefix = f"layers.{index}"
        tensors[f"{prefix}.position"] = layer.position
        present.append(f"{prefix}.position")
        if layer.attn is not None:
            if not np.array_equal(np.asarray(layer.position), np.asarray(layer.attn.count)):
                raise ValueError(f"{prefix} position must equal attention count")
            tensors[f"{prefix}.attn.k"] = layer.attn.k
            tensors[f"{prefix}.attn.v"] = layer.attn.v
            tensors[f"{prefix}.attn.count"] = layer.attn.count
            present.append(f"{prefix}.attn")
        if layer.norm is not None:
            tensors[f"{prefix}.norm.count"] = layer.norm.count
            tensors[f"{prefix}.norm.mean"] = layer.norm.mean
            tensors[f"{prefix}.norm.var"] = layer.norm.var
            present.append(f"{prefix}.norm")
        if layer.ema is not None:
            tensors[f"{prefix}.ema.real"] = jnp.real(layer.ema.h)
            tensors[f"{prefix}.ema.imag"] = jnp.imag(layer.ema.h)
            present.append(f"{prefix}.ema")
    if cache.final_norm is not None:
        tensors["final_norm.count"] = cache.final_norm.count
        tensors["final_norm.mean"] = cache.final_norm.mean
        tensors["final_norm.var"] = cache.final_norm.var
        present.append("final_norm")
    return tensors, present


def save_inference_cache(
    cache: ModelCache,
    path: str | Path,
    config: MegalodonConfig,
) -> None:
    """Save an inference cache bound to an exact model configuration.

    :param ModelCache cache: Inference cache to save.
    :param str | Path path: Destination ending in ``.safetensors``.
    :param MegalodonConfig config: Exact configuration that owns the cache.
    :raises ValueError: If the cache is invalid or the destination suffix is wrong.
    """
    destination = Path(path)
    validate_model_cache_host(cache, config)
    tensors, present = _cache_tensors(cache)
    metadata = {
        "format": CACHE_FORMAT,
        "format_version": CACHE_FORMAT_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "present_json": json.dumps(present, separators=(",", ":")),
        "tensor_manifest_sha256": _manifest(tensors),
    }
    _save_atomic_safetensors(
        destination,
        tensors,
        metadata,
        suffix_error="cache files must use the .safetensors suffix",
    )


def load_inference_cache(
    path: str | Path,
    config: MegalodonConfig,
) -> ModelCache:
    """Load a same-schema inference cache for an exact configuration.

    :param str | Path path: Inference-cache SafeTensors path.
    :param MegalodonConfig config: Exact configuration expected by the caller.
    :raises ValueError: If metadata, tensors, cache structure, or configuration is invalid.
    :return ModelCache: Validated inference cache.
    """
    source = Path(path)
    metadata = _metadata(source)
    if (
        metadata.get("format") != CACHE_FORMAT
        or metadata.get("format_version") != CACHE_FORMAT_VERSION
    ):
        raise ValueError("incompatible or legacy inference cache")
    if metadata.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("cache configuration fingerprint mismatch")
    tensors = _load_manifest_tensors(
        source,
        metadata,
        manifest_key="tensor_manifest_sha256",
        error="cache tensor manifest mismatch",
    )
    try:
        present_payload = json.loads(metadata["present_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("cache presence metadata is invalid") from exc
    if (
        not isinstance(present_payload, list)
        or any(not isinstance(item, str) for item in present_payload)
        or len(present_payload) != len(set(present_payload))
    ):
        raise ValueError("cache presence metadata must be a unique string list")
    present = set(present_payload)
    allowed = {"final_norm"}
    for index in range(config.num_layers):
        prefix = f"layers.{index}"
        allowed.update(
            {
                f"{prefix}.position",
                f"{prefix}.attn",
                f"{prefix}.norm",
                f"{prefix}.ema",
            }
        )
    unknown = sorted(present - allowed)
    if unknown:
        raise ValueError(f"cache presence metadata has unknown entries: {unknown}")

    expected_tensor_keys: set[str] = set()
    for index in range(config.num_layers):
        prefix = f"layers.{index}"
        required_components = {f"{prefix}.attn", f"{prefix}.norm", f"{prefix}.ema"}
        components = present & required_components
        if components and f"{prefix}.position" not in present:
            raise ValueError(f"cache layer {index} components require a position tensor")
        if components and components != required_components:
            raise ValueError(CACHE_INVARIANT_MESSAGE)
        if f"{prefix}.position" in present:
            expected_tensor_keys.add(f"{prefix}.position")
        if f"{prefix}.attn" in present:
            expected_tensor_keys.update(
                {f"{prefix}.attn.k", f"{prefix}.attn.v", f"{prefix}.attn.count"}
            )
        if f"{prefix}.norm" in present:
            expected_tensor_keys.update(
                {f"{prefix}.norm.count", f"{prefix}.norm.mean", f"{prefix}.norm.var"}
            )
        if f"{prefix}.ema" in present:
            expected_tensor_keys.update({f"{prefix}.ema.real", f"{prefix}.ema.imag"})
    if "final_norm" in present:
        expected_tensor_keys.update({"final_norm.count", "final_norm.mean", "final_norm.var"})
    require_exact_keys(
        tensors,
        expected_tensor_keys,
        context="cache tensor keys disagree with presence metadata",
    )

    batch_size: int | None = None

    def bind_batch(name: str, size: int) -> None:
        """Bind or validate the common cache batch size.

        :param str name: Tensor name used in a mismatch error.
        :param int size: Tensor batch dimension.
        :raises ValueError: If ``size`` differs from an earlier tensor.
        """
        nonlocal batch_size
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError(f"cache batch mismatch at {name}: {size} != {batch_size}")

    def require(name: str, shape: tuple[int, ...], dtype: jnp.dtype) -> Array:
        """Retrieve one cache tensor with exact shape and dtype validation.

        :param str name: Tensor key to retrieve.
        :param tuple[int, ...] shape: Required tensor shape.
        :param jnp.dtype dtype: Required tensor dtype.
        :raises KeyError: If ``name`` is absent.
        :raises ValueError: If its shape or dtype differs.
        :return Array: Validated cache tensor.
        """
        return _require_tensor(tensors, name, shape, dtype, context="cache tensor")

    layers: list[LayerCache | None] = []
    for index in range(config.num_layers):
        prefix = f"layers.{index}"
        layer_entries = {
            f"{prefix}.position",
            f"{prefix}.attn",
            f"{prefix}.norm",
            f"{prefix}.ema",
        }
        if not present & layer_entries:
            layers.append(None)
            continue
        position = require(f"{prefix}.position", (), jnp.int32)
        attn = None
        if f"{prefix}.attn" in present:
            k = tensors[f"{prefix}.attn.k"]
            if k.ndim != 4:
                raise ValueError(f"cache tensor {prefix}.attn.k must be rank 4")
            bind_batch(f"{prefix}.attn.k", k.shape[0])
            expected_k = (
                k.shape[0],
                config.cache_capacity,
                config.num_heads,
                config.head_dim,
            )
            v = require(
                f"{prefix}.attn.v",
                (
                    k.shape[0],
                    config.cache_capacity,
                    config.num_heads,
                    config.value_head_dim,
                ),
                config.compute_dtype,
            )
            k = require(f"{prefix}.attn.k", expected_k, config.compute_dtype)
            count = require(f"{prefix}.attn.count", (), jnp.int32)
            if not np.array_equal(np.asarray(position), np.asarray(count)):
                raise ValueError(f"cache {prefix}.position does not equal attention count")
            attn = AttentionCache(
                k=k,
                v=v,
                count=count,
            )
        norm = None
        if f"{prefix}.norm" in present:
            count = tensors[f"{prefix}.norm.count"]
            if count.ndim != 1:
                raise ValueError(f"cache tensor {prefix}.norm.count must be rank 1")
            bind_batch(f"{prefix}.norm.count", count.shape[0])
            norm = NormState(
                count=require(f"{prefix}.norm.count", (count.shape[0],), jnp.int32),
                mean=require(
                    f"{prefix}.norm.mean",
                    (count.shape[0], config.norm_num_groups),
                    jnp.float32,
                ),
                var=require(
                    f"{prefix}.norm.var",
                    (count.shape[0], config.norm_num_groups),
                    jnp.float32,
                ),
            )
        ema = None
        if f"{prefix}.ema" in present:
            real = tensors[f"{prefix}.ema.real"]
            if real.ndim != 3:
                raise ValueError(f"cache tensor {prefix}.ema.real must be rank 3")
            bind_batch(f"{prefix}.ema.real", real.shape[0])
            ema_shape = (real.shape[0], config.model_dim, config.cema_ndim)
            real = require(f"{prefix}.ema.real", ema_shape, jnp.float32)
            imag = require(f"{prefix}.ema.imag", ema_shape, jnp.float32)
            ema = EMAState(h=real + 1j * imag)
        layers.append(LayerCache(attn=attn, norm=norm, ema=ema, position=position))

    final_norm = None
    if "final_norm" in present:
        count = tensors["final_norm.count"]
        if count.ndim != 1:
            raise ValueError("cache tensor final_norm.count must be rank 1")
        bind_batch("final_norm.count", count.shape[0])
        final_norm = NormState(
            count=require("final_norm.count", (count.shape[0],), jnp.int32),
            mean=require(
                "final_norm.mean",
                (count.shape[0], config.norm_num_groups),
                jnp.float32,
            ),
            var=require(
                "final_norm.var",
                (count.shape[0], config.norm_num_groups),
                jnp.float32,
            ),
        )
    cache = ModelCache(layer_caches=tuple(layers), final_norm=final_norm)
    validate_model_cache_host(cache, config)
    return cache
