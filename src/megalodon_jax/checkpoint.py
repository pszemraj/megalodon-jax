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
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from safetensors import safe_open
from safetensors.flax import load_file, save_file

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.model import MegalodonForCausalLM
from megalodon_jax.types import AttentionCache, EMAState, LayerCache, ModelCache, NormState

MODEL_FORMAT = "megalodon-jax"
MODEL_FORMAT_VERSION = "2"
CACHE_FORMAT = "megalodon-jax-cache"
CACHE_FORMAT_VERSION = "1"
_DTYPE_FIELDS = (
    "param_dtype",
    "compute_dtype",
    "accum_dtype",
    "attention_softmax_dtype",
    "loss_softmax_dtype",
)


def _config_dict(config: MegalodonConfig) -> dict[str, Any]:
    data = dataclasses.asdict(config)
    for field in _DTYPE_FIELDS:
        data[field] = str(jnp.dtype(data[field]))
    return data


def _config_json(config: MegalodonConfig) -> str:
    return json.dumps(_config_dict(config), sort_keys=True, separators=(",", ":"))


def _config_from_json(payload: str) -> MegalodonConfig:
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
    """Return the stable fingerprint used to bind caches and checkpoints."""
    return hashlib.sha256(_config_json(config).encode("utf-8")).hexdigest()


def _manifest(tensors: dict[str, Array]) -> str:
    rows = [
        (name, tuple(int(size) for size in value.shape), str(value.dtype))
        for name, value in sorted(tensors.items())
    ]
    payload = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


Entry = tuple[
    str,
    Array,
    Callable[[MegalodonForCausalLM], Array],
]


def _model_entries(model: MegalodonForCausalLM) -> list[Entry]:
    entries: list[Entry] = []

    def add(name: str, value: Array, where: Callable[[MegalodonForCausalLM], Array]) -> None:
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
            add(
                f"{ap}.timenorm.prior_mean",
                attn.timenorm.prior_mean,
                lambda root, i=index: root.model.layers[i].attn.timenorm.prior_mean,
            )
            assert attn.timenorm.prior_logv is not None
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
            add(
                f"{fp}.norm.weight",
                ffn.norm.weight,
                lambda root, i=index: root.model.layers[i].ffn.norm.weight,
            )
            assert ffn.norm.bias is not None
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
    """Return the canonical v2 native parameter mapping."""
    return {name: value for name, value, _ in _model_entries(model)}


def _metadata(path: Path) -> dict[str, str]:
    try:
        with safe_open(str(path), framework="flax") as handle:
            metadata = handle.metadata()
    except Exception as exc:
        raise ValueError(f"{path} is not a readable SafeTensors file") from exc
    return {} if metadata is None else dict(metadata)


def _read_model_file(path: Path) -> tuple[dict[str, Array], dict[str, str]]:
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
    if metadata["rope_layout"] != "adjacent_pair":
        raise ValueError("checkpoint RoPE convention is incompatible")
    if metadata["normalization_storage"] != "plus_one":
        raise ValueError("checkpoint normalization storage is incompatible")
    if metadata["bias_schema"] != "upstream":
        raise ValueError("checkpoint projection-bias schema is incompatible")
    tensors = load_file(str(path))
    actual_manifest = _manifest(tensors)
    if actual_manifest != metadata["parameter_manifest_sha256"]:
        raise ValueError("checkpoint parameter manifest does not match its metadata")
    return tensors, metadata


def _apply_parameters(
    model: MegalodonForCausalLM,
    tensors: dict[str, Array],
    *,
    include: Collection[str] | None = None,
) -> tuple[MegalodonForCausalLM, dict[str, Any]]:
    entries = _model_entries(model)
    expected = {name for name, _, _ in entries}
    available = set(tensors)
    selected = expected if include is None else set(include)
    if include is None:
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        if missing or unexpected:
            raise ValueError(
                f"strict checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
            )
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
        value = tensors[name]
        if value.shape != template.shape:
            raise ValueError(
                f"shape mismatch for {name}: checkpoint {value.shape}, model {template.shape}"
            )
        if value.dtype != template.dtype:
            raise ValueError(
                f"dtype mismatch for {name}: checkpoint {value.dtype}, model {template.dtype}"
            )
        model = eqx.tree_at(where, model, value)
        restored.append(name)
    return model, {"restored": restored, "initialized": initialized}


def save_checkpoint(model: MegalodonForCausalLM, path: str | Path) -> None:
    """Save a strict native v2 model checkpoint."""
    destination = Path(path)
    if destination.suffix != ".safetensors":
        raise ValueError("native checkpoints must use the .safetensors suffix")
    tensors = model_state_dict(model)
    config_json = _config_json(model.config)
    metadata = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "config_json": config_json,
        "config_fingerprint": config_fingerprint(model.config),
        "parameter_manifest_sha256": _manifest(tensors),
        "rope_layout": "adjacent_pair",
        "normalization_storage": "plus_one",
        "bias_schema": "upstream",
        "initializer_schema": "split-boundary-internal-v1",
        "tying": "tied" if model.tied else "untied",
        "dtype_policy": "fp32-params-fp32-or-bf16-compute",
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    key: Array,
) -> MegalodonForCausalLM:
    """Load a strict native v2 model checkpoint."""
    tensors, metadata = _read_model_file(Path(path))
    config = _config_from_json(metadata["config_json"])
    if config_fingerprint(config) != metadata["config_fingerprint"]:
        raise ValueError("checkpoint config fingerprint is invalid")
    model = MegalodonForCausalLM(config, key=key)
    model, _ = _apply_parameters(model, tensors)
    expected_tying = "tied" if model.tied else "untied"
    if metadata["tying"] != expected_tying:
        raise ValueError("checkpoint tying metadata disagrees with its configuration")
    return model


def load_partial_checkpoint(
    path: str | Path,
    config: MegalodonConfig,
    include: Collection[str],
    *,
    key: Array,
) -> tuple[MegalodonForCausalLM, dict[str, Any]]:
    """Restore an explicit parameter allowlist into a freshly initialized model."""
    tensors, _ = _read_model_file(Path(path))
    model = MegalodonForCausalLM(config, key=key)
    return _apply_parameters(model, tensors, include=include)


def _cache_tensors(cache: ModelCache) -> tuple[dict[str, Array], list[str]]:
    tensors: dict[str, Array] = {}
    present: list[str] = []
    for index, layer in enumerate(cache.layer_caches):
        if layer is None:
            continue
        prefix = f"layers.{index}"
        tensors[f"{prefix}.position"] = layer.position
        present.append(f"{prefix}.position")
        if layer.attn is not None:
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
    """Save an inference cache bound to an exact model configuration."""
    destination = Path(path)
    if destination.suffix != ".safetensors":
        raise ValueError("cache files must use the .safetensors suffix")
    if len(cache.layer_caches) != config.num_layers:
        raise ValueError("cache layer count does not match configuration")
    tensors, present = _cache_tensors(cache)
    metadata = {
        "format": CACHE_FORMAT,
        "format_version": CACHE_FORMAT_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "present_json": json.dumps(present, separators=(",", ":")),
        "tensor_manifest_sha256": _manifest(tensors),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    temporary.replace(destination)


def load_inference_cache(
    path: str | Path,
    config: MegalodonConfig,
) -> ModelCache:
    """Load a same-schema inference cache for an exact configuration."""
    source = Path(path)
    metadata = _metadata(source)
    if (
        metadata.get("format") != CACHE_FORMAT
        or metadata.get("format_version") != CACHE_FORMAT_VERSION
    ):
        raise ValueError("incompatible or legacy inference cache")
    if metadata.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("cache configuration fingerprint mismatch")
    tensors = load_file(str(source))
    if _manifest(tensors) != metadata.get("tensor_manifest_sha256"):
        raise ValueError("cache tensor manifest mismatch")
    try:
        present = set(json.loads(metadata["present_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("cache presence metadata is invalid") from exc

    layers: list[LayerCache | None] = []
    for index in range(config.num_layers):
        prefix = f"layers.{index}"
        if not any(item.startswith(prefix) for item in present):
            layers.append(None)
            continue
        position = tensors[f"{prefix}.position"]
        attn = None
        if f"{prefix}.attn" in present:
            attn = AttentionCache(
                k=tensors[f"{prefix}.attn.k"],
                v=tensors[f"{prefix}.attn.v"],
                count=tensors[f"{prefix}.attn.count"],
            )
            expected_k = (
                attn.k.shape[0],
                config.cache_capacity,
                config.num_heads,
                config.head_dim,
            )
            expected_v = (
                attn.v.shape[0],
                config.cache_capacity,
                config.num_heads,
                config.value_head_dim,
            )
            if attn.k.shape != expected_k or attn.v.shape != expected_v:
                raise ValueError("cache KV shape is incompatible with configuration")
        norm = None
        if f"{prefix}.norm" in present:
            norm = NormState(
                count=tensors[f"{prefix}.norm.count"],
                mean=tensors[f"{prefix}.norm.mean"],
                var=tensors[f"{prefix}.norm.var"],
            )
        ema = None
        if f"{prefix}.ema" in present:
            ema = EMAState(
                h=tensors[f"{prefix}.ema.real"].astype(jnp.float32)
                + 1j * tensors[f"{prefix}.ema.imag"].astype(jnp.float32)
            )
        layers.append(LayerCache(attn=attn, norm=norm, ema=ema, position=position))

    final_norm = None
    if "final_norm" in present:
        final_norm = NormState(
            count=tensors["final_norm.count"],
            mean=tensors["final_norm.mean"],
            var=tensors["final_norm.var"],
        )
    return ModelCache(layer_caches=tuple(layers), final_norm=final_norm)
