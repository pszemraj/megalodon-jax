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
"""Precision utilities for Megalodon JAX."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from megalodon_jax.model import MegalodonForCausalLM, MegalodonModel

SensitivePath = tuple[str, Callable[[MegalodonModel], Array]]


def _iter_sensitive_paths(core: MegalodonModel) -> Iterable[SensitivePath]:
    """Yield the single authoritative set of fp32-sensitive model paths.

    :param MegalodonModel core: Core model whose parameter topology selects optional paths.

    Yields:
        SensitivePath: Parameter name and selector for each fp32-sensitive array.
    """
    for i, layer in enumerate(core.layers):
        attn = layer.attn
        prefix = f"layers.{i}.attn"

        yield f"{prefix}.cema.alpha", lambda m, i=i: m.layers[i].attn.cema.alpha
        yield f"{prefix}.cema.delta", lambda m, i=i: m.layers[i].attn.cema.delta
        yield f"{prefix}.cema.theta", lambda m, i=i: m.layers[i].attn.cema.theta
        yield f"{prefix}.cema.gamma_real", lambda m, i=i: m.layers[i].attn.cema.gamma_real
        yield f"{prefix}.cema.gamma_imag", lambda m, i=i: m.layers[i].attn.cema.gamma_imag
        yield f"{prefix}.cema.omega", lambda m, i=i: m.layers[i].attn.cema.omega
        yield f"{prefix}.gamma", lambda m, i=i: m.layers[i].attn.gamma
        yield f"{prefix}.beta", lambda m, i=i: m.layers[i].attn.beta

        if attn.timenorm.weight is not None:
            yield f"{prefix}.timenorm.weight", lambda m, i=i: m.layers[i].attn.timenorm.weight
        if attn.timenorm.bias is not None:
            yield f"{prefix}.timenorm.bias", lambda m, i=i: m.layers[i].attn.timenorm.bias
        if attn.rmsnorm.gamma is not None:
            yield f"{prefix}.rmsnorm.gamma", lambda m, i=i: m.layers[i].attn.rmsnorm.gamma

        if layer.ffn.norm.weight is not None:
            yield f"layers.{i}.ffn.norm.weight", lambda m, i=i: m.layers[i].ffn.norm.weight
        if layer.ffn.norm.bias is not None:
            yield f"layers.{i}.ffn.norm.bias", lambda m, i=i: m.layers[i].ffn.norm.bias
        if layer.ffn.alpha is not None:
            yield f"layers.{i}.ffn.alpha", lambda m, i=i: m.layers[i].ffn.alpha

    if core.norm.weight is not None:
        yield "norm.weight", lambda m: m.norm.weight
    if core.norm.bias is not None:
        yield "norm.bias", lambda m: m.norm.bias


def _iter_sensitive_params(
    model: MegalodonForCausalLM | MegalodonModel,
) -> Iterable[tuple[str, Array]]:
    """Yield (name, array) pairs for precision-sensitive parameters.

    :param model: Model to inspect for sensitive parameters.
    :return Iterable[tuple[str, Array]]: Pairs of parameter name and array.
    """
    core = model.model if isinstance(model, MegalodonForCausalLM) else model

    for name, select in _iter_sensitive_paths(core):
        yield name, select(core)


def audit_sensitive_param_dtypes(
    model: MegalodonForCausalLM | MegalodonModel,
    *,
    expected_dtype: jnp.dtype = jnp.float32,
) -> dict[str, jnp.dtype]:
    """Return precision-sensitive parameters whose dtype differs from expected_dtype.

    :param MegalodonForCausalLM | MegalodonModel model: Model to audit.
    :param jnp.dtype expected_dtype: Expected dtype for sensitive params.
    :return dict[str, jnp.dtype]: Mapping from parameter name to actual dtype.
    """
    mismatches: dict[str, jnp.dtype] = {}
    for name, arr in _iter_sensitive_params(model):
        if not eqx.is_array(arr):
            continue
        if arr.dtype != expected_dtype:
            mismatches[name] = arr.dtype
    return mismatches


def ensure_sensitive_param_dtype(
    model: MegalodonForCausalLM | MegalodonModel,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> MegalodonForCausalLM | MegalodonModel:
    """Cast precision-sensitive parameters to the requested dtype.

    :param MegalodonForCausalLM | MegalodonModel model: Model to update.
    :param jnp.dtype dtype: Target dtype for sensitive params.
    :return MegalodonForCausalLM | MegalodonModel: Updated model.
    """

    def update_core(core: MegalodonModel) -> MegalodonModel:
        """Cast precision-sensitive parameters for a core MegalodonModel.

        :param MegalodonModel core: Model to update.
        :return MegalodonModel: Updated model.
        """
        for _, select in _iter_sensitive_paths(core):
            core = eqx.tree_at(select, core, select(core).astype(dtype))
        return core

    if isinstance(model, MegalodonForCausalLM):
        new_core = update_core(model.model)
        return eqx.tree_at(lambda m: m.model, model, new_core)
    return update_core(model)
