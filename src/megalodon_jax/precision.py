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

from collections.abc import Iterable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from megalodon_jax.model import MegalodonForCausalLM, MegalodonModel


def _iter_sensitive_params(
    model: MegalodonForCausalLM | MegalodonModel,
) -> Iterable[tuple[str, Array]]:
    """Yield (name, array) pairs for precision-sensitive parameters.

    :param model: Model to inspect for sensitive parameters.
    :return Iterable[tuple[str, Array]]: Pairs of parameter name and array.
    """
    core = model.model if isinstance(model, MegalodonForCausalLM) else model

    for i, layer in enumerate(core.layers):
        attn = layer.attn
        prefix = f"layers.{i}.attn"

        yield f"{prefix}.cema.alpha", attn.cema.alpha
        yield f"{prefix}.cema.delta", attn.cema.delta
        yield f"{prefix}.cema.theta", attn.cema.theta
        yield f"{prefix}.cema.gamma_real", attn.cema.gamma_real
        yield f"{prefix}.cema.gamma_imag", attn.cema.gamma_imag
        yield f"{prefix}.cema.omega", attn.cema.omega
        yield f"{prefix}.gamma", attn.gamma
        yield f"{prefix}.beta", attn.beta

        if attn.timenorm.weight is not None:
            yield f"{prefix}.timenorm.weight", attn.timenorm.weight
        if attn.timenorm.bias is not None:
            yield f"{prefix}.timenorm.bias", attn.timenorm.bias
        if attn.rmsnorm.gamma is not None:
            yield f"{prefix}.rmsnorm.gamma", attn.rmsnorm.gamma

        if layer.ffn.norm.weight is not None:
            yield f"layers.{i}.ffn.norm.weight", layer.ffn.norm.weight
        if layer.ffn.norm.bias is not None:
            yield f"layers.{i}.ffn.norm.bias", layer.ffn.norm.bias

    if core.norm.weight is not None:
        yield "norm.weight", core.norm.weight
    if core.norm.bias is not None:
        yield "norm.bias", core.norm.bias


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

    def cast_if_present(arr: Array | None) -> Array | None:
        """Cast array to dtype when present.

        :param Array | None arr: Array to cast (optional).
        :return Array | None: Casted array or None.
        """
        return arr.astype(dtype) if arr is not None else None

    def update_core(core: MegalodonModel) -> MegalodonModel:
        """Cast precision-sensitive parameters for a core MegalodonModel.

        :param MegalodonModel core: Model to update.
        :return MegalodonModel: Updated model.
        """
        for i, layer in enumerate(core.layers):
            attn = layer.attn
            core = eqx.tree_at(
                lambda m: m.layers[i].attn.cema.alpha, core, attn.cema.alpha.astype(dtype)
            )
            core = eqx.tree_at(
                lambda m: m.layers[i].attn.cema.delta, core, attn.cema.delta.astype(dtype)
            )
            core = eqx.tree_at(
                lambda m: m.layers[i].attn.cema.theta, core, attn.cema.theta.astype(dtype)
            )
            core = eqx.tree_at(
                lambda m: m.layers[i].attn.cema.gamma_real,
                core,
                attn.cema.gamma_real.astype(dtype),
            )
            core = eqx.tree_at(
                lambda m: m.layers[i].attn.cema.gamma_imag,
                core,
                attn.cema.gamma_imag.astype(dtype),
            )
            core = eqx.tree_at(
                lambda m: m.layers[i].attn.cema.omega, core, attn.cema.omega.astype(dtype)
            )
            core = eqx.tree_at(lambda m: m.layers[i].attn.gamma, core, attn.gamma.astype(dtype))
            core = eqx.tree_at(lambda m: m.layers[i].attn.beta, core, attn.beta.astype(dtype))

            if attn.timenorm.weight is not None:
                core = eqx.tree_at(
                    lambda m: m.layers[i].attn.timenorm.weight,
                    core,
                    cast_if_present(attn.timenorm.weight),
                )
            if attn.timenorm.bias is not None:
                core = eqx.tree_at(
                    lambda m: m.layers[i].attn.timenorm.bias,
                    core,
                    cast_if_present(attn.timenorm.bias),
                )
            if attn.rmsnorm.gamma is not None:
                core = eqx.tree_at(
                    lambda m: m.layers[i].attn.rmsnorm.gamma,
                    core,
                    cast_if_present(attn.rmsnorm.gamma),
                )

            if layer.ffn.norm.weight is not None:
                core = eqx.tree_at(
                    lambda m: m.layers[i].ffn.norm.weight,
                    core,
                    cast_if_present(layer.ffn.norm.weight),
                )
            if layer.ffn.norm.bias is not None:
                core = eqx.tree_at(
                    lambda m: m.layers[i].ffn.norm.bias,
                    core,
                    cast_if_present(layer.ffn.norm.bias),
                )

        if core.norm.weight is not None:
            core = eqx.tree_at(lambda m: m.norm.weight, core, cast_if_present(core.norm.weight))
        if core.norm.bias is not None:
            core = eqx.tree_at(lambda m: m.norm.bias, core, cast_if_present(core.norm.bias))

        return core

    if isinstance(model, MegalodonForCausalLM):
        new_core = update_core(model.model)
        return eqx.tree_at(lambda m: m.model, model, new_core)
    return update_core(model)
