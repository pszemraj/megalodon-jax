"""Shared test utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import torch


def _require_torch() -> "torch":
    """Return torch or skip the caller if torch is unavailable.

    :return torch: The torch module.
    """
    import pytest

    return pytest.importorskip("torch")


def _require_megalodon() -> Any:
    """Return megalodon-hf module or skip the caller if unavailable.

    :return Any: The megalodon module.
    """
    import pytest

    return pytest.importorskip("megalodon")


def to_jax(t: "torch.Tensor") -> jnp.ndarray:
    """Convert a PyTorch tensor to a JAX array.

    :param torch.Tensor t: Input PyTorch tensor.
    :return jnp.ndarray: JAX array on the default device.
    """
    _require_torch()
    return jnp.array(t.detach().cpu().numpy())


def to_torch(a: jnp.ndarray) -> "torch.Tensor":
    """Convert a JAX array to a PyTorch tensor.

    :param jnp.ndarray a: Input JAX array.
    :return torch.Tensor: Torch tensor on CPU.
    """
    torch = _require_torch()
    return torch.from_numpy(np.array(a))


def require_torch_modeling() -> Any:
    """Return megalodon.modeling_megalodon or skip if missing.

    :return Any: The modeling_megalodon submodule.
    """
    megalodon = _require_megalodon()
    return megalodon.modeling_megalodon
