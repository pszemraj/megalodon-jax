"""Shared test utilities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch


def to_jax(t: torch.Tensor) -> jnp.ndarray:
    """Convert a PyTorch tensor to a JAX array.

    :param torch.Tensor t: Input PyTorch tensor.
    :return jnp.ndarray: JAX array on the default device.
    """
    return jnp.array(t.detach().cpu().numpy())


def to_torch(a: jnp.ndarray) -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor.

    :param jnp.ndarray a: Input JAX array.
    :return torch.Tensor: Torch tensor on CPU.
    """
    return torch.from_numpy(np.array(a))
