"""Pytest configuration and fixtures for megalodon-jax tests."""

import os

# Configure JAX memory BEFORE importing JAX
# Disable preallocation to avoid conflicts with PyTorch GPU memory
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Allocate memory on-demand rather than reserving a fraction
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax.numpy as jnp
import numpy as np
import pytest
import torch


def to_jax(t: torch.Tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(t.detach().cpu().numpy())


def to_torch(a: jnp.ndarray) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor."""
    return torch.from_numpy(np.array(a))


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture(autouse=True)
def clear_gpu_caches():
    """Clear GPU caches before and after each test to prevent OOM."""
    import gc

    # Clear before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    yield

    # Clear after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture
def force_fp32_matmul():
    """Force float32 matmuls for precision-sensitive parity tests.

    GPU matmuls use TensorFloat-32 by default on Ampere+ GPUs, which truncates
    mantissa from 23 to 10 bits. This causes ~1e-3 differences vs fp32.
    Use this fixture when exact parity with PyTorch CPU is required.
    """
    import jax

    original = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "float32")
    yield
    jax.config.update("jax_default_matmul_precision", original)
