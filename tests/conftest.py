"""Pytest configuration and fixtures for megalodon-jax tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

# Configure JAX memory BEFORE importing JAX
# Disable preallocation to avoid conflicts with PyTorch GPU memory
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Allocate memory on-demand rather than reserving a fraction
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None

# Cache GPU availability at module load
_JAX_GPU_AVAILABLE = jax.default_backend() == "gpu"
_TORCH_AVAILABLE = torch is not None
_TORCH_CUDA_AVAILABLE = bool(_TORCH_AVAILABLE and torch.cuda.is_available())
_AGGRESSIVE_CLEANUP = os.environ.get("MEGALODON_TEST_AGGRESSIVE_CLEANUP", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def pytest_sessionstart(session: pytest.Session) -> None:
    """Print device info at test session start.

    :param pytest.Session session: Pytest session object.
    :return None: None.
    """
    jax_device = jax.devices()[0]
    jax_backend = jax.default_backend()
    torch_device = "cuda" if _TORCH_CUDA_AVAILABLE else "cpu"

    print(f"\n{'=' * 60}")
    print("megalodon-jax test session")
    print(f"  JAX backend: {jax_backend} ({jax_device})")
    if _TORCH_AVAILABLE:
        print(f"  PyTorch device: {torch_device}")
    else:
        print("  PyTorch device: unavailable")
    if _JAX_GPU_AVAILABLE:
        print("  GPU tests: ENABLED")
    else:
        print("  GPU tests: DISABLED (running on CPU)")
    print(f"{'=' * 60}\n")


@pytest.fixture
def random_seed() -> int:
    """Seed numpy and torch RNGs for reproducibility.

    :return int: Seed value used for the session.
    """
    if _TORCH_AVAILABLE:
        torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture(autouse=True)
def clear_gpu_caches() -> Iterator[None]:
    """Release accelerator objects around tests that can otherwise retain memory.

    CPU tests skip collection by default because collecting every test adds
    substantial latency without releasing JAX compilation caches. Set
    ``MEGALODON_TEST_AGGRESSIVE_CLEANUP=1`` for constrained CPU environments.

    :return Iterator[None]: Context that clears GPU caches around each test.
    """
    if not (_AGGRESSIVE_CLEANUP or _JAX_GPU_AVAILABLE or _TORCH_CUDA_AVAILABLE):
        yield
        return

    import gc

    # Clear before test
    gc.collect()
    if _TORCH_CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    yield

    # Clear after test
    gc.collect()
    if _TORCH_CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
