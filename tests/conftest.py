"""Pytest configuration and fixtures for megalodon-jax tests."""

import os
from collections.abc import Iterator

# Configure JAX memory BEFORE importing JAX
# Disable preallocation to avoid conflicts with PyTorch GPU memory
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Allocate memory on-demand rather than reserving a fraction
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# Cache GPU availability at module load
_JAX_GPU_AVAILABLE = jax.default_backend() == "gpu"
_TORCH_CUDA_AVAILABLE = torch.cuda.is_available()


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
    print(f"  PyTorch device: {torch_device}")
    if _JAX_GPU_AVAILABLE:
        print("  GPU tests: ENABLED")
    else:
        print("  GPU tests: DISABLED (running on CPU)")
    print(f"{'=' * 60}\n")


# Skip decorator for tests requiring GPU
requires_gpu = pytest.mark.skipif(
    not _JAX_GPU_AVAILABLE,
    reason="Test requires GPU but none available",
)


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Report GPU availability for the session.

    :return bool: True if a JAX GPU backend is available.
    """
    return _JAX_GPU_AVAILABLE


@pytest.fixture(scope="session")
def jax_device() -> jax.Device:
    """Provide the primary JAX device.

    :return jax.Device: Primary JAX device for the session.
    """
    return jax.devices()[0]


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


@pytest.fixture
def random_seed() -> int:
    """Seed numpy and torch RNGs for reproducibility.

    :return int: Seed value used for the session.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture(autouse=True)
def clear_gpu_caches() -> Iterator[None]:
    """Clear GPU caches before and after each test to prevent OOM.

    :return Iterator[None]: Context that clears GPU caches around each test.
    """
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
def force_fp32_matmul() -> Iterator[None]:
    """Force float32 matmuls for precision-sensitive parity tests.

    GPU matmuls use TensorFloat-32 by default on Ampere+ GPUs, which truncates
    mantissa from 23 to 10 bits. This causes ~1e-3 differences vs fp32.
    Use this fixture when exact parity with PyTorch CPU is required.

    :return Iterator[None]: Context that forces float32 matmul precision.
    """
    import jax

    original = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "float32")
    yield
    jax.config.update("jax_default_matmul_precision", original)


@pytest.fixture
def torch_device() -> torch.device:
    """Get the appropriate torch device and ensure TF32 settings match JAX.

    Both PyTorch and JAX should use the same precision mode (TF32 on GPU).

    :return torch.device: Selected torch device for the session.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable TF32 to match JAX's default behavior
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    return device


def sync_and_clear_torch() -> None:
    """Synchronize and clear PyTorch GPU memory before switching to JAX.

    :return None: None.
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def sync_and_clear_jax() -> None:
    """Synchronize and clear JAX GPU memory before switching to PyTorch.

    :return None: None.
    """
    import gc

    import jax

    # Block until all JAX computations complete
    for device in jax.devices():
        if device.platform == "gpu":
            # Force sync by blocking on a trivial computation
            jax.device_get(jnp.array(0))
            break
    gc.collect()
