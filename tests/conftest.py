"""Pytest configuration and fixtures for megalodon-jax tests."""

import os

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


def pytest_sessionstart(session):
    """Print device info at test session start."""
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
def has_gpu():
    """Session-scoped fixture indicating GPU availability."""
    return _JAX_GPU_AVAILABLE


@pytest.fixture(scope="session")
def jax_device():
    """Session-scoped fixture returning the primary JAX device."""
    return jax.devices()[0]


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


@pytest.fixture
def torch_device():
    """Get the appropriate torch device and ensure TF32 settings match JAX.

    Both PyTorch and JAX should use the same precision mode (TF32 on GPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable TF32 to match JAX's default behavior
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    return device


def sync_and_clear_torch():
    """Synchronize and clear PyTorch GPU memory before switching to JAX."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def sync_and_clear_jax():
    """Synchronize and clear JAX GPU memory before switching to PyTorch."""
    import gc

    import jax

    # Block until all JAX computations complete
    for device in jax.devices():
        if device.platform == "gpu":
            # Force sync by blocking on a trivial computation
            jax.device_get(jnp.array(0))
            break
    gc.collect()
