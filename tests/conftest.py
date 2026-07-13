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

# Positive selection keeps routine CI bounded: newly added tests enter the
# complete suite by default and join this gate only after an explicit choice.
_FAST_TEST_PREFIXES = (
    "tests/test_foundation.py::TestMegalodonConfig::test_exact_named_presets_and_counts",
    "tests/test_foundation.py::TestMegalodonConfig::test_parameter_count_matches_allocated_trainable_leaves",
    "tests/test_foundation.py::TestMegalodonConfig::test_structural_and_finite_validation",
    "tests/test_foundation.py::TestMegalodonConfig::test_structural_fields_require_integer_types",
    "tests/test_foundation.py::TestMegalodonConfig::test_dtype_validation",
    "tests/test_foundation.py::TestMegalodonConfig::test_accum_dtype_precision_validation",
    "tests/test_foundation.py::TestMegalodonConfig::test_dropout_and_fresh_init_validation",
    "tests/test_foundation.py::TestLayerNormStorage::test_zero_stored_weight_has_identity_effective_scale",
    "tests/test_foundation.py::TestRotaryEmbeddingParity",
    "tests/test_foundation.py::TestCacheTypes::test_jax_scalar_counters_jit_compatible",
    "tests/test_core_layers.py::TestTimestepNorm::test_streaming_state_continuity",
    "tests/test_core_layers.py::TestTimestepNorm::test_exact_scalar_population_moments",
    "tests/test_core_layers.py::TestTimestepNorm::test_fully_masked_row_preserves_incoming_state",
    "tests/test_core_layers.py::TestTimestepNorm::test_state_dtype_fp32",
    "tests/test_core_layers.py::TestTimestepNorm::test_matches_float64_paper_oracle",
    "tests/test_core_layers.py::TestTimestepNorm::test_large_offset_continuation_matches_oracle_and_gradients",
    "tests/test_core_layers.py::TestTimestepNorm::test_forward_and_backward_have_no_sequence_while",
    "tests/test_core_layers.py::TestTimestepNorm::test_masked_nonfinite_tokens_have_zero_finite_gradients",
    "tests/test_core_layers.py::TestTimestepNormSegmentReset::test_segment_reset_matches_per_doc_alone",
    "tests/test_core_layers.py::TestTimestepNormSegmentReset::test_contiguous_runs_cover_reused_ids_singletons_and_padding",
    "tests/test_core_layers.py::TestTimestepNormSegmentReset::test_fully_padded_packed_row_returns_prior",
    "tests/test_core_layers.py::TestComplexEMA::test_pristine_prefill_matches_sequential_recurrence",
    "tests/test_core_layers.py::TestComplexEMA::test_state_continuity",
    "tests/test_core_layers.py::TestComplexEMASegmentReset::test_associative_matches_sequential_reset",
    "tests/test_core_layers.py::TestPrecisionPolicy::test_timestep_norm_bf16_input",
    "tests/test_core_layers.py::TestPrecisionPolicy::test_complex_ema_fft_vs_sequential_bf16",
    "tests/test_attention.py::TestChunkedAttention::test_pristine_prefill_partition_smoke",
    "tests/test_attention.py::TestChunkedAttention::test_pristine_prefill_has_no_sequence_loop",
    "tests/test_inference.py::TestCacheUtilities::test_explicit_pristine_and_none_cache_match",
    "tests/test_inference.py::TestSamplingAndGeneration::test_sample_controls_validate_before_greedy",
    "tests/test_inference.py::TestSamplingAndGeneration::test_generate_canonicalizes_all_true_mask",
    "tests/test_inference.py::TestSamplingAndGeneration::test_generate_rejects_left_padding",
    "tests/test_inference.py::TestSamplingAndGeneration::test_generate_empty_prompt_matches_explicit_bos",
    "tests/test_inference.py::TestSamplingAndGeneration::test_generate_padded_cache_modes_raise",
    "tests/test_inference.py::TestConversion::test_original_upstream_manifest_is_source_transcribed",
    "tests/test_inference.py::TestConversion::test_native_v2_roundtrip_is_exact",
    "tests/test_inference.py::TestConversion::test_cache_roundtrip_and_config_binding",
    "tests/test_inference.py::TestConversion::test_cache_schema_and_position_invariants_fail_closed",
    "tests/test_model.py::TestModelCache::test_model_cache_rejects_coherent_count_overflow",
    "tests/test_model.py::TestMegalodonForCausalLM::test_bf16_cached_logits_stay_within_compute_envelope",
    "tests/test_upstream_parity.py::test_tiny_forward_matches_released_source_equations",
    "tests/test_upstream_parity.py::test_all_parameter_gradients_match_released_source_equations",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the explicitly curated routine CPU correctness gate."""
    for item in items:
        if any(item.nodeid.startswith(prefix) for prefix in _FAST_TEST_PREFIXES):
            item.add_marker(pytest.mark.fast)


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
    """Clear GPU caches before and after each test to prevent OOM.

    :return Iterator[None]: Context that clears GPU caches around each test.
    """
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
