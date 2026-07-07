# Dev Notes - megalodon-jax

---

- [Dev Notes - megalodon-jax](#dev-notes---megalodon-jax)
  - [Release Notes](#release-notes)
  - [Architecture Overview](#architecture-overview)
    - [Key Differences from Upstream Reference](#key-differences-from-upstream-reference)
    - [Performance Characteristics](#performance-characteristics)
  - [Known Gaps](#known-gaps)
    - [No Fused Kernels](#no-fused-kernels)
    - [No 4D Chunk Parallelism](#no-4d-chunk-parallelism)
    - [Sequential CEMA vs FFT](#sequential-cema-vs-fft)
    - [Packed-Sequence State Isolation](#packed-sequence-state-isolation)
  - [Optional Experiments](#optional-experiments)
  - [Numerical Stability](#numerical-stability)
    - [TimestepNorm - Kahan Summation Analysis](#timestepnorm---kahan-summation-analysis)
    - [EMA Eigenvalue Stability](#ema-eigenvalue-stability)
    - [Q/K Normalization](#qk-normalization)
    - [Attention Masking](#attention-masking)
  - [Weight Conversion](#weight-conversion)
  - [Cache Design](#cache-design)
  - [Testing](#testing)
  - [Profiling](#profiling)

---

## Release Notes

- Unreleased: packed-sequence training with `segment_ids`/`position_ids` now fully isolates documents: strict attention masking plus ComplexEMA and TimestepNorm state resets at segment boundaries (see [Packed-Sequence State Isolation](#packed-sequence-state-isolation)). Models expose `supports_segment_reset = True` for harness capability detection.
- Unreleased: conversion utilities now live in `megalodon_jax.convert` and require torch; install `megalodon-jax[convert]` and import explicitly.
- Unreleased: `generate()` no longer accepts a `seed` argument; padded `attention_mask` is rejected for cached generation (`max_new_tokens > 1`, `return_cache=True`, or cache provided).
- Unreleased: added accum/softmax dtypes, GEMM backend selection, and centralized GEMM ops for future FP8 backends.
- Unreleased: added `docs/dtypes-and-stability.md` with downstream dtype guidance.

## Architecture Overview

Pure JAX/Equinox reimplementation of Megalodon. No CUDA kernels-all paths are JIT-compiled XLA.

### Key Differences from Upstream Reference

| Component    | Upstream Reference (custom kernels) | This Implementation (JAX)                         |
| ------------ | ----------------------------------- | ------------------------------------------------- |
| EMA          | Fused CUDA kernels                  | FFT path (training) / sequential scan (inference) |
| TimestepNorm | Fused CUDA w/ Kahan                 | Vectorized Welford w/ fp32 accumulators           |
| Attention    | Fused SDPA + DropKey                | jnp.einsum, manual chunked attention              |
| Parallelism  | 4D chunk-parallel                   | Single-device only                                |

### Performance Characteristics

- **FFT EMA path**: O(L log L), used during training (no cache)
- **Sequential EMA path**: O(L), maintains complex hidden state for streaming
- **Chunked attention**: Block-diagonal within chunks, no cross-chunk attention edges
- **Streaming prefill**: Chunk-wise attention in faithful chunk-local mode; sub-chunk calls use token updates without padding

## Known Gaps

### No Fused Kernels

Pure JAX without custom XLA ops. Performance is adequate for research/testing but not production-scale training.

### No 4D Chunk Parallelism

The paper's chunk-parallel axis requires multi-device coordination and sharded KV/state exchange. Out of scope for single-device.

### Sequential CEMA vs FFT

When `return_state=True`, CEMA uses `jax.lax.scan` over timesteps. This is ~6x slower than FFT at 4k tokens (26ms vs 0.25ms). However, **JAX is ~5x faster than PyTorch** for both paths:

| Seq Len | PyTorch FFT | JAX FFT | PyTorch Seq | JAX Seq |
| ------- | ----------- | ------- | ----------- | ------- |
| 1024    | 0.31ms      | 0.11ms  | 38.5ms      | 6.8ms   |
| 4096    | 1.27ms      | 0.25ms  | 153ms       | 26ms    |

Training uses FFT automatically (`return_state=False`). Sequential path is only for streaming inference where state must be preserved.

### Packed-Sequence State Isolation

Passing `segment_ids` (0 = padding) isolates packed documents end to end: attention is segment-masked, RoPE positions restart per document, and both ComplexEMA state and TimestepNorm running statistics reset at segment boundaries. Each packed document produces the same outputs (and gradients) as running it alone; verified by exact-zero gradient-isolation tests. `position_ids` is optional: when omitted, per-document positions are derived from `segment_ids`; pass it explicitly only for non-default position schemes.

Design notes:

- **Training-only.** Strict metadata is rejected on any streaming path (`cache` or `return_cache`); the model raises before any compute.
- **CEMA segmented path** defaults to a parallel `jax.lax.associative_scan` over `(A, b)` affine pairs with `A = 0` at boundaries. A sequential `lax.scan` fallback trades speed for minimal memory: the associative path materializes two `(L, B, D, N)` complex64 tensors (see table). Select it model-wide with `MegalodonConfig(use_associative_segment_scan=False)` (or per layer via `ComplexEMA(..., use_associative_segment_scan=False)`). The FFT path cannot express resets and is bypassed whenever `segment_ids` is given.
- **TimestepNorm segmented path** uses a reset-carrying associative scan, not "global cumsum minus value at boundary": the subtraction is exact algebraically but catastrophically cancels in fp32 once earlier segments' magnitudes dwarf the local sums (observed 2e-2 cross-doc contamination on layer-2 activations), and it leaves a gradient path across the boundary.
- **Returned state anchors at the last real token.** With `segment_ids`, the final CEMA state and `NormState` a layer returns reflect each row's last non-padding (and unmasked, for the norm) token: trailing padding starts its own run and resets the scan, so reading position `L-1` would report the fresh-reset baseline instead of the last document's state. All-padding rows return the fresh baseline.
- **Repeated ids are safe.** Isolation is boundary-based everywhere: attention compares contiguous-run indices derived from `segment_ids`, not raw id equality, so a packer that reuses a positive id for non-adjacent documents (e.g. `[1, 1, 2, 2, 1, 1]`) still gets full isolation, matching the CEMA/TimestepNorm reset semantics.
- **Chunk boundaries re-anchor per document.** Documents may start at any offset: the segmented attention path attends over the current + previous global chunk with an exact same-run/same-local-chunk mask, so a document that begins mid-chunk keeps the identical block-diagonal pattern it would have running alone (a re-anchored chunk reaches back at most `chunk_size` positions, so one predecessor chunk covers every allowed edge). Costs ~2x attention FLOPs/KV memory on packed rows versus unsegmented training.
- **Loss boundary masking is automatic.** `compute_loss(..., segment_ids=...)` excludes shifted label pairs that cross a segment boundary and pairs targeting padding (segment id 0); callers do not need to pre-set `ignore_index` at document joins.
- **Capability detection:** harnesses should check `getattr(model, "supports_segment_reset", False)` instead of introspecting `compute_loss`'s signature, which cannot distinguish attention-only isolation from full state isolation.

CEMA path timings (`scratch/benchmark_cema_reset.py`, D=256, N=16, RTX 5090, single trivial segment):

| L    | B   | FFT fwd | assoc fwd | seq fwd | FFT f+b | assoc f+b | seq f+b | A+b mem |
| ---- | --- | ------- | --------- | ------- | ------- | --------- | ------- | ------- |
| 1024 | 8   | 0.10ms  | 0.50ms    | 6.3ms   | 0.20ms  | 1.1ms     | 14.8ms  | 537MB   |
| 4096 | 8   | 0.43ms  | 2.3ms     | 25.3ms  | 0.95ms  | 5.9ms     | 59.3ms  | 2.1GB   |

The associative path is ~5x slower than FFT (training-viable); the sequential fallback is ~10-60x slower than associative on GPU but faster on CPU. Memory scales linearly in `L*B*D*N`; at production dims (D=1024) expect ~4x the table's footprint.

## Optional Experiments

- **Scan-loop token fallback:** The streaming token fallback could be rewritten as a `lax.scan` with strict masking. This is optional and should be validated with streaming/parity tests because it can introduce small numerical drift.
- **Fused SDPA path (dropout=0):** JAX `dot_product_attention` can use the cuDNN backend for faster chunk attention when masks are simple. This matches the PyTorch SDPA fast path (scale=1.0) but needs careful gating for cache/prefix masks and for any dropout use; keep the manual path as a fallback and validate parity on streaming/batch tests.
- **Mixed-precision attention accumulation:** Replace explicit fp32 casts in attention matmuls with `precision` + `preferred_element_type` to keep bf16 inputs while accumulating in fp32. This is closer to SDPA behavior on accelerators and may improve throughput, but it can shift numerics slightly; validate against PyTorch parity tests.
- **Vectorized prefill with cache:** Add a fast prefill path for `return_cache=True` when `L <= chunk_size` to avoid token-by-token cache construction; keep the current path as fallback and validate boundary parity.
- **Layer stack scan:** Consider a `lax.scan` over layers to reduce HLO size and compile time for deep models; requires careful static/dynamic argument handling in Equinox.
- **Export metadata sidecar:** Save config + git SHA + dtype policy alongside weights to prevent accidental mismatched loads.
- **FP8 GEMM backends:** Implement `mxfp8` and `nvfp4` GEMM backends and relax config validation once they exist.

## Numerical Stability

### TimestepNorm - Kahan Summation Analysis

**Status: NOT NEEDED.** Spike test (2025-01-02) showed:

| Seq Length | Standard cumsum (fp32) | Kahan cumsum  | Performance        |
| ---------- | ---------------------- | ------------- | ------------------ |
| 1k         | rel_err ~8e-5          | rel_err ~8e-5 | Kahan 200x slower  |
| 8k         | rel_err ~4e-5          | rel_err ~4e-5 | Kahan 1400x slower |
| 32k        | rel_err ~5e-5          | rel_err ~5e-5 | Kahan 4700x slower |
| 64k        | rel_err ~2e-5          | rel_err ~2e-5 | Kahan 4200x slower |

**Finding**: TimestepNorm already uses fp32 accumulators internally (search for `stats_dtype = jnp.float32` in `timestep_norm.py`). This provides sufficient precision-variance floor (1e-6) was never triggered at 64k tokens. Kahan compensation via `jax.lax.scan` is catastrophically slow (~4000x overhead) because scan is sequential while `jnp.cumsum` uses optimized XLA primitives.

**Recommendation**: Keep current implementation. If precision issues arise at 100k+ tokens, consider fp64 accumulators (simple, 2x memory) rather than Kahan.

### EMA Eigenvalue Stability

Stable by construction:

- `|q| = 1 - sigmoid(alpha) * sigmoid(delta)` with phase from `theta`
- `gamma` scaled by `sqrt(1/ndim)`
- Complex coefficients forced to complex64 regardless of parameter dtype

### Q/K Normalization

Per-head RMSNorm before affine transform. Attention uses `scale=1.0` (no `/sqrt(d_head)`). This matches the paper exactly.

### Attention Masking

Uses `-jnp.inf` for masked positions (not `finfo.min`). Ensures fully-masked queries produce NaN in softmax, triggering NaN guard to zero outputs.

## Weight Conversion

`convert.py` provides bidirectional PyTorch ↔ JAX conversion: it requires torch (install `megalodon-jax[convert]`) and is imported from `megalodon_jax.convert`.

**JAX → PyTorch** (`convert_jax_to_torch`):

- Exports all weights including `lm_head.weight` for tied models (PyTorch strict loading compatibility)
- Uses `.clone()` on tied weights to avoid SafeTensors shared memory errors
- Skips `inner.rope.inv_freq` by default because it is non-persistent in PyTorch; opt in with `include_rope_inv_freq=True`
- Supports `dtype=` for exported tensors; floats are normalized to fp32 before optional casting
- CEMA `gamma_{real,imag}` is exported in fp32 for stability
- Compatible with `safetensors.torch.save_file`

**PyTorch → JAX** (`load_weights_from_torch`):

- Shape validation on critical weights (embed, cema.alpha, wz, fc1, lm_head)
- Layer count validation before loading
- Clear error messages for config/checkpoint mismatch
- Handles both tied and untied LM heads

## Cache Design

All cache objects are JAX pytrees with position counters as JAX scalar arrays (not Python ints) to prevent JIT recompilation on value changes.

## Testing

150+ tests (200+ cases via parametrization) covering:

- Parity with the reference implementation (rtol=1e-4 for fp32, rtol=1e-2 for bf16)
- Streaming equivalence: chunk-wise streaming matches batch processing (token fallback for partial chunks)
- JIT stability: no retracing on repeated calls with same shapes
- GPU/CPU coverage via pytest fixtures
- PyTorch parity tests import `megalodon` from the external `megalodon-hf` package (dev dependency), not any in-repo Torch code.
- PyTorch parity dependency is pinned to commit [`f85a4784`](https://github.com/pszemraj/megalodon-hf/commit/f85a47849d07d52982a2e6e4cf0297c2621e9916) to avoid drift.
- Non-deterministic runs require a PRNG key when any dropout is enabled (enforced in `MegalodonForCausalLM.__call__` and `compute_loss`).

## Profiling

This repo no longer ships profiling scripts. For the PyTorch reference timing helpers, use [megalodon-hf/scripts](https://github.com/pszemraj/megalodon-hf/tree/f85a47849d07d52982a2e6e4cf0297c2621e9916/scripts).
