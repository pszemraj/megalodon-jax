# Dev Notes - megalodon-jax

## Architecture Overview

Pure JAX/Equinox reimplementation of Megalodon. No CUDA kernels-all paths are JIT-compiled XLA.

### Key Differences from PyTorch Reference

| Component    | PyTorch Reference   | This Implementation                                   |
| ------------ | ------------------- | ----------------------------------------------------- |
| EMA          | Fused CUDA kernel   | JAX FFT path (training) / sequential scan (inference) |
| TimestepNorm | Fused CUDA w/ Kahan | Vectorized Welford w/ fp32 accumulators               |
| Attention    | SDPA + DropKey      | jnp.einsum, manual chunked attention                  |
| Parallelism  | 4D chunk-parallel   | Single-device only                                    |

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

### Sequential CEMA is Slow

When `return_cache=True`, CEMA uses `jax.lax.scan` over timesteps. This is correct but ~10-100x slower than the FFT path. The reference uses fused CUDA kernels to return last-state cheaply. In JAX, `return_cache` now defaults to False and is automatically disabled when `deterministic=False`, matching the PyTorch training guard; opt in to caching explicitly for streaming inference.

## Optional Experiments

- **Scan-loop token fallback:** The streaming token fallback could be rewritten as a `lax.scan` with strict masking. This is optional and should be validated with streaming/parity tests because it can introduce small numerical drift.
- **Fused SDPA path (dropout=0):** JAX `dot_product_attention` can use the cuDNN backend for faster chunk attention when masks are simple. This matches the PyTorch SDPA fast path (scale=1.0) but needs careful gating for cache/prefix masks and for any dropout use; keep the manual path as a fallback and validate parity on streaming/batch tests.
- **Mixed-precision attention accumulation:** Replace explicit fp32 casts in attention matmuls with `precision` + `preferred_element_type` to keep bf16 inputs while accumulating in fp32. This is closer to SDPA behavior on accelerators and may improve throughput, but it can shift numerics slightly; validate against PyTorch parity tests.

## Numerical Stability

### TimestepNorm - Kahan Summation Analysis

**Status: NOT NEEDED.** Spike test (2025-01-02) showed:

| Seq Length | Standard cumsum (fp32) | Kahan cumsum  | Performance        |
| ---------- | ---------------------- | ------------- | ------------------ |
| 1k         | rel_err ~8e-5          | rel_err ~8e-5 | Kahan 200x slower  |
| 8k         | rel_err ~4e-5          | rel_err ~4e-5 | Kahan 1400x slower |
| 32k        | rel_err ~5e-5          | rel_err ~5e-5 | Kahan 4700x slower |
| 64k        | rel_err ~2e-5          | rel_err ~2e-5 | Kahan 4200x slower |

**Finding**: TimestepNorm already uses fp32 accumulators internally (line 146: `stats_dtype = jnp.float32`). This provides sufficient precision-variance floor (1e-6) was never triggered at 64k tokens. Kahan compensation via `jax.lax.scan` is catastrophically slow (~4000x overhead) because scan is sequential while `jnp.cumsum` uses optimized XLA primitives.

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

`convert.py` provides `load_weights_from_torch()` for PyTorch checkpoint loading with:

- Shape validation on critical weights (embed, cema.alpha, wz, fc1, lm_head)
- Layer count validation before loading
- Clear error messages for config/checkpoint mismatch

## Cache Design

All cache objects are JAX pytrees with position counters as JAX scalar arrays (not Python ints) to prevent JIT recompilation on value changes.

## Testing

166 tests covering:

- Parity with PyTorch reference (rtol=1e-4 for fp32, rtol=1e-2 for bf16)
- Streaming equivalence: chunk-wise streaming matches batch processing (token fallback for partial chunks)
- JIT stability: no retracing on repeated calls with same shapes
- GPU/CPU coverage via pytest fixtures

## Profiling

See `docs/profiling.md` for detailed timing analysis.
