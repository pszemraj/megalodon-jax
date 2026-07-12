# Dev Notes - megalodon-jax

## Release Notes

- Unreleased: TimestepNorm now implements exact scalar-group population moments, block-Welford continuation, packed resets, and upstream masked-output semantics.
- Unreleased: RoPE uses released adjacent-pair coordinates and derived non-trainable frequencies.
- Unreleased: initialization, projection biases, plus-one normalization storage, explicit output tying, and exact named presets now match their documented contracts.
- Unreleased: faithful and optional sliding caches are invariant to arbitrary call partitioning and use fixed-capacity state.
- Unreleased: native model/cache format v2 fails closed; original-upstream conversion targets only the exact released schema.
- Unreleased: FP32 and BF16 are the complete supported numerical surface. Float16 is rejected.
- Unreleased: FP32 dot products request per-operation highest precision on GPU; TensorFloat-32 is not treated as FP32 correctness mode.
- Unreleased: downstream Megalodon packages were removed from development and tests. Parity uses the exact local source plus an independent differentiable Torch oracle.
- Unreleased: the verified dependency window is JAX 0.8.2 through 0.10.x with Equinox 0.13.x; the baseline and latest endpoint pairs pass the complete verifier.

## Architecture Overview

Pure JAX/Equinox reimplementation of Megalodon. No custom CUDA extension is required; executable paths are JIT-compiled by XLA.

### Key Differences from Upstream Reference

| Component    | Upstream Reference (custom kernels) | This Implementation (JAX)                         |
| ------------ | ----------------------------------- | ------------------------------------------------- |
| EMA          | Fused CUDA kernels                  | FFT path (training) / sequential scan (inference) |
| TimestepNorm | Fused CUDA block-Welford            | FP32 block-Welford scan                           |
| Attention    | Fused/manual source paths           | Manual chunked/sliding attention                  |
| Parallelism  | 4D chunk-parallel                   | Single-device only                                |

### Performance Characteristics

- **FFT EMA path**: O(L log L), used during training (no cache)
- **Sequential EMA path**: O(L), maintains complex hidden state for streaming
- **Chunked attention**: Block-diagonal within chunks, no cross-chunk attention edges
- **Streaming prefill**: Chunk-wise attention in faithful chunk-local mode; sub-chunk calls use token updates without padding

## Known Gaps

### No Fused Kernels

Pure JAX without custom extension code. Correctness gates do not build the original extension; CUDA source remains an authoritative reference for fused semantics.

### No 4D Chunk Parallelism

The paper's chunk-parallel axis requires multi-device coordination and sharded KV/state exchange. Out of scope for single-device.

### Sequential CEMA vs FFT

When state is required, CEMA uses `jax.lax.scan` over timesteps and is slower than the FFT path. Training uses FFT automatically when no cache or packed reset is required. Treat historical timing numbers as hardware/version-specific and rerun benchmarks after any correctness or dependency change.

### Packed-Sequence State Isolation

Passing `segment_ids` (0 = padding) isolates packed documents end to end: attention is segment-masked, RoPE positions restart per document, and both ComplexEMA state and TimestepNorm running statistics reset at segment boundaries. Each packed document produces the same outputs (and gradients) as running it alone; verified by exact-zero gradient-isolation tests. `position_ids` is optional: when omitted, per-document positions are derived from `segment_ids`; pass it explicitly only for non-default position schemes.

Design notes:

- **Training-only.** Strict metadata is rejected on any streaming path (`cache` or `return_cache`); the model raises before any compute.
- **CEMA segmented path** defaults to a parallel `jax.lax.associative_scan` over `(A, b)` affine pairs with `A = 0` at boundaries. A sequential `lax.scan` fallback trades speed for minimal memory: the associative path materializes two `(L, B, D, N)` complex64 tensors (see table). Select it model-wide with `MegalodonConfig(use_associative_segment_scan=False)` (or per layer via `ComplexEMA(..., use_associative_segment_scan=False)`). The FFT path cannot express resets and is bypassed whenever `segment_ids` is given.
- **TimestepNorm segmented path** uses reset-aware population block-Welford updates. Each token block contributes its within-group feature mean and variance; masked blocks do not update state and emit exact zeros.
- **Returned state anchors at the last real token.** With `segment_ids`, the final CEMA state and `NormState` a layer returns reflect each row's last non-padding (and unmasked, for the norm) token: trailing padding starts its own run and resets the scan, so reading position `L-1` would report the fresh-reset baseline instead of the last document's state. All-padding rows return the fresh baseline.
- **Repeated ids are safe.** Isolation is boundary-based everywhere: attention compares contiguous-run indices derived from `segment_ids`, not raw id equality, so a packer that reuses a positive id for non-adjacent documents (e.g. `[1, 1, 2, 2, 1, 1]`) still gets full isolation, matching the CEMA/TimestepNorm reset semantics.
- **Chunk boundaries re-anchor per document.** Documents may start at any offset: the segmented attention path attends over the current + previous global chunk with an exact same-run/same-local-chunk mask, so a document that begins mid-chunk keeps the identical block-diagonal pattern it would have running alone (a re-anchored chunk reaches back at most `chunk_size` positions, so one predecessor chunk covers every allowed edge). Costs ~2x attention FLOPs/KV memory on packed rows versus unsegmented training.
- **Loss boundary masking is automatic.** `compute_loss(..., segment_ids=...)` excludes shifted label pairs that cross a segment boundary and pairs targeting padding (segment id 0); callers do not need to pre-set `ignore_index` at document joins.
- **Capability detection:** harnesses should check `getattr(model, "supports_segment_reset", False)` instead of introspecting `compute_loss`'s signature, which cannot distinguish attention-only isolation from full state isolation.

CEMA path timings ([benchmarks/benchmark_cema_reset.py](../benchmarks/benchmark_cema_reset.py), D=256, N=16, RTX 5090, single trivial segment):

| L    | B   | FFT fwd | assoc fwd | seq fwd | FFT f+b | assoc f+b | seq f+b | A+b mem |
| ---- | --- | ------- | --------- | ------- | ------- | --------- | ------- | ------- |
| 1024 | 8   | 0.10ms  | 0.50ms    | 6.3ms   | 0.20ms  | 1.1ms     | 14.8ms  | 537MB   |
| 4096 | 8   | 0.43ms  | 2.3ms     | 25.3ms  | 0.95ms  | 5.9ms     | 59.3ms  | 2.1GB   |

The associative path is ~5x slower than FFT (training-viable); the sequential fallback is ~10-60x slower than associative on GPU but faster on CPU. Memory scales linearly in `L*B*D*N`; at production dims (D=1024) expect ~4x the table's footprint.

## Numerical Stability

### TimestepNorm moments

State uses FP32 population mean/variance and a token-block count. A valid token contributes all features in each group, including within-token variance, using the same block-Welford algebra as the released CUDA implementation. The configured `norm_eps` is added only when normalizing; no artificial state variance floor or nonzero empty-state M2 is injected.

The production training and continuation paths evaluate this causal recurrence with a sequential `jax.lax.scan`. This replaces the former vectorized cumulative-sum path to preserve exact block-Welford, masking, reset, and chunk-continuation semantics, with a throughput tradeoff that should be measured separately from correctness.

### EMA Eigenvalue Stability

Stable by construction:

- `|q| = 1 - sigmoid(alpha) * sigmoid(delta)` with phase from `theta`
- `gamma` scaled by `sqrt(1/ndim)`
- Complex coefficients forced to complex64 regardless of parameter dtype

### Q/K Normalization

Per-head RMSNorm before affine transform. Attention uses `scale=1.0` (no `/sqrt(d_head)`). This matches the paper exactly.

### Attention Masking

Uses `-jnp.inf` for masked positions and explicitly zeroes fully masked rows after softmax. Attention masks and segment masks cannot contribute values or gradients across invalid edges.

## Weight Conversion

`checkpoint.py` owns strict native SafeTensors v2 persistence. `convert.py` owns only the exact original released PyTorch keyspace and requires Torch. `export_upstream_state_dict`, `load_upstream_state_dict`, and `load_upstream_checkpoint` validate the complete key set, shapes, CEMA representation, normalization storage, bias topology, RoPE frequencies, and tying. See [jax-torch.md](jax-torch.md).

## Cache Design

All cache objects are JAX pytrees with fixed-capacity KV rings and JAX scalar counters. `attention_window=None` preserves released chunk-local semantics; a positive width opts into sliding attention. Both are invariant to call partitioning. Model and cache persistence use separate versioned formats bound to the exact configuration fingerprint.

## Testing

The suite covers:

- Paper-equation and released-source TimestepNorm examples with exact moments/state
- Source-derived Torch/JAX full-logit and every-parameter gradient parity
- Three-step AdamW parity and deterministic tiny-batch overfit
- Arbitrary cache partitioning, cache save/reload, and packed-document isolation
- Exact preset parameter counts, initialization distributions, non-trainable buffer audits, and strict checkpoint conversion
- FP32 and BF16 dtype behavior; FP16 rejection
- JIT stability: no retracing on repeated calls with same shapes
- GPU/CPU coverage via pytest fixtures

Before resuming an expensive training run, execute both complete gates from the repository root: `conda run --name mega-jax pytest` and `conda run --name mega-jax python tools/verify_modeling_correctness.py --include-slow`. The verifier reads the exact local paper/source, records source hashes and environment details, reports skipped runtime-only checks, writes machine-readable JSON, and exits nonzero on a failed invariant. A run without `--include-slow` records the omitted training/cache checks as skipped and is not a training-resumption gate. The original fused CUDA implementation is never built as part of this gate.
