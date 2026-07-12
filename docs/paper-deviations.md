# Paper, Released Source, and Intentional JAX Differences

This document separates mathematical architecture, original released checkpoint compatibility, and intentional JAX extensions. A paper/source disagreement is not silently resolved in favor of either one.

## Released-source compatibility choices

- **CEMA coefficient placement.** Paper equation (2) can be read as applying phase to both input and recurrence terms. The released implementation uses real `p = sigmoid(alpha)` and places phase in complex `q = polar(1 - alpha * delta, theta)`. JAX follows the released behavior so original checkpoints retain their meaning.
- **CEMA omega residual.** The `omega`-weighted skip is not explicit in the abbreviated paper equation but is a released trainable parameter and is preserved.
- **Normalized Q/K.** The paper's L2 normalization is implemented as per-head RMS normalization followed by the released `1/sqrt(d_head)` affine factor. Attention scores do not receive an additional Transformer `1/sqrt(d_head)` scale.
- **Normalization storage.** TimestepNorm, LayerNorm, and RMSNorm store zero-initialized scale offsets and apply `stored + 1`. This is forward-equivalent to direct unit scale at initialization but materially different for serialization and weight decay, so the released storage convention is preserved.
- **TimestepNorm moments.** Each valid token contributes a block containing every scalar in a group. Population variance includes within-token feature variance, and continuation uses block-Welford merging. Masked positions do not update state and emit exact zeros. No variance floor is added beyond configured `norm_eps`.
- **TimestepNorm accumulation precision.** The released CUDA kernel wraps Welford updates in Kahan-compensated accumulation, while JAX evaluates the same block-Welford recurrence in plain FP32. This is a precision-only divergence that can accumulate additional drift on extremely long streams; adding compensation would change the serialized `NormState`/cache schema and requires an explicit format decision.
- **RoPE coordinates.** Adjacent coordinate pairs are interpreted as complex values, matching the released `view_as_complex(...reshape(..., -1, 2))` convention. Frequencies are derived, non-trainable data.

## Intentional JAX extensions

- **Pure JAX/XLA execution.** The original uses custom CUDA kernels. JAX provides FFT, scan, and manual attention implementations. The fused extension is authoritative source evidence but is not a build or runtime prerequisite.
- **Packed training isolation.** `segment_ids` and optional `position_ids` isolate contiguous documents across attention, CEMA, TimestepNorm, RoPE, gradients, and shifted loss pairs. This is training-only and is rejected on cached calls.
- **Sliding attention.** `attention_window=None` is faithful released chunk-local behavior. A positive `attention_window` opts into a fixed-capacity sliding window with an exact per-query age mask. Both modes are call-partition invariant.
- **Dropout mode selection.** `attention_dropout_mode="post_softmax"` provides released unfused behavior; `"dropkey"` provides the pre-softmax compatibility option. Both require explicit PRNG keys in non-deterministic execution.
- **Versioned native persistence.** Native model and cache SafeTensors formats include version/config/manifest metadata and fail closed. The original release did not define a JAX checkpoint format.
- **No 4D chunk-parallel implementation.** Model-parallel checkpoint shards can be consolidated during conversion, but distributed chunk-parallel execution is outside this single-device implementation.

## Deliberately unsupported

- Float16 compute or storage. Supported modes are FP32 and BF16 compute with FP32 parameters/accumulation.
- Silent loading of pre-v2 JAX, Hugging Face-shaped, raw FSDP, or otherwise ambiguous checkpoints.
- Cached decoding with padding masks or packed-sequence metadata because the KV cache does not store per-position validity/segment metadata.
