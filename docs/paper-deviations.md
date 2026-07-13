# Paper, released source, and intentional JAX differences

Paper theory, released checkpoint semantics, and intentional JAX extensions are treated as separate compatibility targets.

## Released-source compatibility choices

- **CEMA coefficient placement.** Paper equation (2) can be read as applying phase to both input and recurrence terms. The released implementation uses real `p = sigmoid(alpha)` and places phase in complex `q = polar(1 - alpha * delta, theta)`. JAX follows the released behavior so original checkpoints retain their meaning.
- **CEMA omega residual.** The `omega`-weighted skip is not explicit in the abbreviated paper equation but is a released trainable parameter and is preserved.
- **Normalized Q/K.** The paper's L2 normalization is implemented as per-head RMS normalization followed by the released `1/sqrt(d_head)` affine factor. Attention scores do not receive an additional Transformer `1/sqrt(d_head)` scale.
- **Normalization storage.** TimestepNorm, LayerNorm, and RMSNorm store zero-initialized scale offsets and apply `stored + 1`. This is forward-equivalent to direct unit scale at initialization but materially different for serialization and weight decay, so the released storage convention is preserved.
- **TimestepNorm moments.** Each valid token contributes a block containing every scalar in a group. Population variance includes within-token feature variance, and continuation uses block-Welford merging. Masked positions do not update state and emit exact zeros. No variance floor is added beyond configured `norm_eps`.
- **TimestepNorm accumulation precision.** The released CUDA kernel wraps Welford updates in Kahan-compensated accumulation. JAX's [FP32 moment paths](dtypes-and-stability.md#fixed-precision-behavior) do not store compensation. This precision-only divergence can accumulate additional drift on extremely long streams; adding compensation would change the serialized `NormState` and cache schema.
- **State counter width.** Released TimestepNorm stores its count as int64. JAX uses int32 counters for TimestepNorm and attention/cache positions because 64-bit JAX integers require the global x64 mode; construction and updates fail loudly before int32 overflow. This limits one uninterrupted state timeline to 2,147,483,647 tokens without changing ordinary-sequence mathematics.
- **RoPE coordinates.** Adjacent coordinate pairs are interpreted as complex values, matching the released `view_as_complex(...reshape(..., -1, 2))` convention. Frequencies are derived, non-trainable data.
- **Initialization.** Internal projections default to the released Gaussian form of `kaiming_normal_(a=sqrt(5))`, with standard deviation `1/sqrt(3 * fan_in)`. Embeddings and untied output heads use their separate model-dimension-based truncated-normal policy; `init_mode` does not override these boundary tensors.

## Intentional JAX extensions

- **Pure JAX/XLA execution.** The original uses custom CUDA kernels. JAX provides FFT, scan, and manual attention implementations. The fused extension is authoritative source evidence but is not a build or runtime prerequisite.
- **Packed training isolation.** `segment_ids` and optional `position_ids` isolate contiguous documents across attention, CEMA, TimestepNorm, RoPE, gradients, and shifted loss pairs. The complete contract is described in [Long-context streaming](long-context-streaming.md#packed-sequence-training).
- **Sliding attention.** `attention_window=None` preserves released chunk-local behavior. A positive `attention_window` opts into the partition-invariant sliding mode described in [Long-context streaming](long-context-streaming.md#optional-sliding-kv-window).
- **Dropout mode selection.** `attention_dropout_mode="post_softmax"` provides released unfused behavior; `"dropkey"` provides the pre-softmax compatibility option. Both require explicit PRNG keys in non-deterministic execution.
- **Versioned native persistence.** Native model and cache SafeTensors formats include version, configuration, and manifest metadata and fail closed. See [JAX and PyTorch interoperability](jax-torch.md).
- **No 4D chunk-parallel implementation.** Model-parallel checkpoint shards can be consolidated during conversion, but distributed chunk-parallel execution is outside this single-device implementation.

## Deliberately unsupported

- Float16 compute or storage. [Supported dtype policies](dtypes-and-stability.md#supported-policies) are FP32 and BF16 compute with FP32 parameters and accumulation.
- Silent loading of pre-v2 JAX, Hugging Face-shaped, raw FSDP, or otherwise ambiguous checkpoints.
- Cached decoding with padding masks or packed-sequence metadata because the KV cache does not store per-position validity/segment metadata.
