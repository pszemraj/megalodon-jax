# Paper, released source, and intentional JAX differences

Paper theory, released checkpoint semantics, and intentional JAX extensions are treated as separate compatibility targets. [Upstream parity and production contracts](upstream-parity-contract.md) defines the normative boundary between them.

## Named model presets

Preset names are explicit because the paper 7B configuration and the released configurations are not interchangeable. With `vocab_size=32_000`, the exact trainable counts are:

| Factory | SwiGLU | Chunk | RoPE base | Tied | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `from_upstream_mega200m` | no | 2,048 | 10,000 | no | 220,627,968 |
| `from_upstream_mega1_3b` | no | 2,048 | 10,000 | no | 1,342,832,640 |
| `from_upstream_mega1_3b_pg19` | yes | 2,048 | 10,000 | yes | 1,327,628,288 |
| `from_upstream_mega7_1b` | no | 2,048 | 10,000 | no | 7,117,381,632 |
| `from_upstream_mega7_3b` | yes | 2,048 | 10,000 | no | 7,385,817,088 |
| `from_paper_7b` | yes | 4,096 | 100,000 | no | 7,385,817,088 |

`from_7b()` intentionally raises because its historical definition mixed incompatible upstream presets. Use `config.parameter_count_breakdown()` for exact counts at another vocabulary size or output width. Output tying is controlled only by the explicit `share_emb` field; it is never inferred from matching shapes.

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
- **Packed training isolation.** Packed execution must isolate every unrelated contiguous document run and remain practically trainable; a correctness-only slow path is not sufficient. The user-facing semantics are described in [Long-context streaming](long-context-streaming.md#packed-sequence-training), and the production/performance contract is normative in [Upstream parity and production contracts](upstream-parity-contract.md#packed-training-is-a-required-production-capability).
- **Sliding attention.** The optional sliding mode is described in [Long-context streaming](long-context-streaming.md#optional-sliding-kv-window).
- **Dropout mode selection.** `attention_dropout_mode="post_softmax"` provides released unfused behavior; `"dropkey"` provides the pre-softmax compatibility option. Both require explicit PRNG keys in non-deterministic execution.
- **Versioned native persistence.** Model checkpoints record the complete configuration; cache files bind to its fingerprint. Both SafeTensors formats carry versioned manifests and fail closed. See [JAX and PyTorch interoperability](jax-torch.md).
- **No 4D chunk-parallel implementation.** Model-parallel checkpoint shards can be consolidated during conversion, but distributed chunk-parallel execution is outside this single-device implementation.

## Deliberately unsupported

- Float16 compute or storage. [Supported dtype policies](dtypes-and-stability.md#supported-policies) allow FP32 or BF16 ordinary parameters and compute while retaining FP32-sensitive parameters and accumulation.
- Silent loading of pre-v2 JAX, Hugging Face-shaped, raw FSDP, or otherwise ambiguous checkpoints.
- Cached decoding with padding masks or packed-sequence metadata because the KV cache does not store per-position validity/segment metadata.
