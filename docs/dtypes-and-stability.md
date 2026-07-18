# Dtypes and numerical stability

Megalodon JAX supports FP32 and BF16 compute. Ordinary embedding and projection parameters may use FP32 or BF16 storage, while precision-sensitive parameters and accumulation remain FP32.

> [!IMPORTANT]
> Float16 is never supported - the EMA, FFT, normalization, and long-context state paths are not reliable in that range, and config validation rejects it. Select BF16 through `MegalodonConfig`; never blanket-cast the model tree (see [Do not cast the model tree](#do-not-cast-the-model-tree)).

These are intentional JAX policies, not a promise to reproduce the released trainer's FairScale/FSDP1 storage mechanics. The normative distinction between released-model parity and supported precision policy is in [Upstream parity and production contracts](upstream-parity-contract.md#precision-policies).

---

- [Supported policies](#supported-policies)
- [Fixed precision behavior](#fixed-precision-behavior)
- [Do not cast the model tree](#do-not-cast-the-model-tree)
- [Training loop](#training-loop)
- [Memory-bounded loss head](#memory-bounded-loss-head)
- [Single-host data parallelism](#single-host-data-parallelism)
- [Checkpoint dtypes](#checkpoint-dtypes)
- [Troubleshooting](#troubleshooting)

---

## Supported policies

`MegalodonConfig` separates the numerical roles explicitly:

- `param_dtype` controls ordinary embedding and projection storage and may be `jnp.float32` or `jnp.bfloat16`; CEMA, normalization, Q/K affine, and residual-scale parameters remain FP32.
- `compute_dtype` controls projections and activations and may be `jnp.float32` or `jnp.bfloat16`.
- `accum_dtype` controls GEMM/reduction accumulation and must be `jnp.float32`.
- `attention_softmax_dtype` controls attention softmax and may be FP32 or BF16.
- `loss_softmax_dtype` controls language-model log-softmax and may be FP32 or BF16.

Full FP32 is the reference policy:

```python
config = MegalodonConfig(
    ...,
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    accum_dtype=jnp.float32,
    attention_softmax_dtype=jnp.float32,
    loss_softmax_dtype=jnp.float32,
)
```

BF16 compute with FP32 ordinary parameter storage retains the best update fidelity:

```python
config = MegalodonConfig(
    ...,
    param_dtype=jnp.float32,
    compute_dtype=jnp.bfloat16,
    accum_dtype=jnp.float32,
    attention_softmax_dtype=jnp.float32,
    loss_softmax_dtype=jnp.float32,
)
```

BF16 compute with compact ordinary parameter storage halves storage for the embedding and dense projections while retaining the sensitive FP32 subset:

```python
config = MegalodonConfig(
    ...,
    param_dtype=jnp.bfloat16,
    compute_dtype=jnp.bfloat16,
    accum_dtype=jnp.float32,
    attention_softmax_dtype=jnp.float32,
    loss_softmax_dtype=jnp.float32,
)
```

The split keeps FP32 where it is numerically useful without materially eroding compact storage. In the exact paper-7B configuration, the sensitive subset is 9,445,376 of 7,385,817,088 parameters (0.127885%). Keeping that subset FP32 costs 18,890,752 bytes, about 18 MiB, over an unsupported all-BF16 tree.

A canonical packed `B=4, L=2048` forward/backward audit with gradient checkpointing and BF16 compute measured the storage choice directly. Both policies produced finite loss and gradients and passed the independent packed-loss reference. GB below means 10^9 bytes.

| Ordinary parameter storage | Parameter storage | Compiler peak | Compiler temporary | Measured peak device | Synchronized runtime |
| -------------------------- | ----------------: | ------------: | -----------------: | -------------------: | -------------------: |
| FP32                       |          0.686 GB |      3.385 GB |           2.013 GB |             3.458 GB |           541.486 ms |
| BF16                       |          0.345 GB |      2.711 GB |           2.021 GB |             2.948 GB |           544.811 ms |

These single-iteration measurements are decision evidence for the canonical model and tested device, not portable memory limits or performance guarantees. They show that compact storage nearly halves parameter bytes and lowers observed total memory without materially changing runtime; compiler temporaries remain dominated by execution rather than parameter storage.

These choices are explicit model configuration, not ambient autocast state. Native checkpoints serialize every dtype field and restore it automatically when loaded.

Use BF16 only on accelerators with native BF16 support. There is no FP16 fallback for older GPUs; use FP32 instead.

## Fixed precision behavior

- TimestepNorm state, running moments, RMSNorm statistics, and LayerNorm statistics are FP32.
- TimestepNorm uses shifted FP32 first/second-moment prefixes only for fresh unmasked input with no learned prior (`prior_count=0`). Masked, packed, learned-prior, and continuation inputs use associative FP32 moment merging. The [paper/source differences](paper-deviations.md#released-source-compatibility-choices) describe the CUDA compensation-state divergence.
- TimestepNorm and cache position counters are int32 and guard against overflow; the released TimestepNorm count is int64, which is not enabled implicitly because JAX x64 is a global execution policy.
- CEMA coefficients and state are FP32/complex64.
- RoPE angles are generated in FP32 and are derived data, not trainable leaves. A model call materializes one broadcast-ready cosine/sine table from the resolved timeline and shares it across the layer stack; it is neither serialized nor exposed to optimizers.
- Fresh BF16 ordinary parameters are sampled with the configured initializer in FP32 and then cast to BF16, avoiding the coarse random grid produced by direct BF16 sampling.
- FP32 matrix contractions request JAX's per-operation `HIGHEST` precision, so NVIDIA GPUs do not silently substitute TensorFloat-32 products. BF16 contractions retain BF16 inputs with FP32 accumulation.
- BF16 result buffers are selected by consumer. Biasless projections whose public output is BF16 and the attention probability-times-value contraction return BF16 directly while using the explicit `BF16_BF16_F32` contraction algorithm. Biasful projections retain an FP32 result through bias addition and downcast once afterward. The language-model head also retains its required FP32 output.
- Attention QK scores are always FP32 because they accumulate in `accum_dtype`, which config validation fixes at FP32; a BF16 `attention_softmax_dtype` casts the scores only afterward. The resulting probabilities return to `compute_dtype` before dropout and the value contraction, matching the released mixed-precision boundary.
- KV cache tensors follow `compute_dtype`; norm state remains FP32 and EMA state remains complex64.
- Returned logits are always FP32, matching the original released model.
- Parameter gradients follow parameter storage: ordinary gradients are BF16 in compact storage mode, while sensitive gradients remain FP32.

The two softmax fields are deliberately independent. Changing `attention_softmax_dtype` does not alter loss math, and changing `loss_softmax_dtype` does not alter attention. FP32 is recommended for both. BF16 attention softmax is an opt-in direct BF16 reduction, not an approximation/error-corrected kernel and not released-source parity; its probabilities still return to `compute_dtype` before the value contraction. The selection is serialized with the rest of the configuration.

## Do not cast the model tree

Do not blanket-cast the model to BF16:

```python
# Unsupported: this also quantizes precision-sensitive parameters and derived arrays.
model = jax.tree.map(lambda x: x.astype(jnp.bfloat16), model)
```

Select BF16 compute and storage through `MegalodonConfig`. The model constructors apply `param_dtype` only to ordinary parameters and preserve the FP32-sensitive subset. `audit_sensitive_param_dtypes` is available as a defensive check:

```python
from megalodon_jax.precision import audit_sensitive_param_dtypes

assert not audit_sensitive_param_dtypes(model)
```

## Training loop

The model owns compute casting; token IDs remain `int32`. The model does not own optimizer state: initialize it according to the downstream policy and preserve each model leaf's configured dtype after updates. The example uses Optax, which is not a runtime dependency (`pip install optax`).

Released upstream training applies one uniform AdamW decay to the entire trainable parameter tree, with no exclusions for embeddings, biases, normalization parameters, TimestepNorm priors, CEMA parameters, or per-head affine parameters. For upstream-faithful training, set `weight_decay=0.1` explicitly and do not pass an Optax mask. The [released plus-one normalization storage](paper-deviations.md#released-source-compatibility-choices) is load-bearing under this policy: decay moves a stored scale offset toward zero, which moves its effective scale toward one rather than zero.

Bare Optax AdamW initializes both moment trees from the parameter tree's leaf dtypes. Its `mu_dtype` option can override only the first moment; the second moment still follows the corresponding parameter dtype. AdamW stores moments and a step count, not FP32 master parameters. Compact-storage training must therefore choose and test its optimizer-state dtypes and update-cast behavior deliberately rather than assuming that AdamW supplies hidden FP32 update fidelity.

The minimal loop below deliberately chooses the memory-first policy by initializing AdamW from the mixed model tree. Ordinary parameters, gradients, moments, and applied updates remain BF16; the sensitive subset remains FP32. This is a supported compact policy, not a claim of upstream optimizer-trajectory parity or hidden FP32 master weights.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from megalodon_jax import MegalodonConfig, MegalodonForCausalLM

config = MegalodonConfig(
    ...,
    param_dtype=jnp.bfloat16,
    compute_dtype=jnp.bfloat16,
    accum_dtype=jnp.float32,
    attention_softmax_dtype=jnp.float32,
    loss_softmax_dtype=jnp.float32,
)
model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.1)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_value_and_grad
def loss_fn(candidate, input_ids, labels, key):
    return candidate.compute_loss(input_ids, labels, deterministic=False, key=key)


def train_step(candidate, state, input_ids, labels, key):
    loss, grads = loss_fn(candidate, input_ids, labels, key)
    updates, state = optimizer.update(grads, state, candidate)
    return eqx.apply_updates(candidate, updates), state, loss
```

For exact gradient accumulation across packed microbatches with different numbers of valid targets, differentiate each microbatch's summed loss, add its gradients, and divide the accumulated gradients by the total valid-target count before the optimizer update. Request both quantities from the model instead of averaging microbatch means:

```python
loss_sum, valid_count = model.compute_loss(
    input_ids,
    labels,
    segment_ids=segment_ids,
    reduction="sum",
    return_valid_count=True,
)
```

## Memory-bounded loss head

`compute_loss(..., loss_chunk_size=N)` is an opt-in training path for shapes where FP32 vocabulary logits constrain the usable batch or sequence length. The ordinary path materializes logits with shape `(batch, sequence, vocabulary)`; for `B=4`, `L=4096`, and `V=32000`, one such FP32 tensor is about 2 GiB before backward storage and compiler workspace.

The bounded path runs the model body once, flattens the shifted hidden states across batch and sequence, and uses a static `lax.scan` to project at most `N` token states per iteration. The projection and cross-entropy body is rematerialized during backward, so vocabulary-sized chunk intermediates are recomputed instead of retained across the scan. Only the loss head is chunked: attention, CEMA, TimestepNorm, packed-document isolation, masks, label shifting, reductions, and valid-token counts keep the same contracts.

```python
loss = model.compute_loss(
    input_ids,
    labels,
    segment_ids=segment_ids,
    loss_chunk_size=64,
)
```

`loss_chunk_size` counts shifted token states across the flattened batch, not attention chunks or documents. It must be a positive static integer and specializes the compiled loss function. `None` remains the default and preserves the unbounded full-logits path. `reduction="none"`, `"sum"`, and `"mean"` are all supported, as is `return_valid_count=True`; normal `model(...)` calls always return complete FP32 logits.

Choose the largest chunk that fits the target training shape. Smaller chunks bound head intermediates more aggressively but may reduce matrix-multiplication efficiency and repeat more work during backward. Compare compiler temporaries, observed device peak, and synchronized forward/backward time rather than assuming one portable value. The production benchmark accepts the same switch:

```bash
python benchmarks/benchmark_model_paths.py \
  --repo current=. \
  --suite training \
  --training-operations forward_backward \
  --training-modes plain,packed \
  --training-lengths 2048 \
  --training-batches 1 \
  --loss-chunk-size 64 \
  --output local-scratch/model-paths-loss-chunk-64.json
```

Run an otherwise identical report without `--loss-chunk-size` for the full-logits baseline. Keep model topology, dtype policy, seed, warmups, iterations, compiler configuration, and GPU workload identical between reports.

## Single-host data parallelism

JAX named sharding can split only the leading batch axis while replicating the mixed-dtype model and optimizer state. Attached shardings are honored by the filtered JIT-compiled step, and the global mean in `compute_loss` reduces across the complete sharded batch. No explicit `pmean` is needed with this global-array form.

```python
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = np.asarray(jax.devices())
if input_ids.shape[0] % devices.size:
    raise ValueError("global batch size must be divisible by the number of devices")

mesh = Mesh(devices, ("data",))
replicated = NamedSharding(mesh, P())
batch_sharded = NamedSharding(mesh, P("data"))

model, opt_state = eqx.filter_shard((model, opt_state), replicated)
input_ids, labels = eqx.filter_shard((input_ids, labels), batch_sharded)
key = eqx.filter_shard(key, replicated)

compiled_train_step = eqx.filter_jit(train_step)
model, opt_state, loss = compiled_train_step(model, opt_state, input_ids, labels, key)
```

`eqx.filter_shard` applies the JAX sharding only to array leaves, ignores static Python metadata, and preserves every leaf's configured dtype. This example assumes one host and a batch whose leading dimension divides evenly over its local devices; multi-host input assembly also needs process-aware data loading.

## Checkpoint dtypes

Native model checkpoints preserve the exact mixed parameter tree and store `param_dtype`, compute, accumulation, and both softmax dtypes in configuration metadata. Loading reconstructs the model from that configuration before validating and restoring tensor dtypes, so callers do not need to repeat training-time dtype flags.

`export_upstream_state_dict(model)` exports ordinary embedding, projection, and output tensors in the model's storage dtype. Its optional `dtype` override may cast those ordinary tensors for transport. CEMA, normalization, RoPE, and other sensitive values remain FP32. Loading casts ordinary upstream tensors to `param_dtype` while retaining sensitive values in FP32. Format versions, strict loading, partial restore, and conversion behavior are described in [JAX and PyTorch interoperability](jax-torch.md).

## Troubleshooting

If BF16 training diverges:

1. Confirm `accum_dtype` is FP32 and restore FP32 attention and loss softmax.
2. Run `audit_sensitive_param_dtypes(model)`.
3. Remove downstream tree-wide casts.
4. Compare with `param_dtype=jnp.float32` to isolate BF16 parameter-update quantization.
5. Confirm the accelerator has native BF16 support.
6. Reproduce the issue in full FP32 before changing model mathematics.
