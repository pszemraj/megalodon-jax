# Dtypes and numerical stability

Megalodon JAX supports FP32 and BF16 compute. Ordinary embedding and projection parameters may use FP32 or BF16 storage, while precision-sensitive parameters and accumulation remain FP32. Float16 is intentionally unsupported because the EMA, FFT, normalization, and long-context state paths are not reliable in that range.

These are intentional JAX policies, not a promise to reproduce the released trainer's FairScale/FSDP1 storage mechanics. The normative distinction between released-model parity and supported precision policy is in [Upstream parity and production contracts](upstream-parity-contract.md#precision-policies).

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

These choices are explicit model configuration, not ambient autocast state. Native checkpoints serialize every dtype field and restore it automatically when loaded.

Use BF16 only on accelerators with native BF16 support. There is no FP16 fallback for older GPUs; use FP32 instead.

## Fixed precision behavior

- TimestepNorm state, running moments, RMSNorm statistics, and LayerNorm statistics are FP32.
- TimestepNorm uses shifted FP32 first/second-moment prefixes only for fresh unmasked input with no learned prior (`prior_count=0`). Masked, packed, learned-prior, and continuation inputs use associative FP32 moment merging. The [paper/source differences](paper-deviations.md#released-source-compatibility-choices) describe the CUDA compensation-state divergence.
- TimestepNorm and cache position counters are int32 and guard against overflow; the released TimestepNorm count is int64, which is not enabled implicitly because JAX x64 is a global execution policy.
- CEMA coefficients and state are FP32/complex64.
- RoPE angles are generated in FP32 and are derived data, not trainable leaves.
- Fresh BF16 ordinary parameters are sampled with the configured initializer in FP32 and then cast to BF16, avoiding the coarse random grid produced by direct BF16 sampling.
- FP32 matrix contractions request JAX's per-operation `HIGHEST` precision, so NVIDIA GPUs do not silently substitute TensorFloat-32 products. BF16 contractions retain BF16 inputs with FP32 accumulation.
- Attention scores and softmax are FP32 by default. The resulting probabilities return to `compute_dtype` before dropout and the value contraction, matching the released mixed-precision boundary.
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

Select BF16 compute and storage through `MegalodonConfig`. The constructors apply `param_dtype` only to ordinary parameters and preserve the FP32-sensitive subset. `audit_sensitive_param_dtypes` is available as a defensive check:

```python
from megalodon_jax.precision import audit_sensitive_param_dtypes

assert not audit_sensitive_param_dtypes(model)
```

## Training loop

The model owns compute casting; token IDs remain `int32`, and optimizer state should be initialized from the model's mixed parameter tree. The example uses Optax, which is not a runtime dependency (`pip install optax`).

Released upstream training applies one uniform AdamW decay to the entire trainable parameter tree, with no exclusions for embeddings, biases, normalization parameters, TimestepNorm priors, CEMA parameters, or per-head affine parameters. For upstream-faithful training, set `weight_decay=0.1` explicitly and do not pass an Optax mask. The [released plus-one normalization storage](paper-deviations.md#released-source-compatibility-choices) is load-bearing under this policy: decay moves a stored scale offset toward zero, which moves its effective scale toward one rather than zero.

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

`eqx.filter_shard` applies the JAX sharding only to array leaves, leaves static Python metadata alone, and preserves every leaf's configured dtype. This example assumes one host and a batch whose leading dimension divides evenly over its local devices; multi-host input assembly also needs process-aware data loading.

Compact BF16 storage is intentional pure-BF16 updating for ordinary parameters: their gradients and applied updates are quantized to BF16. An optimizer may retain some or all accumulator state in FP32, but that does not create an FP32 master copy of the parameters. Use `param_dtype=jnp.float32` with BF16 compute when FP32 master-parameter update fidelity is required.

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
