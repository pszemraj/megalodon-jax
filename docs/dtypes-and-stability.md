# Dtypes and numerical stability

Megalodon JAX supports two numerical modes: FP32 and BF16 compute with FP32 storage/accumulation. Float16 is intentionally unsupported because the EMA, FFT, normalization, and long-context state paths are not reliable in that range.

## Supported policies

`MegalodonConfig` separates the numerical roles explicitly:

- `param_dtype` is parameter storage and must be `jnp.float32`.
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

BF16 compute is the accelerator policy:

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

Use BF16 only on accelerators with native BF16 support. There is no FP16 fallback for older GPUs; use FP32 instead.

## Fixed precision behavior

- TimestepNorm state, running moments, RMSNorm statistics, and LayerNorm statistics are FP32.
- TimestepNorm uses shifted FP32 first/second-moment prefixes for unmasked input and associative FP32 Welford prefixes for masked or packed input. Unlike the released CUDA kernel, none of these paths stores Kahan compensation terms.
- TimestepNorm and cache position counters are int32 and guard against overflow; the released TimestepNorm count is int64, which is not enabled implicitly because JAX x64 is a global execution policy.
- CEMA coefficients and state are FP32/complex64.
- RoPE angles are generated in FP32 and are derived data, not trainable leaves.
- FP32 matrix contractions request JAX's per-operation `HIGHEST` precision, so NVIDIA GPUs do not silently substitute TensorFloat-32 products. BF16 contractions retain BF16 inputs with FP32 accumulation.
- KV cache tensors follow `compute_dtype`; norm state remains FP32 and EMA state remains complex64.
- Returned logits are always FP32, matching the original released model.
- Parameter gradients have the FP32 storage dtype even under BF16 compute.

The two softmax fields are deliberately independent. Changing `attention_softmax_dtype` does not alter loss math, and changing `loss_softmax_dtype` does not alter attention. FP32 is recommended for both. BF16 attention softmax is an experimental JAX tradeoff, not released-source parity.

## Do not cast the model tree

Do not blanket-cast the model to BF16:

```python
# Unsupported: this quantizes persistent parameters and sensitive dynamics.
model = jax.tree.map(lambda x: x.astype(jnp.bfloat16), model)
```

Select BF16 through `MegalodonConfig(compute_dtype=jnp.bfloat16)`. The model casts compute operands while retaining FP32 master parameters. `audit_sensitive_param_dtypes` is available as a defensive check:

```python
from megalodon_jax.precision import audit_sensitive_param_dtypes

assert not audit_sensitive_param_dtypes(model)
```

## Training loop

The model owns compute casting; token IDs remain `int32`, and optimizer state should be initialized from the FP32 parameter tree. The example uses Optax, which is not a runtime dependency (`pip install optax`).

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

config = MegalodonConfig(
    ...,
    compute_dtype=jnp.bfloat16,
    attention_softmax_dtype=jnp.float32,
    loss_softmax_dtype=jnp.float32,
)
model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
optimizer = optax.adamw(learning_rate=1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_value_and_grad
def loss_fn(candidate, input_ids, labels, key):
    return candidate.compute_loss(input_ids, labels, deterministic=False, key=key)

def train_step(candidate, state, input_ids, labels, key):
    loss, grads = loss_fn(candidate, input_ids, labels, key)
    updates, state = optimizer.update(grads, state, candidate)
    return eqx.apply_updates(candidate, updates), state, loss
```

## Checkpoint dtypes

Native model checkpoints preserve the FP32 parameter contract and store the dtype policy in configuration metadata. Loading converts original-upstream floating tensors to FP32 parameter storage.

`export_upstream_state_dict(model, dtype=torch.bfloat16)` casts ordinary embedding, projection, and output tensors for transport while preserving CEMA, normalization, RoPE, and other sensitive values in FP32. Format versions, strict loading, partial restore, and conversion behavior are described in [JAX and PyTorch interoperability](jax-torch.md).

## Troubleshooting

If BF16 training diverges:

1. Confirm `param_dtype` and `accum_dtype` are FP32.
2. Restore FP32 attention and loss softmax.
3. Run `audit_sensitive_param_dtypes(model)`.
4. Remove downstream tree-wide casts.
5. Confirm the accelerator has native BF16 support.
6. Reproduce the issue in full FP32 before changing model mathematics.
