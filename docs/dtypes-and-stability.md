# Dtypes and Stability Guide

This guide explains how to use Megalodon JAX safely in downstream training
loops with fp32 or bf16 compute. The core rule is: keep sensitive parameters in
fp32 while allowing compute to run in bf16 via the config.

## Recommended dtype policy

MegalodonConfig separates:

- `param_dtype`: parameter storage dtype (default fp32).
- `compute_dtype`: matmul/activation dtype (default fp32).
- `accum_dtype`: accumulation dtype for GEMM and reductions (default fp32).
- `softmax_dtype`: softmax/log-softmax dtype (default fp32).
- `gemm_backend`: `"default"` (reserved for future FP8 backends).

Use one of the following:
TODO: implement the `mxfp8` and `nvfp4` GEMM backends and relax config validation once they exist.

### 1) Full fp32 (most stable)

```python
config = MegalodonConfig(
    ...,
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    accum_dtype=jnp.float32,
    softmax_dtype=jnp.float32,
)
```

### 2) AMP-style bf16 compute (recommended for speed)

```python
config = MegalodonConfig(
    ...,
    param_dtype=jnp.float32,
    compute_dtype=jnp.bfloat16,
    accum_dtype=jnp.float32,
    softmax_dtype=jnp.float32,
)
```

This keeps master weights in fp32 while running most compute in bf16 with
fp32 accumulation where needed.

## What not to do

Do not blanket-cast the entire model to bf16, for example:

```python
# Avoid this: it quantizes sensitive params.
model = jax.tree.map(lambda x: x.astype(jnp.bfloat16), model)
```

This will quantize EMA parameters, normalization weights, and per-head Q/K
affines, which can destabilize training.

If you must cast a model, call the precision helpers afterward:

```python
from megalodon_jax.precision import ensure_sensitive_param_dtype

model = jax.tree.map(...)
model = ensure_sensitive_param_dtype(model)
```

## Precision-sensitive parameters

The following are always expected to stay fp32:

- ComplexEMA parameters: `alpha`, `delta`, `theta`, `gamma_real`, `gamma_imag`, `omega`
- Norm parameters: RMSNorm, LayerNorm, and TimestepNorm weights/biases
- Per-head Q/K affine parameters: `gamma`, `beta`

You can audit these at runtime:

```python
from megalodon_jax.precision import audit_sensitive_param_dtypes

mismatches = audit_sensitive_param_dtypes(model)
assert not mismatches, mismatches
```

## Training loop guidance (fp32 or bf16 compute)

The model already casts activations to `compute_dtype`. Use config-driven
precision rather than manual casting in your loop.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

config = MegalodonConfig(
    ...,
    param_dtype=jnp.float32,
    compute_dtype=jnp.bfloat16,
    accum_dtype=jnp.float32,
    softmax_dtype=jnp.float32,
)
model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

optimizer = optax.adamw(learning_rate=1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_value_and_grad
def loss_fn(m, input_ids, labels, key):
    return m.compute_loss(input_ids, labels, deterministic=False, key=key)

def train_step(m, opt_state, input_ids, labels, key):
    loss, grads = loss_fn(m, input_ids, labels, key)
    updates, opt_state = optimizer.update(grads, opt_state, m)
    m = eqx.apply_updates(m, updates)
    return m, opt_state, loss
```

Notes:
- Keep `param_dtype` as fp32 to maintain master weights.
- Let the model handle compute casting; avoid manual `astype` on the model.
- Input token ids should remain int32.
- Loss is computed in `softmax_dtype` (default fp32) for stability.

## Inference and cache dtype

Use `init_cache(config, ...)` without a dtype override so cache dtypes match
`config.compute_dtype`. This avoids unexpected dtype promotion inside attention.

## Loading and conversion

- `load_from_pretrained(..., dtype=jnp.bfloat16)` will cast non-sensitive
  parameters to bf16 while keeping sensitive params in fp32.
- `load_weights_from_torch` now enforces fp32 for sensitive params after load.
- `convert_jax_to_torch(..., dtype=torch.bfloat16)` exports sensitive params in
  fp32 even when the rest is bf16.

## Troubleshooting checklist

If bf16 training looks unstable:

1. Verify config uses `param_dtype=jnp.float32`, `compute_dtype=jnp.bfloat16`,
   and `softmax_dtype=jnp.float32`.
2. Run `audit_sensitive_param_dtypes(model)` and ensure no mismatches.
3. Remove any `jax.tree.map` casts applied in downstream code.
4. Confirm you are not using float16 (unsupported).
