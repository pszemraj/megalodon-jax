# EMA Implementation (JAX)

This note describes the JAX ComplexEMA implementation in this repo
(`src/megalodon_jax/layers/complex_ema.py`) and how it relates to the upstream
reference.

## Upstream Reference (Context)

- The output is computed via FFT-based convolution when no streaming state is
  required.
- The final streaming state is produced by a fused kernel to avoid Python-side
  loops.

## JAX Implementation

The JAX version provides two numerically equivalent paths:

1. **FFT path (training / no state)**  
   When no state is requested, ComplexEMA builds the EMA kernel and applies an
   FFT-based convolution (O(L log L)).

2. **Sequential path (streaming)**  
   When `h_init` is provided or `return_state=True`, ComplexEMA runs the
   recurrence with `jax.lax.scan` (O(L)) and returns the final complex state.

### Path Selection

- **FFT** when `h_init is None` and `return_state is False`
- **Sequential** otherwise

### Mask Handling

If a boolean `mask` is provided, masked positions are zeroed before the
recurrence. This prevents padded tokens from contaminating the EMA state. This
is intentionally more conservative than the upstream behavior.

## Stability Notes

- Coefficients are computed in float32/complex64 regardless of parameter dtype.
- `|q| < 1` by construction, ensuring a decaying impulse response.
- Kernel powers use magnitude/phase (`|q|^t * exp(i * phi * t)`) to avoid
  numerical issues with cumprod or `log(0)`.

## Performance Notes

The sequential path is much slower than the FFT path in pure JAX. For training,
keep `return_cache=False` so the FFT path is used; use the sequential path only
for streaming inference or when you explicitly need the EMA state.
