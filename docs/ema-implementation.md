# EMA Implementation (JAX)

This note describes the JAX ComplexEMA implementation in this repo (`src/megalodon_jax/layers/complex_ema.py`) and how it relates to the upstream reference.

## Reference Context

- The original CUDA reference uses FFT-based convolution when no streaming state is required.
- The original CUDA reference uses fused FFT/state kernels. This repository validates those kernels at source level and uses a source-derived differentiable Torch recurrence for numerical parity without building the extension.

## JAX Implementation

The JAX version provides three numerically equivalent paths:

1. **FFT path (training / no state)** When no state is requested, ComplexEMA builds the EMA kernel and applies an FFT-based convolution (O(L log L)).

2. **Sequential path (streaming)** When `h_init` is provided or `return_state=True`, ComplexEMA runs the recurrence with `jax.lax.scan` (O(L)). The final complex state is returned only when `return_state=True`.

3. **Segmented path (packed training)** When `segment_ids` is provided, the EMA state resets at every segment boundary so packed documents cannot leak into each other. The default implementation is a parallel `jax.lax.associative_scan` over `(A, b)` affine pairs with `A = 0` at boundaries; a sequential low-memory fallback is available via `MegalodonConfig(use_associative_segment_scan=False)` (threaded down to every layer's CEMA). The FFT path cannot express resets and is bypassed. See [dev.md](dev.md#packed-sequence-state-isolation) for benchmarks and design notes.

### Path Selection

- **Segmented** when `segment_ids` is provided (incompatible with `h_init`; training-only)
- **FFT** when `h_init is None` and `return_state is False`
- **Sequential** otherwise

### Mask Handling

If a boolean `mask` is provided, masked positions are zeroed before the recurrence so they cannot contaminate EMA state. With `segment_ids`, positions in segment 0 are additionally invalid, composing with `mask`. Token IDs alone never imply padding.

## Stability Notes

- Parameters are stored in float32 to avoid bf16 quantization of EMA dynamics.
- Coefficients are computed in float32/complex64 regardless of parameter dtype.
- `|q| < 1` by construction, ensuring a decaying impulse response.
- Kernel powers use magnitude/phase (`|q|^t * exp(i * phi * t)`) to avoid numerical issues with cumprod or `log(0)`.

## Performance Notes

The sequential path is much slower than the FFT path in pure JAX. For training, keep `return_cache=False` so the FFT path is used; use the sequential path only for streaming inference or when you explicitly need the EMA state. The segmented associative path is ~5x slower than FFT on GPU (training-viable); measured numbers are in [dev.md](dev.md#packed-sequence-state-isolation).
