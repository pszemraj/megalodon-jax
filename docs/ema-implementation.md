# ComplexEMA implementation

`ComplexEMA` in [`complex_ema.py`](../src/megalodon_jax/layers/complex_ema.py) provides FFT, recurrent, and reset-aware execution paths with the same released coefficient semantics.

## Released behavior

- The original CUDA reference uses FFT-based convolution when no streaming state is required.
- The original CUDA reference uses fused FFT/state kernels. The JAX implementation reproduces their recurrence without requiring the extension at runtime. Coefficient and residual choices that differ from the abbreviated paper equation are listed in [Paper and source differences](paper-deviations.md#released-source-compatibility-choices).

## Execution paths

The JAX version provides four execution paths:

1. **FFT path (training / no state)** When no state is requested, ComplexEMA builds the EMA kernel and applies an FFT-based convolution (O(L log L)).

2. **Pristine prefill path (FFT output + compact state recurrence)** When `h_init` is absent and `return_state=True`, outputs use FFT convolution while a state-only `jax.lax.scan` carries `(batch, dim, ndim)` complex state and emits no per-token outputs.

3. **Sequential continuation path** When `h_init` is provided, ComplexEMA runs the recurrence with `jax.lax.scan` (O(L)) because every output depends on incoming history.

4. **Segmented path (packed training)** When `segment_ids` is provided, the EMA state resets at every segment boundary so packed documents cannot leak into each other. The FFT path cannot express resets and is bypassed.

### Path selection

- **Segmented** when `segment_ids` is provided (incompatible with `h_init`; training-only)
- **FFT output only** when `h_init is None` and `return_state is False`
- **FFT output plus compact state recurrence** when `h_init is None` and `return_state is True`
- **Sequential output/state** when `h_init` is provided

### Mask handling

If a boolean `mask` is provided, masked positions are zeroed before the recurrence so they cannot contaminate EMA state. With `segment_ids`, positions in segment 0 are additionally invalid, composing with `mask`. Token IDs alone never imply padding.

## Segmented path

The default packed path applies `jax.lax.associative_scan` to complex affine pairs `(A, b)`, with `A = 0` at every boundary. It materializes two `(length, batch, model_dim, cema_ndim)` complex64 tensors and favors GPU throughput. Set `MegalodonConfig(use_associative_segment_scan=False)` to use the O(1)-extra-memory sequential fallback when that allocation is too large.

The sequential fallback is substantially slower on GPU, so choose it from measured memory constraints rather than as a general default. [`benchmark_cema_reset.py`](../benchmarks/benchmark_cema_reset.py) isolates the FFT, associative, and sequential paths; the invocation and evidence requirements are in [Development](dev.md#performance-benchmarks).

## Stability

- Parameters are stored in float32 to avoid bf16 quantization of EMA dynamics.
- Coefficients are computed in float32/complex64 regardless of parameter dtype.
- `|q| < 1` by construction, ensuring a decaying impulse response.
- Kernel powers use magnitude/phase (`|q|^t * exp(i * phi * t)`) to avoid numerical issues with cumprod or `log(0)`.

## Performance

Sequential continuation is slower than FFT convolution in pure JAX. Training and pristine prompt prefill use FFT outputs; only continuation from nonzero history uses sequential outputs. Packed execution adds reset semantics and should be benchmarked at the intended batch, sequence, model, and CEMA dimensions because its associative-path memory scales with their product.
