# ComplexEMA implementation

`ComplexEMA` in [`complex_ema.py`](../src/megalodon_jax/layers/complex_ema.py) provides FFT, recurrent, and reset-aware execution paths with the same released coefficient semantics.

Packed CEMA isolation and useful accelerator throughput are required production behavior. The normative boundary and evidence gates are defined in [Upstream parity and production contracts](upstream-parity-contract.md#packed-training-is-a-required-production-capability).

## Released behavior

- The original CUDA reference uses FFT-based convolution when no streaming state is required.
- The original CUDA reference uses fused FFT/state kernels. The JAX implementation reproduces their recurrence without requiring the extension at runtime. Coefficient and residual choices that differ from the abbreviated paper equation are listed in [Paper and source differences](paper-deviations.md#released-source-compatibility-choices).

## Execution paths

The JAX version provides four execution paths:

1. **FFT path (training / no state)** When no state is requested, ComplexEMA builds the EMA kernel and applies an FFT-based convolution (O(L log L)).

2. **Pristine prefill path** Outputs use FFT convolution. When continuation state is requested, short inputs use a compact state-only recurrence and inputs of at least 32 tokens use the released closed-form final-state reduction in parallel.

3. **Cached continuation path** Token decode and cached chunks shorter than 32 tokens use the sequential recurrence. Longer chunks follow the released decomposition: FFT convolution for the driven response, a projected `h_init * q**(t+1)` bias for incoming history, and the parallel closed-form final-state reduction when a new state is requested.

4. **Segmented path (packed training)** When `segment_ids` is provided, the EMA state resets at every segment boundary so packed documents cannot leak into each other. The FFT path cannot express resets and is bypassed.

### Path selection

The dispatch below is the fine-grained version of the same four paths, split by the conditions that select them:

- **Segmented** when `segment_ids` is provided (incompatible with `h_init`; training-only)
- **FFT output only** when `h_init is None` and `return_state is False`
- **Sequential output/state** when `h_init` is provided and the static sequence length is below 32
- **FFT output plus sequential final-state recurrence** for a pristine input shorter than 32 tokens when `return_state is True`
- **FFT output plus parallel final-state reduction** for a pristine input of at least 32 tokens when `return_state is True`
- **FFT output plus initial-state bias and optional parallel final-state reduction** when `h_init` is provided and the static sequence length is at least 32

### Mask handling

If a boolean `mask` is provided, masked positions are zeroed before the recurrence so they cannot contaminate EMA state. With `segment_ids`, positions in segment 0 are additionally invalid, composing with `mask`. Token IDs alone never imply padding.

## Non-segmented closed form

For an incoming state `h_init`, the released source computes long-chunk outputs as the sum of the input convolution and the decaying contribution of that state. It computes the final state directly rather than replaying a token loop:

```text
h_L = q**L * h_init + p * sum(x[L - 1 - j] * q**j, j=0..L-1)
```

The JAX path uses explicit real/imaginary multiply-reductions instead of a complex batched dot because the latter caused an excessive cuBLAS autotuning allocation at target shapes. Coefficient powers are generated in blocks of at most 4,096 positions and each final-state block is reduced immediately. The 32-token switch is shape-static, so it does not add data-dependent dispatch inside a compiled call. It keeps tokenwise generation on the low-overhead recurrence while removing a serial sequence loop from prompt prefill and from substantial cache updates.

## Segmented path

The default packed path applies `jax.lax.associative_scan` to complex affine pairs `(A, b)`, with `A = 0` at every boundary. It materializes two `(length, batch, model_dim, cema_ndim)` complex64 operands and is the production throughput path. Set `MegalodonConfig(use_associative_segment_scan=False)` to use the sequential fallback when the associative allocation is unsuitable.

The sequential path has a compact forward carry, but automatic differentiation can still produce sequence-scaled compiler temporaries; it must not be described as an O(1)-memory training solution without compiled backward measurements. It is generally slower on GPU and does not by itself satisfy the efficient packed-training contract. Choose it from measured end-to-end constraints rather than as a general default. [`benchmark_cema_reset.py`](../benchmarks/benchmark_cema_reset.py) isolates the FFT, associative, and sequential paths; the invocation and evidence requirements are in [Development](dev.md#packed-cema-gate).

## Stability

- Parameters are stored in float32 to avoid bf16 quantization of EMA dynamics.
- Coefficients are computed in float32/complex64 regardless of parameter dtype.
- `|q| < 1` by construction, ensuring a decaying impulse response.
- Kernel powers use magnitude/phase (`|q|^t * exp(i * phi * t)`) to avoid numerical issues with cumprod or `log(0)`.

## Performance

The hybrid continuation boundary was measured at `B=1, D=1024, N=16` on the reference RTX 5090. These figures are decision evidence rather than portable thresholds.

| Cached chunk length | FFT/bias/parallel-state | Sequential recurrence |
| ------------------: | ----------------------: | --------------------: |
|                   1 |                0.075 ms |              0.053 ms |
|                  32 |                0.074 ms |              0.232 ms |
|                 512 |                0.262 ms |              2.945 ms |
|               2,048 |                1.291 ms |              11.46 ms |

The production benchmark confirmed the effect on the canonical 12-layer, 171M-parameter model with FP32 ordinary storage and BF16 compute at `B=1, L=2048`. Each row passed the operation's independent correctness gate.

| Production cache operation | Before hybrid |   Hybrid | Compiler temporary before | Compiler temporary hybrid |
| -------------------------- | ------------: | -------: | ------------------------: | ------------------------: |
| Pristine prefill           |      87.73 ms | 14.47 ms |                  293.8 MB |                  326.4 MB |
| Continuation, 37 tokens    |      12.79 ms | 12.25 ms |                   85.4 MB |                    6.3 MB |
| Decode, 1 token            |       3.11 ms |  3.20 ms |                   0.04 MB |                   0.04 MB |

The mapped/scanned power-block schedule was also compiled well beyond the 4,096-token block boundary. These are isolated single-layer component measurements with full parameter and input differentiation, not full-model feasibility claims.

| Isolated shape               |    Pristine prefill |        Continuation | Full forward/backward |
| ---------------------------- | ------------------: | ------------------: | --------------------: |
| `B=1, D=1024, N=16, L=32768` |   13.82 ms / 806 MB |   15.87 ms / 940 MB |   19.81 ms / 1,342 MB |
| `B=1, D=4096, N=16, L=16384` | 21.39 ms / 2,418 MB | 25.39 ms / 2,417 MB |   30.35 ms / 3,359 MB |

Each cell reports synchronized runtime and compiler temporary memory; every result and gradient was finite. The temporary sizes are consistent with one rematerialized power block plus FFT/output workspace rather than a retained `D * N * L` complex tensor.

Packed execution has different reset semantics and bypasses this hybrid entirely. It must be benchmarked at the intended batch, sequence, model, and CEMA dimensions because its associative-path memory scales with their product. The supported downstream envelope and required measurements are part of the [packed CEMA gate](dev.md#packed-cema-gate); a reduced or forward-only microbenchmark is not sufficient evidence for a production training claim.
