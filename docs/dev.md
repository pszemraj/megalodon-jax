# Development

Install the project as described in the README's [development install](../README.md#development-install). The commands below use the repository's `mega-jax` conda environment; use the equivalent environment wrapper in another checkout.

## Code quality and tests

Run formatting and lint checks before each commit:

```bash
conda run --name mega-jax ruff check --fix .
conda run --name mega-jax ruff format .
```

The routine CPU gate covers the highest-value model, cache, conversion, and source-transcription consistency checks:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= conda run --name mega-jax pytest -m fast
```

Tests join this gate through an explicit `pytest.mark.fast` marker on the test or module. CPU runs do not force garbage collection around every test; set `MEGALODON_TEST_AGGRESSIVE_CLEANUP=1` only when running in a constrained environment that benefits from it. Accelerator runs retain per-test cleanup to reduce cross-test device-memory pressure.

Run the complete suite on the selected JAX backend before merging modeling changes:

```bash
conda run --name mega-jax pytest
```

The `torch_ref` marker selects the [source-transcribed PyTorch consistency checks](jax-torch.md#parity-gates):

```bash
conda run --name mega-jax pytest -m torch_ref
```

Hosted CI is intentionally limited to Ruff lint/format checks and the CPU fast gate on Python 3.11 and 3.13 with the minimum supported JAX and Equinox versions. It does not run the unmarked full suite, the complete `torch_ref` parity suite, original-upstream conversion round-trips, the source-backed modeling verifier, or the GPU benchmark matrix. A few `torch_ref` tests also belong to the fast subset, but their presence does not make CI a full parity run. Run the complete local suite and the relevant manual gates before merging modeling changes; the sections below give the source-verifier and GPU benchmark commands.

## Modeling verifier

`tools/verify_modeling_correctness.py` runs the [independent references, source-transcription checks, and source anchors](jax-torch.md#parity-gates). A release-quality run supplies the released source checkout and enables the slower executable checks:

```bash
conda run --name mega-jax python tools/verify_modeling_correctness.py \
  --jax-repo . \
  --upstream-repo local-scratch/megalodon-upstream-cuda_torch \
  --backend gpu \
  --include-slow \
  --json local-scratch/modeling-correctness-verification.json
```

The verifier writes explicit skipped results and exits nonzero when an executed invariant fails. The released CUDA code is a source reference and is not built by this workflow.

## Performance benchmarks

`benchmarks/benchmark_model_paths.py` measures synchronized production inference and training paths in isolated worker processes. Compilation is reported separately from runtime, and the output records repository revisions, configuration, environment, correctness checks, median, p90, and memory data.

### Reference optimization environment

The 2026-07 optimization work is pinned to the [recorded reference environment](optimization-reference-environment.txt): RTX 5090, NVIDIA driver 595.71.05, Python 3.12.13, JAX/jaxlib 0.10.2 with the pip-managed CUDA 13 plugin stack, and Equinox 0.13.8. The snapshot records the complete resolved Python package set at base revision `4445209`. It is benchmark provenance rather than a replacement for the supported dependency ranges in `pyproject.toml`; do not mix a driver, CUDA, Python, JAX, Equinox, or attention-backend change into a comparison against this optimization series.

The benchmark evidence rules implement the normative [upstream parity and production contracts](upstream-parity-contract.md), including the requirement that packed CEMA be both isolated and useful at downstream training shapes.

The default training matrix enables gradient checkpointing and covers batches 1/2/4, lengths 2,048/4,096, forward and forward-backward execution, and plain, all-valid-mask, and packed inputs. Its canonical model is deliberately reduced and stable for cross-revision comparison: model dimension 1024, 12 layers, one attention head, vocabulary 16,000, and tied output. “Target” in this matrix refers only to requested batch and sequence dimensions; it is not the paper-7B topology and is not evidence of single-device 7B feasibility.

The cross-revision default uses tied embedding/output weights because historical `main` inferred tying for vocabulary-sized outputs. Measure the current untied production topology separately with `--config-json '{"share_emb": false}'`. Topology-sensitive cases that cannot match the requested topology are recorded as `completed_noncomparable` and excluded from cross-revision timing ratios.

Stochastic training cases derive a repeatable dropout key from the case seed. When dropout is active, packed cases validate the timed stochastic loss and gradients for finiteness, then use a separate untimed deterministic packed-versus-independent reference because changing document batch shapes necessarily changes dropout masks.

Run the benchmark with the default compiler configuration. Do not mix results produced with different `XLA_FLAGS` or CUDA library search paths.

```bash
conda run --name mega-jax python benchmarks/benchmark_model_paths.py \
  --repo current=. \
  --suite inference \
  --inference-lengths 64,512,2048,4096 \
  --output local-scratch/model-paths-inference.json
```

### Cross-revision comparisons

Run the benchmark driver from the candidate checkout and pass every revision as a named repository path. The workers import model code from those paths while sharing the candidate's benchmark driver and Python environment, so harness changes do not become part of the measured revision difference. Repository names such as `candidate` and `baseline` are labels, not special values.

#### Prepare clean revisions

Commit the candidate work first and verify that its worktree is clean. For the usual comparison against the latest remote `main`, fetch it and create a detached worktree without changing the candidate checkout:

```bash
conda run --name mega-jax git status --short
conda run --name mega-jax git fetch origin main
conda run --name mega-jax git worktree add --detach /tmp/megalodon-jax-baseline origin/main
```

Replace `origin/main` with any commit, tag, branch, or other revision when a different target is appropriate. An existing clean checkout at another path works equally well. Record the exact revisions before running:

```bash
conda run --name mega-jax git rev-parse HEAD
conda run --name mega-jax git -C /tmp/megalodon-jax-baseline rev-parse HEAD
```

When the candidate branch contains unrelated work, a branch-versus-main result cannot attribute its delta to one change. Create another detached worktree at the commit immediately before the change and repeat the same matrix as a separate pre/post comparison.

The benchmark rejects dirty or revision-less repositories by default. `--allow-dirty` and `--allow-unknown-revision` are useful for smoke tests, not reportable cross-revision results.

#### Choose a comparable matrix

Use the same model topology, dtypes, batches, sequence lengths, operations, and compiler environment for both repositories. The canonical cross-revision configuration is intentionally stable and tied; use `--config-json` only when the comparison calls for a different shared policy, then inspect each case's resolved configuration and unsupported-field list in the output. If the comparison intentionally changes a policy such as parameter storage dtype, report that distinction instead of presenting it as a like-for-like implementation speedup.

Start with the operations and shapes affected by the change, then expand to representative short and long contexts. A full inference run is also a correctness gate for cached execution:

```bash
JAX_ENABLE_COMPILATION_CACHE=false conda run --name mega-jax python benchmarks/benchmark_model_paths.py \
  --repo candidate=. \
  --repo baseline=/tmp/megalodon-jax-baseline \
  --suite inference \
  --inference-lengths 64,512,2048,4096 \
  --output local-scratch/candidate-vs-baseline-inference-a.json
```

Setting `JAX_ENABLE_COMPILATION_CACHE=false` makes `compile_ms` a cold-compilation measurement rather than a persistent-cache lookup. Every case already runs in a fresh worker, and compilation is excluded from the synchronized runtime samples.

For a focused training comparison, select the execution modes explicitly so the report says exactly what was measured:

```bash
JAX_ENABLE_COMPILATION_CACHE=false conda run --name mega-jax python benchmarks/benchmark_model_paths.py \
  --repo candidate=. \
  --repo baseline=/tmp/megalodon-jax-baseline \
  --suite training \
  --training-operations forward,forward_backward \
  --training-modes plain \
  --training-lengths 512,1024 \
  --training-batches 1 \
  --output local-scratch/candidate-vs-baseline-training-a.json
```

#### Gate timings on correctness

Inspect correctness before comparing latency. Only cases with `status: "passed"` and `comparability.eligible_for_cross_revision_ratio: true` are valid cross-revision timing pairs. A `failed` cache path, non-finite loss or gradient, or `completed_noncomparable` topology must be reported and excluded; fast incorrect execution is not a performance result. The supervisor still writes its JSON report when a case fails, but exits nonzero unless `--allow-failures` is supplied.

If a historical target fails only some correctness gates, narrow subsequent timing runs to the valid operation set rather than publishing ratios for the failures. For example, compare only the vectorized noncached path with `--inference-operations noncached` when cached execution is not correct in the baseline.

#### Reverse the run order

GPU temperature, clocks, allocator state, and unrelated host activity can bias a single sequential run. Repeat the valid matrix with the repository arguments reversed and write a second output file:

```bash
JAX_ENABLE_COMPILATION_CACHE=false conda run --name mega-jax python benchmarks/benchmark_model_paths.py \
  --repo baseline=/tmp/megalodon-jax-baseline \
  --repo candidate=. \
  --suite inference \
  --inference-lengths 64,512,2048,4096 \
  --output local-scratch/candidate-vs-baseline-inference-b.json
```

Apply the same argument reversal to training comparisons. Use the same matrix in both orders. If the initial correctness run reveals failed cases, preserve that report as evidence and rerun only the valid operation set in both orders; for example, add `--inference-operations noncached` to both timing commands when cached execution is not correct in the baseline. Prefer an otherwise idle accelerator; the recorded `nvidia-smi` output makes resident workloads visible but does not remove their interference.

#### Aggregate and report

For each comparable repository, operation, batch, and length, pool `metrics.timing.samples_ms` from both run orders and take the median and p90 of the pooled samples. Take the median of the independent `metrics.compile_ms` values for cold compilation. Report compiler peak and temporary memory separately from runtime-device memory, because they answer different questions.

Define runtime speedup as `baseline_median / candidate_median`, so a value greater than one favors the candidate. Aggregate multiple shapes with a geometric mean of their speedup ratios rather than an arithmetic mean of unrelated latencies. Treat changes near run-to-run variation as ties and repeat surprising or marginal results.

Before drawing a conclusion, verify the following output fields:

- `repositories` records the intended clean revisions and source-tree hashes.
- `config.resolved`, `model.parameter_count`, and `model.bytes_by_dtype` match the intended comparison.
- `comparability` permits a cross-revision ratio.
- `correctness` passes for every timed pair.
- `environment` agrees on JAX, CUDA libraries, devices, `XLA_FLAGS`, and compilation-cache policy.
- `metrics.timing` contains the synchronized runtime samples, while `metrics.compile_ms` and `metrics.compiler.memory_analysis` contain compilation and compiler-memory evidence.

Keep both raw JSON files with the reported table. State the revisions, hardware, model shape, dtype policy, batch and sequence matrix, warmups, timed iterations, order balancing, correctness exclusions, speedup formula, compilation-cache setting, and any visible competing workload. Distinguish the total branch-versus-target result from an isolated commit-range comparison when the branch contains unrelated work.

Remove the temporary worktree after preserving the reports:

```bash
conda run --name mega-jax git worktree remove /tmp/megalodon-jax-baseline
```

### BF16 contraction result-buffer gate

BF16 contractions use call-site-specific result dtypes. Attention QK scores, biasful projections, and the language-model head retain FP32 results for their FP32 consumers or bias epilogues. Attention probability-times-value and biasless projections whose public output is BF16 use `BF16_BF16_F32` with no FP32 preferred result, so StableHLO must show BF16 operands, FP32 accumulation, and a BF16 result. The FP32 reference path must continue to show genuine FP32 operands/results with `HIGHEST` precision and no BF16 algorithm metadata.

The retained policy was isolated against its immediate predecessor on 2026-07-16 with JAX 0.10.2 and an RTX 5090. A contraction probe used BF16 projection operands shaped `(4, 512, 1024) @ (4096, 1024).T` and a PV contraction shaped `(4, 8, 256, 512) @ (4, 512, 8, 128)`. Direct BF16 projection and PV outputs were bit-identical to FP32-result-then-downcast outputs. Projection forward/backward improved from 0.521 ms to 0.350 ms, process peak allocation fell from 260.0 MB to 180.4 MB, and the largest allocation fell from 134.2 MB to 58.7 MB. The compiler's abstract temporary-size report did not expose that allocator difference, so both measurements are retained rather than treating either one as the complete memory picture.

The end-to-end comparison used clean worktrees, the canonical 171,475,968-parameter model with FP32 ordinary storage, BF16 compute, FP32 accumulation/softmax, gradient checkpointing, `B=1`, `L=512`, two warmups, ten synchronized iterations per run, and both repository orders. Each row pools twenty samples. A resident 2.76 GB background process was idle at 0% GPU utilization, so these are order-balanced latency and per-process allocation results, not clean-device capacity limits.

| Forward/backward mode | FP32-result baseline | Direct-BF16 result | Runtime reduction | Baseline compiler peak | Candidate compiler peak | Baseline runtime peak | Candidate runtime peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain | 16.332 ms | 15.641 ms | 4.23% | 1,627.7 MB | 1,625.6 MB | 2,416.8 MB | 2,412.6 MB |
| Packed | 32.034 ms | 30.999 ms | 3.23% | 1,517.0 MB | 1,517.0 MB | 2,309.5 MB | 2,309.5 MB |

Both full-model modes passed finite-loss and all-gradient checks. Separate fixed-seed three-step GPU trajectories compared the exact pre-change and candidate revisions for plain and packed inputs: maximum loss drift was `1.07e-3`, maximum gradient drift was `5.86e-3`, maximum updated-parameter drift was `1.27e-5`, and every loss, gradient, parameter, and final logit remained within the predefined BF16 `rtol=atol=5e-2` envelope. Changes to this policy must repeat lowering, negative-control, trajectory, cache/generation, and end-to-end gates; a microbenchmark alone is insufficient.

### Packed CEMA gate

Use `benchmarks/benchmark_cema_reset.py` for the isolated FFT, associative packed, and sequential CEMA paths. Benchmark representative sequence shapes and masks: an all-valid `attention_mask` intentionally exercises the general masked path, while `attention_mask=None` selects the unmasked path described in [Long-context streaming](long-context-streaming.md#padding-and-generation).

For a production-path change, run full forward/backward at the downstream envelope of `D=1024, L=2048` with at least `B=8, N=16` and `B=4, N=32`, then run the canonical full-model packed matrix. Record synchronized latency, compile time, compiler argument/output/alias/temporary bytes, peak device memory when available, and correctness/finiteness. Measure gradients with respect to both the CEMA parameters and input; a forward-only or input-gradient-only result can hide training costs.

```bash
conda run --name mega-jax python benchmarks/benchmark_cema_reset.py --dim 1024 --ndim 16 --lengths 2048 --batches 8 --dtype bfloat16
conda run --name mega-jax python benchmarks/benchmark_cema_reset.py --dim 1024 --ndim 32 --lengths 2048 --batches 4 --dtype bfloat16
conda run --name mega-jax python benchmarks/benchmark_model_paths.py --repo current=. --suite training --training-operations forward_backward --training-modes packed --training-lengths 2048,4096 --training-batches 1,2,4 --output local-scratch/model-paths-packed.json
```

The associative scan intentionally remains the default because it is the useful-throughput packed path on the representative downstream workload. The sequential implementation is a fallback and a correctness cross-check. Its compact forward recurrence does not guarantee compact autodiff temporaries, so any memory claim must include the compiled forward/backward analysis. A proposed blockwise or hierarchical replacement must demonstrate an end-to-end improvement on this gate rather than relying only on the size of one conceptual tensor.

For paper-scale component evidence, additionally test `D=4096, N=16, L=4096` at the relevant local shard batch. Report the sharding policy and do not present a single-layer result as proof that the full paper-7B model, gradients, optimizer state, and compiler workspace fit on one device.

#### Current decision evidence

The retained default was revalidated on 2026-07-15 with JAX 0.10.2 on an RTX 5090. These numbers are a decision record, not portable performance thresholds; rerun the commands above after compiler, hardware, sharding, or model changes.

| Isolated BF16-input CEMA shape | Associative forward/backward | Sequential forward/backward | Associative compiler temporary | Sequential compiler temporary |
|---|---:|---:|---:|---:|
| `B=8, D=1024, N=16, L=2048` | 31.40 ms | 45.68 ms | 5,302 MB | 4,901 MB |
| `B=4, D=1024, N=32, L=2048` | 15.50 ms | 41.34 ms | 3,944 MB | 4,733 MB |
| `B=1, D=4096, N=16, L=4096` | 26.53 ms | 69.89 ms | 3,926 MB | 4,902 MB |
| `B=2, D=4096, N=16, L=4096` | 62.31 ms | 83.03 ms | 11,273 MB | 9,801 MB |

Every completed isolated run differentiated with respect to the CEMA parameters and input and produced finite values and gradients. At exact paper width, local `B=4` associative forward/backward exceeded the single process's available device memory while requesting a 21.00 GiB allocation; the isolated sequential fallback completed with finite gradients in 107.5 ms and a 19,600 MB compiler temporary. This is a measured local-shard boundary, not evidence that packed CEMA is unusable: the production default remains faster at the representative downstream shapes and exact-width local batches 1 and 2, while a paper-7B deployment must report its actual batch/model sharding because local `B=4` on one device is not the relevant full-model topology.

The canonical 171,475,968-parameter model also passed packed `B=4, L=2048` forward/backward with gradient checkpointing and BF16 compute under both ordinary-parameter storage policies. Both runs produced finite loss and gradients for every parameter and passed the packed-versus-independent loss reference.

| Ordinary parameter storage | Parameter bytes | Compiler peak | Measured peak device bytes | Synchronized runtime |
|---|---:|---:|---:|---:|
| FP32 | 685,903,872 | 3,384,683,324 | 3,457,565,184 | 541.49 ms |
| BF16 | 344,725,504 | 2,710,714,684 | 2,948,102,400 | 544.81 ms |

Compact BF16 storage cut model parameter bytes by 49.7% and reduced the compiler-reported and observed peaks without a meaningful runtime regression in this CEMA-dominated one-iteration gate. The BF16 row retains the configured FP32-sensitive CEMA, normalization, affine, and residual-scale leaves; it is not a blanket-cast model.

Benchmark JSON records the installed JAX CUDA plugin/PJRT and CUDA library wheels, NVIDIA driver and `nvidia-smi` output, compiler-affecting paths and environment variables, and `jax.print_environment_info()`. Add `--profile-dir PATH` to capture one extra synchronized iteration per case as an XProf trace without contaminating timing samples; select a narrow operation and shape matrix when profiling.
