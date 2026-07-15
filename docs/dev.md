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

The default training matrix enables gradient checkpointing and covers batches 1/2/4, lengths 2,048/4,096, forward and forward-backward execution, and plain, all-valid-mask, and packed inputs.

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

Use `benchmarks/benchmark_cema_reset.py` for the isolated FFT, associative packed, and sequential CEMA paths. Benchmark representative sequence shapes and masks: an all-valid `attention_mask` intentionally exercises the general masked path, while `attention_mask=None` selects the unmasked path described in [Long-context streaming](long-context-streaming.md#padding-and-generation).

Benchmark JSON records the installed JAX CUDA plugin/PJRT and CUDA library wheels, NVIDIA driver and `nvidia-smi` output, compiler-affecting paths and environment variables, and `jax.print_environment_info()`. Add `--profile-dir PATH` to capture one extra synchronized iteration per case as an XProf trace without contaminating timing samples; select a narrow operation and shape matrix when profiling.
