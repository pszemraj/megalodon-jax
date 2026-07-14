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

Tests join this gate through an explicit `@pytest.mark.fast` declaration at the test site. CPU runs do not force garbage collection around every test; set `MEGALODON_TEST_AGGRESSIVE_CLEANUP=1` only when running in a constrained environment that benefits from it. Accelerator runs retain per-test cleanup to reduce cross-test device-memory pressure.

Run the complete suite on the selected JAX backend before merging modeling changes:

```bash
conda run --name mega-jax pytest
```

The `torch_ref` marker selects the [source-transcribed PyTorch consistency checks](jax-torch.md#parity-gates):

```bash
conda run --name mega-jax pytest -m torch_ref
```

CI runs the fast gate on Python 3.11 and 3.13 with the minimum supported JAX and Equinox versions. The full local suite remains the merge gate for GPU, slow compilation, and integration coverage.

## Modeling verifier

`tools/verify_modeling_correctness.py` compares the implementation with independent mathematical references and, when available, the exact released source. Its PyTorch transcription is a cross-implementation consistency check, not a ground-truth runtime oracle. Repository-only checks run without the untracked source checkout and report source-anchoring checks as skipped. A release-quality run supplies that source path and includes the slow forward, gradient, optimizer, conversion, save/reload, and cache-partition checks:

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

To compare revisions, provide each clean checkout explicitly:

```bash
conda run --name mega-jax python benchmarks/benchmark_model_paths.py \
  --repo current=. \
  --repo baseline=/path/to/baseline-checkout \
  --suite training \
  --training-lengths 512,1024 \
  --training-batches 1 \
  --output local-scratch/model-paths-training.json
```

Use `benchmarks/benchmark_cema_reset.py` for the isolated FFT, associative packed, and sequential CEMA paths. Benchmark representative sequence shapes and masks: an all-valid `attention_mask` intentionally exercises the general masked path, while `attention_mask=None` selects the unmasked path described in [Long-context streaming](long-context-streaming.md#padding-and-generation).

Benchmark JSON records the installed JAX CUDA plugin/PJRT and CUDA library wheels, NVIDIA driver and `nvidia-smi` output, compiler-affecting paths and environment variables, and `jax.print_environment_info()`. Add `--profile-dir PATH` to capture one extra synchronized iteration per case as an XProf trace without contaminating timing samples; select a narrow operation and shape matrix when profiling.
