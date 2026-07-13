# Development

Install the project as described in the README's [development install](../README.md#development-install). The commands below use the repository's `mega-jax` conda environment; use the equivalent environment wrapper in another checkout.

## Code quality and tests

Run formatting and lint checks before each commit:

```bash
conda run --name mega-jax ruff check --fix .
conda run --name mega-jax ruff format .
```

The routine CPU gate covers the highest-value model, cache, conversion, and source-derived parity checks:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= conda run --name mega-jax pytest -m fast
```

Run the complete suite on the selected JAX backend before merging modeling changes:

```bash
conda run --name mega-jax pytest
```

The `torch_ref` marker selects the [source-derived PyTorch parity checks](jax-torch.md#parity-gates):

```bash
conda run --name mega-jax pytest -m torch_ref
```

CI runs the fast gate on Python 3.11 and 3.13 with the minimum supported JAX and Equinox versions. The full local suite remains the merge gate for GPU, slow compilation, and integration coverage.

## Modeling verifier

`tools/verify_modeling_correctness.py` compares the implementation with independent mathematical oracles and the exact released source. It requires local paths to the paper and original source tree. A release-quality run includes the slow forward, gradient, optimizer, conversion, save/reload, and cache-partition checks:

```bash
conda run --name mega-jax python tools/verify_modeling_correctness.py \
  --jax-repo . \
  --upstream-repo local-scratch/megalodon-upstream-cuda_torch \
  --paper local-scratch/megalodon_paper.md \
  --backend gpu \
  --include-slow \
  --json local-scratch/modeling-correctness-verification.json
```

The verifier writes explicit skipped results and exits nonzero when an executed invariant fails. Its fused CUDA runtime comparison is always recorded as scope-excluded; the released CUDA code is a source reference and is not built by this workflow.

## Performance benchmarks

`benchmarks/benchmark_model_paths.py` measures synchronized production inference and training paths in isolated worker processes. Compilation is reported separately from runtime, and the output records repository revisions, configuration, environment, correctness checks, median, p90, and memory data.

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
