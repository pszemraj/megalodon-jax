# Documentation

Start with the [main README](../README.md) for the project overview, installation commands, and a first model call.

## Setup and use

- [Installation and setup](installation.md): CPU and CUDA installation, optional features, unreleased builds, and development setup.
- [Long-context streaming](long-context-streaming.md): generation, cache behavior, padding constraints, sliding attention, and packed training.
- [Dtypes and numerical stability](dtypes-and-stability.md): FP32/BF16 policies, training, memory-bounded loss, and data parallelism.
- [JAX and PyTorch interoperability](jax-torch.md): checkpoints, original-upstream conversion, resume state, and parity gates.

## Model behavior and compatibility

- [Paper and source differences](paper-deviations.md): named model presets, released-source compatibility choices, and intentional JAX extensions.
- [ComplexEMA implementation](ema-implementation.md): execution paths, packed resets, and numerical behavior.
- [Upstream parity and production contracts](upstream-parity-contract.md): normative compatibility, precision, distributed-execution, and benchmark boundaries.

## Contributing

- [Development](dev.md): code quality, test gates, release process, correctness verification, and performance benchmarks.
