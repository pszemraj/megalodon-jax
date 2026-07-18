# megalodon-jax

A JAX/Equinox implementation of [Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length](https://arxiv.org/abs/2404.08801), aligned with the original released PyTorch/CUDA implementation.

## Highlights

- Long-context language modeling with chunked attention and ComplexEMA
- JIT-compatible streaming inference with fixed-capacity caches
- Packed-sequence training with document-isolated attention, recurrent state, normalization, and loss
- SafeTensors checkpoints and original-upstream PyTorch checkpoint conversion
- Source-level parity checks and the [official released tokenizer bundle](https://github.com/pszemraj/megalodon-jax/tree/main/assets/tokenizer)

## Installation

For CPU:

```bash
pip install megalodon-jax
```

For NVIDIA GPUs using the pip-managed CUDA 13 runtime:

```bash
pip install "megalodon-jax[cuda13]"
```

Python 3.11 or newer is required. See [Installation and setup](https://github.com/pszemraj/megalodon-jax/blob/main/docs/installation.md) for CUDA requirements, optional features, unreleased `main` installs, development setup, and troubleshooting.

## Quick start

```python
import jax
from megalodon_jax import MegalodonConfig, MegalodonForCausalLM

model_key, input_key = jax.random.split(jax.random.PRNGKey(0))
config = MegalodonConfig(
    vocab_size=32_000,
    model_dim=512,
    num_layers=8,
    num_heads=1,
    chunk_size=256,
    cema_ndim=16,
)
model = MegalodonForCausalLM(config, key=model_key)

input_ids = jax.random.randint(input_key, (1, 128), 0, config.vocab_size)
logits, _ = model(input_ids)
print(logits.shape)  # (1, 128, 32000)
```

## Where to go next

- [Installation and setup](https://github.com/pszemraj/megalodon-jax/blob/main/docs/installation.md): CUDA compatibility, optional extras, source installs, and development setup
- [Long-context streaming](https://github.com/pszemraj/megalodon-jax/blob/main/docs/long-context-streaming.md): generation, cache behavior, sliding attention, padding, and packed training
- [Dtypes and numerical stability](https://github.com/pszemraj/megalodon-jax/blob/main/docs/dtypes-and-stability.md): precision policies, training, memory-bounded loss, and data parallelism
- [JAX and PyTorch interoperability](https://github.com/pszemraj/megalodon-jax/blob/main/docs/jax-torch.md): checkpoints, conversion, resume state, and parity gates
- [Paper and source differences](https://github.com/pszemraj/megalodon-jax/blob/main/docs/paper-deviations.md): named model presets and deliberate compatibility choices
- [Development](https://github.com/pszemraj/megalodon-jax/blob/main/docs/dev.md): tests, release process, correctness verification, and benchmarks

The [documentation index](https://github.com/pszemraj/megalodon-jax/blob/main/docs/README.md) covers the remaining implementation and compatibility references.

## Related

- [Original Megalodon](https://github.com/XuezheMax/megalodon) — released PyTorch/CUDA source

## Citation

```bibtex
@misc{ma2024megalodon,
      title={Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length},
      author={Xuezhe Ma and Xiaomeng Yang and Wenhan Xiong and Beidi Chen and Lili Yu and Hao Zhang and Jonathan May and Luke Zettlemoyer and Omer Levy and Chunting Zhou},
      year={2024},
      eprint={2404.08801},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
