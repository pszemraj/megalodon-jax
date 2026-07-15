# megalodon-jax

A JAX/Equinox reimplementation of [Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length](https://arxiv.org/abs/2404.08801), aligned with the original released PyTorch/CUDA implementation.

## Features

- Pure JAX/Equinox implementation in [src/megalodon_jax](src/megalodon_jax)
- Core architecture: ComplexEMA, chunked rotary attention, fixed-capacity streaming caches, RMSNorm, and TimestepNorm
- Packed-sequence training with full document isolation: `segment_ids` masks attention, resets EMA/norm state at boundaries, and excludes cross-document label pairs from the loss
- JAX pytree caches for JIT-compatible streaming inference
- Strict native SafeTensors checkpoints plus exact original-upstream PyTorch checkpoint conversion
- Source-transcribed PyTorch/JAX forward, all-parameter gradient, and optimizer consistency gates without building the fused CUDA extension
- Official released tokenizer bundle in [assets/tokenizer](assets/tokenizer), versioned in-tree for paper and released-repository reproducibility

## Installation

Install with pip+git for the latest version:

```sh
pip install "git+https://github.com/pszemraj/megalodon-jax.git"
```

On Linux with an NVIDIA CUDA 13-capable driver, install JAX's official bundled CUDA stack:

```sh
pip install "megalodon-jax[cuda13] @ git+https://github.com/pszemraj/megalodon-jax.git"
```

The `cuda13` extra installs the matching JAX plugin, PJRT, CUDA, and cuDNN wheels. Keep `LD_LIBRARY_PATH` unset when using these bundled libraries; pointing it at another CUDA toolkit can make XLA load an incompatible library set.

### Development install

For development on an NVIDIA system, clone the repository and install the CUDA 13 and developer extras together:

```bash
git clone https://github.com/pszemraj/megalodon-jax.git
cd megalodon-jax
pip install -e ".[cuda13,dev]"
```

Requires Python 3.11+, JAX `>=0.10.2,<0.11`, and Equinox `>=0.13.8,<0.14`. CPU-only development can use `pip install -e ".[dev]"`. PyTorch is optional; install `.[convert]` for original-upstream checkpoint conversion or `.[dev]` for conversion, parity tests, and developer tooling.

## Quick start

```python
import jax
import jax.numpy as jnp
from megalodon_jax import MegalodonConfig, MegalodonForCausalLM

key = jax.random.PRNGKey(0)
cfg = MegalodonConfig(
    vocab_size=32_000,
    model_dim=512,
    num_layers=8,
    num_heads=1,
    chunk_size=256,
    cema_ndim=16,
)
model = MegalodonForCausalLM(cfg, key=key)

# Forward pass
input_ids = jax.random.randint(key, (1, 128), 0, cfg.vocab_size)
logits, cache = model(input_ids, return_cache=True)
print(logits.shape)  # (1, 128, 32000)
```

### Exact named presets

Preset names are explicit because the paper 7B configuration and the two released 7B configurations are not interchangeable. With `vocab_size=32_000`, the exact trainable counts are:

| Factory | SwiGLU | Chunk | RoPE base | Tied | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `from_upstream_mega200m` | no | 2,048 | 10,000 | no | 220,627,968 |
| `from_upstream_mega1_3b` | no | 2,048 | 10,000 | no | 1,342,832,640 |
| `from_upstream_mega1_3b_pg19` | yes | 2,048 | 10,000 | yes | 1,327,628,288 |
| `from_upstream_mega7_1b` | no | 2,048 | 10,000 | no | 7,117,381,632 |
| `from_upstream_mega7_3b` | yes | 2,048 | 10,000 | no | 7,385,817,088 |
| `from_paper_7b` | yes | 4,096 | 100,000 | no | 7,385,817,088 |

`from_7b()` intentionally raises because its historical definition mixed incompatible upstream presets. Use `config.parameter_count_breakdown()` for exact counts at another vocabulary size or output width. Output tying is controlled only by the explicit `share_emb` field; it is never inferred from matching shapes.

### Streaming inference

```python
# Continue from cache
next_token = jax.random.randint(key, (1, 1), 0, cfg.vocab_size)
next_logits, new_cache = model(next_token, cache=cache, return_cache=True)
```

```python
from megalodon_jax.inference import generate

# Autoregressive generation (sampling returns the next PRNG key)
prompt_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
tokens, cache, key = generate(
    model,
    prompt_ids,
    max_new_tokens=16,
    key=key,
    temperature=1.0,
    return_cache=True,
)
```

Cached-generation constraints and cache semantics are described in [Long-context streaming](docs/long-context-streaming.md#padding-and-generation).

### Training with loss

```python
import equinox as eqx

labels = input_ids  # For causal LM, labels = input_ids
loss = model.compute_loss(input_ids, labels)

# Gradient computation
@eqx.filter_grad
def loss_fn(model, input_ids, labels):
    return model.compute_loss(input_ids, labels)

grads = loss_fn(model, input_ids, labels)
```

### Packed-sequence training

Multiple documents can share one row by passing segment and position metadata:

```python
# doc A (3 tokens) + doc B (4 tokens) + padding (1), packed in one row
input_ids = jnp.array([[5, 6, 7, 8, 9, 10, 11, 0]])
attention_mask = jnp.array([[True] * 7 + [False]])
segment_ids = jnp.array([[1, 1, 1, 2, 2, 2, 2, 0]])  # 0 = padding
# Optional: RoPE positions restarting per document. When omitted, they are
# derived from segment_ids automatically.
position_ids = jnp.array([[0, 1, 2, 0, 1, 2, 3, 0]])

loss = model.compute_loss(
    input_ids,
    input_ids,
    attention_mask=attention_mask,
    segment_ids=segment_ids,
    position_ids=position_ids,
)
```

See [Packed-sequence training](docs/long-context-streaming.md#packed-sequence-training) for isolation, boundary, padding, cache, and loss semantics.

## Documentation

The [documentation index](docs/README.md) covers streaming and packed execution, precision, ComplexEMA, checkpoint interoperability, paper/source differences, tests, and benchmarks.

### Source layout

```
src/megalodon_jax/
├── config.py          # MegalodonConfig (frozen dataclass)
├── cache.py           # Cache schema and invariant checks
├── checkpoint.py      # Strict native model/cache persistence
├── convert.py         # Exact original-upstream checkpoint conversion
├── inference.py       # Cache indexing, sampling, and generation
├── model.py           # MegalodonBlock, MegalodonModel, MegalodonForCausalLM
├── ops.py             # Dtype-aware linear algebra and dropout
├── precision.py       # Sensitive-parameter dtype audit and repair
├── types.py           # Cache/state pytrees
├── utils.py           # Weight initialization
└── layers/
    ├── attention.py      # ChunkedAttention, MegalodonAttention, NormalizedFFN
    ├── complex_ema.py    # ComplexEMA
    ├── norms.py          # RMSNorm and plus-one LayerNorm
    ├── rotary.py         # RotaryEmbedding
    ├── segments.py       # Packed boundary and position helpers
    └── timestep_norm.py  # TimestepNorm (streaming GroupNorm)
```

## Related

- [Original Megalodon](https://github.com/XuezheMax/megalodon) - Released PyTorch/CUDA source

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
