# megalodon-jax

A JAX/Equinox reimplementation of [Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length](https://arxiv.org/abs/2404.08801), ported from [megalodon-hf](https://github.com/pszemraj/megalodon-hf).

## Features

- Pure JAX/Equinox implementation in [src/megalodon_jax](src/megalodon_jax)
- Core architecture: ComplexEMA (FFT + sequential paths), chunked rotary attention, streaming cache, RMS/Timestep norms
- JAX pytree caches for JIT-compatible streaming inference
- Weight conversion utilities for PyTorch ↔ JAX interop
- 200+ tests covering parity with the PyTorch reference

## Installation

Install with pip+git for the latest version:

```sh
pip install "git+https://github.com/pszemraj/megalodon-jax.git"
```

### development install

For development, clone the repository and install with the `[dev]` extras:

```bash
git clone https://github.com/pszemraj/megalodon-jax.git
cd megalodon-jax
pip install -e ".[dev]"
```

Requires Python 3.11+ with JAX 0.7.0+ (tested with 0.8.x), Equinox 0.12.0+.

## Quick Start

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

### Streaming Inference

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
)
```

Note: when `attention_mask` contains padding and `max_new_tokens > 1`, generation groups
by prompt length (left padding only) and does not support `return_cache=True`.

### Training with Loss

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

## Architecture

### Key Design Decisions

1. **JAX Pytree Caches**: All cache/state objects are registered as JAX pytrees. Position counters are JAX scalar arrays (not Python ints) to prevent recompilation.

2. **Two-Path ComplexEMA**:
   - FFT path: O(L log L), used during training when no state needed
   - Sequential scan: O(L), maintains complex hidden state for streaming

3. **Normalized Attention**: Q/K use per-head RMSNorm before affine transform. Attention uses `scale=1.0` (no `/sqrt(d_head)` scaling).

4. **Two-Hop Residual**: FFN adds residual from block input, not post-attention activations.

5. **Mask-Aware Loss**: Padding tokens are excluded from loss computation (matches PyTorch/HF).

### Source Layout

```
src/megalodon_jax/
├── config.py          # MegalodonConfig (frozen dataclass)
├── types.py           # Cache/state pytrees
├── utils.py           # Weight initialization
├── model.py           # MegalodonBlock, MegalodonModel, MegalodonForCausalLM
├── convert.py         # Weight conversion (PyTorch ↔ JAX)
└── layers/
    ├── norms.py       # RMSNorm
    ├── rotary.py      # RotaryEmbedding
    ├── timestep_norm.py  # TimestepNorm (streaming GroupNorm)
    ├── complex_ema.py    # ComplexEMA
    └── attention.py      # ChunkedAttention, MegalodonAttention, NormalizedFFN
```

## Precision

- Float32 for numerical stability
- BFloat16 compatible via JAX autocast
- **Never use float16** (EMA/FFT overflow)

## Performance

- **Sequence length padding**: JAX recompiles for each unique input shape. Pad sequences to consistent lengths (e.g., powers of 2) during training to avoid excessive recompilation.

## Current Status

- Core components + streaming cache utilities
- Sampling + `generate()` loop for text generation
- PyTorch ↔ JAX conversion (SafeTensors via PyTorch state dicts)

## Limitations

- Pure JAX implementation (no fused CUDA kernels)
- Sequential CEMA path is slower than FFT; training uses FFT automatically (JAX is ~5x faster than PyTorch for both paths)
- No 4D chunk parallelism (out of scope for single-device)
- CEMA zeros masked positions before recurrence to avoid padding contamination (matches PyTorch)

## Testing

```bash
pytest                          # All CPU tests
pytest -m torch_ref             # PyTorch reference parity tests
pytest tests/test_model.py -v   # Single file
```

## Related

- [megalodon-hf](https://github.com/pszemraj/megalodon-hf) - PyTorch/Transformers implementation
- [Original Megalodon](https://github.com/XuezheMax/megalodon) - Reference CUDA implementation

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
