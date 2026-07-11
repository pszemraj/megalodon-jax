# megalodon-jax

A JAX/Equinox reimplementation of [Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length](https://arxiv.org/abs/2404.08801), aligned with the original released PyTorch/CUDA implementation.

## Features

- Pure JAX/Equinox implementation in [src/megalodon_jax](src/megalodon_jax)
- Core architecture: ComplexEMA (FFT + sequential paths), chunked rotary attention, streaming cache, RMS/Timestep norms
- Packed-sequence training with full document isolation: `segment_ids` masks attention, resets EMA/norm state at boundaries, and excludes cross-document label pairs from the loss
- JAX pytree caches for JIT-compatible streaming inference
- Strict native SafeTensors checkpoints plus exact original-upstream PyTorch checkpoint conversion
- Independent source-derived PyTorch/JAX forward, all-parameter gradient, and optimizer parity gates without building the fused CUDA extension

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

Requires Python 3.11+, JAX 0.8.2 through 0.10.x, and Equinox 0.13.x. The correctness gate passes at both the installed baseline (JAX 0.8.2 / Equinox 0.13.2) and the current releases (JAX 0.10.2 / Equinox 0.13.8). PyTorch is optional; install `.[convert]` for original-upstream checkpoint conversion or `.[dev]` for conversion, parity tests, and developer tooling.

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

Note: when `attention_mask` contains padding, cached generation (`max_new_tokens > 1`, `return_cache=True`, or `cache` provided) is not supported.

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

### Packed-Sequence Training

Multiple documents can share one row with full isolation - attention, EMA state, and running norm statistics all reset at document boundaries, and the loss automatically skips cross-document label pairs:

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

Each packed document produces the same outputs (and gradients) as running it alone. Packed metadata is training-only: it is rejected whenever a cache is involved. Harnesses can detect support via `getattr(model, "supports_segment_reset", False)`. See [docs/dev.md](docs/dev.md#packed-sequence-state-isolation) for design notes and benchmarks.

## Architecture

### Key Design Decisions

1. **JAX Pytree Caches**: All cache/state objects are registered as JAX pytrees. Position counters are JAX scalar arrays (not Python ints) to prevent recompilation.

2. **Three-Path ComplexEMA**:
   - FFT path: O(L log L), used during training when no state needed
   - Sequential scan: O(L), maintains complex hidden state for streaming
   - Segmented associative scan: O(L log L) work, resets state at document boundaries for packed training

3. **Normalized Attention**: Q/K use per-head RMSNorm before affine transform. Attention uses `scale=1.0` (no `/sqrt(d_head)` scaling).

4. **Two-Hop Residual**: FFN adds residual from block input, not post-attention activations.

5. **Mask-Aware Loss**: Attention masks, ignored labels, padding segments, and cross-document shifted pairs are excluded explicitly.

### Source Layout

```
src/megalodon_jax/
├── config.py          # MegalodonConfig (frozen dataclass)
├── types.py           # Cache/state pytrees
├── utils.py           # Weight initialization
├── checkpoint.py      # Strict native model/cache persistence
├── model.py           # MegalodonBlock, MegalodonModel, MegalodonForCausalLM
├── convert.py         # Exact original-upstream checkpoint conversion
└── layers/
    ├── norms.py       # RMSNorm
    ├── rotary.py      # RotaryEmbedding
    ├── timestep_norm.py  # TimestepNorm (streaming GroupNorm)
    ├── complex_ema.py    # ComplexEMA
    └── attention.py      # ChunkedAttention, MegalodonAttention, NormalizedFFN
```

## Precision

- Parameter storage and accumulation are float32; compute is float32 by default.
- Set `compute_dtype=jnp.bfloat16` for BF16 compute while retaining FP32 parameters and accumulation.
- `attention_softmax_dtype` and `loss_softmax_dtype` control attention and language-model loss math independently; FP32 is recommended for both.
- Use `megalodon_jax.precision.audit_sensitive_param_dtypes` to verify fp32-sensitive params.
- Avoid blanket `jax.tree.map` casts to bf16; use the config dtypes or `ensure_sensitive_param_dtype`.
- See `docs/dtypes-and-stability.md` for downstream training guidance.
- Float16 is deliberately unsupported. Use FP32, or BF16 on hardware with native BF16 support; older GPUs without native BF16 must use FP32.

## Performance

- **Sequence length padding**: JAX recompiles for each unique input shape. Pad sequences to consistent lengths (e.g., powers of 2) during training to avoid excessive recompilation.

## Current Status

- Core components + streaming cache utilities
- Sampling + `generate()` loop for text generation
- Versioned native SafeTensors persistence and exact original-upstream `.pth` conversion

## Limitations

- Pure JAX implementation (no fused CUDA kernels)
- Sequential CEMA is slower than FFT; training uses FFT automatically when no state or segment reset is required
- No 4D chunk parallelism (out of scope for single-device)
- Cached decoding does not support padded batches
- Packed-sequence metadata (`segment_ids`/`position_ids`) is training-only; rejected on cached/streaming calls
- Token IDs are never treated as padding implicitly; callers must provide attention/loss masks and keep unknown-token semantics separate from padding metadata

## Testing

```bash
pytest                          # All CPU tests
pytest -m "not torch_ref"       # JAX-only tests
pytest -m torch_ref             # Independent source-derived Torch parity tests
pytest tests/test_model.py -v   # Single file
python tools/verify_modeling_correctness.py --jax-repo . --backend cpu --include-slow
```

## Related

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
