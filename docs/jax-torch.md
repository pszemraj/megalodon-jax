# JAX and PyTorch Interop

This repo provides the JAX implementation. The PyTorch reference lives in [megalodon-hf](https://github.com/pszemraj/megalodon-hf) and is the target for deployment and parity checks.

## Why the JAX version exists

- Fast training and experimentation with JIT compilation.
- Clean parity path to PyTorch for production inference and tooling.
- Explicit, readable kernels that are easy to audit for correctness.

## When to use which

Use JAX for:

- Training and large-scale ablations.
- Research workflows where compile-time shape control matters.
- Exporting checkpoints for downstream PyTorch use.

Use PyTorch (megalodon-hf) for:

- HuggingFace integration and generation APIs.
- Deployment and production inference.
- Compatibility with existing PyTorch tooling and ecosystems.

## Conversion workflows

Conversion utilities live in `megalodon_jax.convert` and require torch (install with `megalodon-jax[convert]`).

### JAX -> PyTorch

Use `convert_jax_to_torch` to get a PyTorch-style state dict, or `save_safetensors` to write a `.safetensors` file.

```python
from megalodon_jax.convert import convert_jax_to_torch, save_safetensors

state_dict = convert_jax_to_torch(model)
save_safetensors(model, "model.safetensors")
```

Notes:

- `inner.rope.inv_freq` is not in the PyTorch state dict. The JAX exporter skips it by default. Use `include_rope_inv_freq=True` only if you need it for tooling (it will be an unexpected key under strict loading).
- Use `dtype=` to export bf16 checkpoints (fp32 is the default and recommended).
- Precision-sensitive parameters (CEMA params, norms, per-head Q/K affine) are exported in fp32 for stability.

### PyTorch -> JAX

Use `load_from_pretrained` for a checkpoint file, or `load_weights_from_torch` if you already have a state dict in memory.

```python
from megalodon_jax.convert import load_from_pretrained

model = load_from_pretrained(
    "model.safetensors",
    config=config,
    key=key,
)
```

Make sure the JAX config matches the PyTorch config (dims, layer count, swiglu, norm_affine, output_size, etc).

Notes:

- Conversion paths require torch for `safetensors.torch` and `torch.load`.

## Functional differences (current)

- **Cache + padding**: JAX does not support padded inputs for cached generation; `generate()` rejects padded `attention_mask` when `max_new_tokens > 1`, `return_cache=True`, or a cache is provided. Use unpadded prompts or generate one token at a time.
- **Cache sizing**: JAX uses a fixed-size ring buffer for KV cache (required for JIT). PyTorch can grow dynamically.
- **Generation API**: JAX provides its own `generate` loop; it does not implement the full HuggingFace `GenerationMixin` surface.
- **Compilation shapes**: JAX recompiles on new shapes. Pad sequences to a consistent length for throughput.

## Parity and compatibility notes

- The torch fixes for masking, cache semantics, and per-batch positions do not add or rename parameters; weight mapping is unchanged.
- `max_cache_len` must be `>= chunk_size` when provided (validated in config).
- Loss masking uses `ignore_index=-100` and attention masks consistently, same as the PyTorch/HF convention.
- Parity tests that rely on the external `megalodon-hf` package are marked with `@pytest.mark.torch_ref` in `tests/` for easy selection.

## Quick parity checklist

- Configs match (dims, layers, swiglu, norm_affine, output_size).
- Export with `convert_jax_to_torch` and load in torch with `strict=True`.
- Compare logits on a small batch in fp32 with deterministic settings.
