# JAX and original PyTorch interoperability

Conversion targets the exact released Megalodon PyTorch keyspace, not a Hugging Face port or an installed package with a similar name. A local upstream checkout is used to audit that mapping but is not required to load a checkpoint. Torch is optional at runtime; install `megalodon-jax[convert]` when conversion is needed.

## Native JAX checkpoints

Use the versioned SafeTensors format for JAX training and inference:

```python
import jax

from megalodon_jax import load_checkpoint, save_checkpoint

save_checkpoint(model, "model.safetensors")
restored = load_checkpoint("model.safetensors", key=jax.random.PRNGKey(1))
```

Native format v2 stores the complete configuration, a configuration fingerprint, the tensor manifest, parameter names, and dtype policy. Loading is strict. Metadata-free files, legacy layouts, missing tensors, extra tensors, changed shapes, and changed dtypes are rejected.

Partial restore is explicit and cannot silently fall back:

```python
from megalodon_jax import load_partial_checkpoint

model, report = load_partial_checkpoint(
    "model.safetensors",
    config,
    include={"model.embed.weight"},
    key=jax.random.PRNGKey(2),
)
```

The report lists restored and freshly initialized leaves plus source/target configuration fingerprints and an `exact_config_match` flag. A selection containing an unknown or unavailable name raises; intentional cross-config partial restores remain possible but cannot hide their provenance.

## Original upstream to JAX

For a world-size-one released checkpoint:

```python
from megalodon_jax.convert import load_upstream_checkpoint

model = load_upstream_checkpoint(
    "consolidated.pth",
    config,
    key=jax.random.PRNGKey(0),
)
```

Directories produced by the original consolidation workflow are also accepted when they contain `consolidate_config.json` and the declared `consolidated.pth` or `consolidated.NN.pth` shards. Raw FSDP directories and SafeTensors from unrelated schemas are refused.

The loader validates the complete exact key set, source tensor shapes, model-parallel merge axes, CEMA real/complex representation, RoPE frequencies, plus-one normalization storage, projection bias topology, tied-output equality, and configuration compatibility. When `share_emb=True`, the separately serialized embedding and output tensors must be bit-identical because they represent one logical upstream parameter; approximate equality is deliberately rejected.

For an already loaded original state dictionary:

```python
from megalodon_jax.convert import load_upstream_state_dict

model = load_upstream_state_dict(model, state_dict)
```

## JAX to original upstream

Export the exact released world-size-one keyspace and save it with Torch:

```python
import torch
from megalodon_jax.convert import export_upstream_state_dict

state_dict = export_upstream_state_dict(model)
torch.save(state_dict, "consolidated.pth")
```

Weights use the original `(out_features, in_features)` layout without speculative transposes. CEMA gamma is reassembled as real/imag pairs, omega regains its singleton axis, and tied output weights are emitted as an equal clone so serialization does not depend on shared storage.

## Compatibility boundary

The historical `convert_jax_to_torch`, `load_weights_from_torch`, `load_from_pretrained`, and `save_safetensors` functions guessed between incompatible downstream and native schemas. They now raise with the explicit replacement API.

Existing pre-v2 JAX checkpoints require an owner-controlled offline migration because their metadata does not identify RoPE coordinates, normalization storage, bias topology, tying intent, or preset identity. Runtime loading deliberately refuses to infer those choices. A safe migration must supply the original config and provenance, transform affected tensors, then write a v2 checkpoint and verify logits before deleting the source file.

## Inference cache persistence

Continuation state has a separate versioned format:

```python
from megalodon_jax import load_inference_cache, save_inference_cache

save_inference_cache(cache, "cache.safetensors", config)
cache = load_inference_cache("cache.safetensors", config)
```

Cache files are bound to the full configuration fingerprint and validate the exact presence/tensor schema, fixed KV capacity, batch consistency, layer count, every state dtype/shape, and the mirrored attention position/count invariant. They are continuation artifacts, not portable model checkpoints.

## Parity gates

The load-bearing core-math checks use independent NumPy references for TimestepNorm, RoPE, initialization statistics, and dense attention. They do not import `megalodon_jax` production implementations as their expected-value path.

`tests/test_upstream_parity.py` uses a small differentiable Torch transcription of the released equations. It compares full logits, every trainable upstream-schema gradient, and short optimizer trajectories, but it was written alongside this JAX port and is not an independent oracle. Agreement demonstrates consistency between the two implementations; disagreement identifies a divergence that must be investigated rather than proving which side is correct.

`tools/verify_modeling_correctness.py` additionally reads a local upstream source path when it is available; without one it runs repository-only checks and reports source-anchoring checks as skipped. The strict conversion regression uses a hand-authored released-source manifest for keys, shapes, dtypes, literal RoPE frequencies, and model-parallel partition axes. No check imports another Megalodon package or builds the fused extension, and no released trained checkpoint is available for a ground-truth logits comparison. Fused CUDA is checked at source level only.
