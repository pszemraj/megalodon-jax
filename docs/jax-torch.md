# JAX and original PyTorch interoperability

Conversion targets the exact released Megalodon PyTorch keyspace, not a Hugging Face port or an installed package with a similar name. A local upstream checkout is used to audit that mapping but is not required to load a checkpoint. Torch is optional at runtime; install `megalodon-jax[convert]` when conversion is needed.

Conversion parity, intentional JAX extensions, and historical distributed-runtime details are separate concerns. Their normative boundary is defined in [Upstream parity and production contracts](upstream-parity-contract.md).

## Native JAX checkpoints

Use the versioned SafeTensors format for JAX training and inference:

```python
import jax

from megalodon_jax import load_checkpoint, save_checkpoint

save_checkpoint(model, "model.safetensors")
restored = load_checkpoint("model.safetensors", key=jax.random.PRNGKey(1))
```

Native format v2 stores the complete configuration, a configuration fingerprint, the tensor manifest, parameter names, and dtype policy. Loading is strict. Metadata-free files, legacy layouts, missing tensors, extra tensors, changed shapes, and changed dtypes are rejected.

Model, inference-cache, and generation-state writes use a same-directory temporary file followed by atomic replacement.

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

## Training-state resume

Native v2 checkpoints intentionally contain only model state. Persist an Optax state beside the model checkpoint with Equinox leaf serialization, and give the optimizer artifact a small sidecar containing the same configuration fingerprint:

```python
import json
from pathlib import Path

import equinox as eqx
import jax

from megalodon_jax import load_checkpoint, save_checkpoint
from megalodon_jax.checkpoint import config_fingerprint

step_dir = Path("checkpoints/step-000100")
step_dir.mkdir(parents=True, exist_ok=False)
save_checkpoint(model, step_dir / "model.safetensors")
eqx.tree_serialise_leaves(step_dir / "optimizer.eqx", opt_state)
(step_dir / "training-state.json").write_text(
    json.dumps({
        "format_version": 1,
        "step": 100,
        "config_fingerprint": config_fingerprint(model.config),
    })
    + "\n",
    encoding="utf-8",
)
```

On restore, load the strict model checkpoint first, verify the sidecar fingerprint, reconstruct the identical optimizer and schedule, and initialize a template state from the restored model before deserializing:

```python
metadata = json.loads((step_dir / "training-state.json").read_text(encoding="utf-8"))
restored_model = load_checkpoint(
    step_dir / "model.safetensors",
    key=jax.random.PRNGKey(1),
)
if metadata.get("format_version") != 1:
    raise ValueError("unsupported training-state format")
if metadata["config_fingerprint"] != config_fingerprint(restored_model.config):
    raise ValueError("optimizer state belongs to a different model configuration")

opt_state_template = optimizer.init(eqx.filter(restored_model, eqx.is_array))
restored_opt_state = eqx.tree_deserialise_leaves(
    step_dir / "optimizer.eqx",
    opt_state_template,
)
```

The template is the optimizer-state schema, so changing the optimizer transformation, schedule, parameter filtering, shape, or dtype invalidates the artifact. Record the exact optimizer and schedule configuration with the run metadata. Write each resume point into a fresh step directory and publish it only after the model, optimizer, and sidecar writes all succeed; never overwrite one member of an existing pair independently.

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

The loader validates:

- the complete exact key set and source tensor shapes;
- model-parallel merge axes and the CEMA real/complex representation;
- RoPE frequencies, plus-one normalization storage, and projection bias topology;
- tied-output equality and configuration compatibility.

When `share_emb=True`, the separately serialized embedding and output tensors must be bit-identical because they represent one logical upstream parameter; approximate equality is deliberately rejected.

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

- Weights use the original `(out_features, in_features)` layout without speculative transposes.
- Ordinary tensors follow the model's `param_dtype` by default; the exporter's explicit `dtype` argument can cast them. CEMA, normalization, RoPE, and other sensitive tensors remain FP32.
- CEMA gamma is reassembled as real/imag pairs, and omega regains its singleton axis.
- Tied output weights are emitted as an equal clone so serialization does not depend on shared storage.

## Compatibility boundary

The historical `convert_jax_to_torch`, `load_weights_from_torch`, `load_from_pretrained`, and `save_safetensors` functions guessed between incompatible downstream and native schemas. They now raise with the explicit replacement API.

Existing pre-v2 JAX checkpoints require an owner-controlled offline migration because their metadata does not identify RoPE coordinates, normalization storage, bias topology, tying intent, or preset identity. Runtime loading deliberately refuses to infer those choices. A safe migration must supply the original config and provenance, transform affected tensors, then write a v2 checkpoint and verify logits before deleting the source file.

## Inference state persistence

Model-forward caches have a separate versioned format:

```python
from megalodon_jax import load_inference_cache, save_inference_cache

save_inference_cache(cache, "cache.safetensors", config)
cache = load_inference_cache("cache.safetensors", config)
```

Cache files are bound to the full configuration fingerprint and validate the exact presence/tensor schema, fixed KV capacity, batch consistency, layer count, every state dtype/shape, and the mirrored attention position/count invariant. They are model-forward continuation artifacts — not portable model checkpoints, and not complete `generate()` continuation state.

Persist `GenerationState` when generation itself must resume without replaying a token:

```python
from megalodon_jax import load_generation_state, save_generation_state

save_generation_state(state, "generation-state.safetensors", config)
state = load_generation_state("generation-state.safetensors", config)
```

Generation-state files use their own fail-closed format and add exact next-logit, finished-row, and EOS schemas on top of the validated cache payload. The sampling PRNG key remains the explicit third value returned by `generate()`; persist and restore that key separately when stochastic continuation must be bit-for-bit reproducible.

## Parity gates

The load-bearing core-math checks use independent NumPy references for TimestepNorm, RoPE, initialization statistics, and dense attention. They do not import `megalodon_jax` production implementations as their expected-value path.

`tests/test_upstream_parity.py` uses a small differentiable Torch transcription of the released equations. It compares full logits, every trainable upstream-schema gradient, and short optimizer trajectories, but it was written alongside this JAX port and is not an independent oracle. Agreement demonstrates consistency between the two implementations; disagreement identifies a divergence that must be investigated rather than proving which side is correct.

`tools/verify_modeling_correctness.py` additionally reads a local upstream source path when it is available; without one it runs repository-only checks and reports source-anchoring checks as skipped. The strict conversion regression uses a hand-authored released-source manifest for keys, shapes, dtypes, literal RoPE frequencies, and model-parallel partition axes.

No check imports another Megalodon package or builds the fused extension; fused CUDA is checked at source level only. No released trained checkpoint is available, so there is no ground-truth logits comparison.
