"""Phase 5 inference and conversion tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax import MegalodonConfig, MegalodonForCausalLM
from megalodon_jax.checkpoint import (
    load_checkpoint,
    load_inference_cache,
    load_partial_checkpoint,
    model_state_dict,
    save_checkpoint,
    save_inference_cache,
)
from megalodon_jax.convert import (
    convert_jax_to_torch,
    export_upstream_state_dict,
    load_from_pretrained,
    load_upstream_checkpoint,
    load_upstream_state_dict,
    save_safetensors,
)
from megalodon_jax.inference import generate, index_cache, init_cache, sample_token
from megalodon_jax.types import AttentionCache, LayerCache, ModelCache
from tests.factories import tiny_config


def small_config() -> MegalodonConfig:
    """Create a tiny config for fast inference tests.

    :return MegalodonConfig: Minimal configuration for inference tests.
    """
    return tiny_config()


class TestCacheUtilities:
    """Cache initialization and batch-indexing helpers."""

    def test_init_cache_shapes(self) -> None:
        """Validate fixed-capacity cache initialization shapes.

        :return None: None.
        """
        config = small_config()
        cache = init_cache(config, batch_size=2, allocate_kv=True)

        layer0 = cache.layer_caches[0]
        assert layer0 is not None and layer0.attn is not None
        assert layer0.attn.k.shape == (
            2,
            config.cache_capacity,
            config.num_heads,
            config.head_dim,
        )

    def test_index_cache_slices_batch(self) -> None:
        """Ensure index_cache slices the batch dimension correctly.

        :return None: None.
        """
        config = small_config()
        cache = init_cache(config, batch_size=2, allocate_kv=True)
        layer0 = cache.layer_caches[0]
        assert layer0 is not None and layer0.attn is not None

        # Make batch slices distinguishable
        k_mod = layer0.attn.k.at[0].set(jnp.ones_like(layer0.attn.k[0]))
        k_mod = k_mod.at[1].set(jnp.full_like(layer0.attn.k[1], 2.0))
        attn_mod = AttentionCache(
            k=k_mod,
            v=layer0.attn.v,
            count=layer0.attn.count,
        )
        cache_mod = ModelCache(
            layer_caches=(
                LayerCache(
                    attn=attn_mod,
                    norm=layer0.norm,
                    ema=layer0.ema,
                    position=layer0.position,
                ),
            ),
            final_norm=cache.final_norm,
        )

        indexed = index_cache(cache_mod, jnp.array([1]))
        assert indexed.layer_caches[0].attn is not None
        np.testing.assert_allclose(
            np.array(indexed.layer_caches[0].attn.k[0]),
            np.array(k_mod[1]),
        )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
    def test_init_cache_rejects_dtype_override(self, dtype: jnp.dtype) -> None:
        """Cache construction and persistence share the model compute dtype."""
        config = replace(small_config(), compute_dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="cache dtype must equal config.compute_dtype"):
            init_cache(config, batch_size=1, dtype=dtype, allocate_kv=True)

    def test_preallocated_and_lazy_cache_defaults_match(self) -> None:
        """Config-built initial states match the layers' lazy state constructors."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
        lazy = init_cache(config, batch_size=1)
        preallocated = init_cache(
            config,
            batch_size=1,
            allocate_kv=True,
            allocate_norm=True,
            allocate_ema=True,
        )

        lazy_logits, lazy_result = model(tokens, cache=lazy, return_cache=True)
        allocated_logits, allocated_result = model(
            tokens,
            cache=preallocated,
            return_cache=True,
        )

        np.testing.assert_array_equal(np.asarray(allocated_logits), np.asarray(lazy_logits))
        assert lazy_result is not None and allocated_result is not None
        for expected, actual in zip(
            jax.tree.leaves(lazy_result),
            jax.tree.leaves(allocated_result),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


class TestSamplingAndGeneration:
    """Sampling primitives and generation loop."""

    def test_sample_token_top_k(self) -> None:
        """Test top-k sampling selects the highest logit.

        :return None: None.
        """
        logits = jnp.array([[0.1, 2.0, -1.0]], dtype=jnp.float32)
        token = sample_token(
            logits,
            jax.random.PRNGKey(0),
            temperature=0.0,
            top_k=1,
        )
        assert int(token[0]) == 1

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"temperature": float("nan")}, "temperature must be finite"),
            ({"temperature": float("inf")}, "temperature must be finite"),
            ({"temperature": -1.0}, "temperature must be finite"),
            ({"temperature": 0.0, "top_k": -1}, "top_k must be"),
            ({"temperature": 0.0, "top_k": 4}, "top_k must be"),
            ({"temperature": 0.0, "top_p": 0.0}, "top_p must be finite"),
            ({"temperature": 0.0, "top_p": float("nan")}, "top_p must be finite"),
        ],
    )
    def test_sample_controls_validate_before_greedy(
        self, kwargs: dict[str, float | int], message: str
    ) -> None:
        """Greedy execution must not bypass invalid sampling controls."""
        logits = jnp.zeros((1, 3), dtype=jnp.float32)
        with pytest.raises(ValueError, match=message):
            sample_token(logits, None, **kwargs)

    def test_generate_shapes_and_determinism(self) -> None:
        """Test generate shape, determinism, and vocabulary bounds.

        :return None: None.
        """
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        key = jax.random.PRNGKey(42)
        out1, _, key1 = generate(
            model,
            prompt,
            max_new_tokens=4,
            key=key,
            temperature=0.0,  # greedy for determinism
        )
        out2, _, key2 = generate(
            model,
            prompt,
            max_new_tokens=4,
            key=key,
            temperature=0.0,
        )

        assert out1.shape == (1, 7)
        np.testing.assert_array_equal(np.array(out1), np.array(out2))
        np.testing.assert_array_equal(np.array(key1), np.array(key2))
        assert jnp.all(out1 >= 0)
        assert jnp.all(out1 < config.vocab_size)

    def test_generate_single_token_updates_cache(self) -> None:
        """Test single-token generation updates cache counts.

        :return None: None.
        """
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        out, cache, _ = generate(
            model,
            prompt,
            max_new_tokens=1,
            key=jax.random.PRNGKey(123),
            temperature=0.0,
            return_cache=True,
        )

        assert out.shape == (1, 4)
        assert cache is not None
        layer0 = cache.layer_caches[0]
        assert layer0 is not None and layer0.attn is not None
        assert int(layer0.attn.count) == prompt.shape[1] + 1

    def test_generate_multi_token_updates_cache(self) -> None:
        """Test multi-token generation updates cache counts.

        :return None: None.
        """
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        out, cache, _ = generate(
            model,
            prompt,
            max_new_tokens=3,
            key=jax.random.PRNGKey(123),
            temperature=0.0,
            return_cache=True,
        )

        assert out.shape == (1, 6)
        assert cache is not None
        layer0 = cache.layer_caches[0]
        assert layer0 is not None and layer0.attn is not None
        assert int(layer0.attn.count) == prompt.shape[1] + 3

    @pytest.mark.parametrize("max_new_tokens", [1, 2])
    @pytest.mark.parametrize("return_cache", [False, True])
    def test_generate_canonicalizes_all_true_mask(
        self, max_new_tokens: int, return_cache: bool
    ) -> None:
        """An all-valid mask is exactly equivalent to omitting mask metadata."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
        expected, _, _ = generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            return_cache=return_cache,
        )
        actual, cache, _ = generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            attention_mask=jnp.ones_like(prompt, dtype=jnp.bool_),
            return_cache=return_cache,
        )

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert (cache is not None) is return_cache

    def test_generate_rejects_non_vocabulary_output_space(self) -> None:
        """Autoregressive output IDs must always be valid embedding IDs."""
        config = replace(small_config(), output_size=small_config().vocab_size + 1)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="effective_output_size.*vocab_size"):
            generate(
                model,
                jnp.asarray([[1, 2]], dtype=jnp.int32),
                max_new_tokens=1,
                temperature=0.0,
            )

    def test_generate_zero_new_tokens_raises(self) -> None:
        """Ensure generate rejects zero-length outputs.

        :return None: None.
        """
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="max_new_tokens"):
            generate(
                model,
                prompt,
                max_new_tokens=0,
                key=jax.random.PRNGKey(0),
                temperature=0.0,
            )

    def test_generate_uses_last_valid_token_with_padding(
        self,
    ) -> None:
        """Greedy generation selects logits from the last valid right-padded token."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = [[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]]
        attention_mask = [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
        prompt_array = jnp.asarray(prompt, dtype=jnp.int32)
        mask_array = jnp.asarray(attention_mask)

        logits, _ = model(prompt_array, attention_mask=mask_array, return_cache=False)
        positions = jnp.arange(prompt_array.shape[1], dtype=jnp.int32)
        masked_positions = jnp.where(mask_array, positions, -1)
        last_idx = masked_positions.max(axis=1)
        gather_idx = jnp.broadcast_to(last_idx[:, None, None], (2, 1, logits.shape[-1]))
        last_logits = jnp.take_along_axis(logits, gather_idx, axis=1)[:, 0, :]
        expected = jnp.argmax(last_logits, axis=-1)

        out, _, _ = generate(
            model,
            prompt_array,
            max_new_tokens=1,
            key=jax.random.PRNGKey(123),
            temperature=0.0,
            attention_mask=mask_array,
            return_cache=False,
        )

        np.testing.assert_array_equal(np.array(out[:, -1]), np.array(expected))

    def test_generate_rejects_left_padding(self) -> None:
        """Physical left padding cannot silently shift chunk-local semantics."""
        model = MegalodonForCausalLM(small_config(), key=jax.random.PRNGKey(0))
        with pytest.raises(Exception, match="left-padded.*right padding"):
            generate(
                model,
                jnp.asarray([[0, 0, 1, 2]], dtype=jnp.int32),
                max_new_tokens=1,
                temperature=0.0,
                attention_mask=jnp.asarray([[False, False, True, True]]),
            )

    def test_generate_empty_prompt_uses_bos(self) -> None:
        """Ensure empty prompts use BOS token.

        :return None: None.
        """
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.empty((1, 0), dtype=jnp.int32)

        out, _, _ = generate(
            model,
            prompt,
            max_new_tokens=1,
            temperature=0.0,
        )

        assert out.shape == (1, 2)
        assert int(out[0, 0]) == config.bos_token_id

    def test_generate_sampling_requires_key(self) -> None:
        """Ensure sampling requires a PRNG key.

        :return None: None.
        """
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="key is required"):
            generate(
                model,
                prompt,
                max_new_tokens=1,
                temperature=1.0,
            )

    @pytest.mark.parametrize(
        ("attention_mask", "max_new_tokens", "return_cache", "with_cache"),
        [
            pytest.param([[False, False, True, True]], 2, False, False, id="left-multistep"),
            pytest.param([[True, True, False, False]], 2, False, False, id="right-multistep"),
            pytest.param([[False, True, True, True]], 1, True, False, id="return-cache"),
            pytest.param([[False, True, True, True]], 1, False, True, id="existing-cache"),
        ],
    )
    def test_generate_padded_cache_modes_raise(
        self,
        attention_mask: list[list[bool]],
        max_new_tokens: int,
        return_cache: bool,
        with_cache: bool,
    ) -> None:
        """Every cache-enabling mode rejects padded generation explicitly."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        cache = init_cache(config, batch_size=1) if with_cache else None

        with pytest.raises(ValueError, match="Cannot use cache with padded attention_mask"):
            generate(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                attention_mask=jnp.asarray(attention_mask),
                cache=cache,
                return_cache=return_cache,
            )


class TestConversion:
    """Strict native and original-upstream checkpoint tests."""

    @pytest.mark.torch_ref
    def test_original_upstream_manifest_is_source_transcribed(self) -> None:
        """Check converter keys, shapes, and dtypes against a hand-authored source manifest."""
        torch = pytest.importorskip("torch")
        config = MegalodonConfig(
            vocab_size=17,
            output_size=19,
            model_dim=8,
            num_layers=2,
            num_heads=2,
            z_dim=8,
            value_dim=8,
            ffn_hidden_dim=12,
            cema_ndim=2,
            chunk_size=8,
            norm_num_groups=2,
            swiglu=True,
            rescale_nffn=True,
            share_emb=False,
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        # Literal dimensions below are transcribed from the constructors in
        # the released moving-average attention, CEMA, TimestepNorm, NFFN,
        # rotary, and output-layer sources. Do not derive this mapping through
        # either converter direction or through JAX parameter leaves.
        manifest: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "embed.weight": ((17, 8), torch.float32),
            "rope.freqs": ((2,), torch.float32),
            "output.final_norm.prior_count": ((), torch.int64),
            "output.final_norm.prior_mean": ((2,), torch.float32),
            "output.final_norm.prior_logv": ((2,), torch.float32),
            "output.final_norm.weight": ((8,), torch.float32),
            "output.final_norm.bias": ((8,), torch.float32),
            "output.output.weight": ((19, 8), torch.float32),
        }
        for index in range(2):
            ap = f"layers.{index}.mega"
            fp = f"layers.{index}.nffn"
            manifest.update(
                {
                    f"{ap}.timenorm.prior_count": ((), torch.int64),
                    f"{ap}.timenorm.prior_mean": ((2,), torch.float32),
                    f"{ap}.timenorm.prior_logv": ((2,), torch.float32),
                    f"{ap}.timenorm.weight": ((8,), torch.float32),
                    f"{ap}.timenorm.bias": ((8,), torch.float32),
                    f"{ap}.cema.alpha": ((8, 2, 1), torch.float32),
                    f"{ap}.cema.delta": ((8, 2, 1), torch.float32),
                    f"{ap}.cema.theta": ((8, 1, 1), torch.float32),
                    f"{ap}.cema.gamma": ((8, 2, 2), torch.float32),
                    f"{ap}.cema.omega": ((8, 1), torch.float32),
                    f"{ap}.rmsnorm.weight": ((8,), torch.float32),
                    f"{ap}.wz.weight": ((8, 8), torch.float32),
                    f"{ap}.wz.bias": ((8,), torch.float32),
                    f"{ap}.wv.weight": ((8, 8), torch.float32),
                    f"{ap}.wv.bias": ((8,), torch.float32),
                    f"{ap}.wr.weight": ((8, 8), torch.float32),
                    f"{ap}.wr.bias": ((8,), torch.float32),
                    f"{ap}.wh1.weight": ((8, 8), torch.float32),
                    f"{ap}.wh1.bias": ((8,), torch.float32),
                    f"{ap}.wh2.weight": ((8, 8), torch.float32),
                    f"{ap}.gamma": ((2, 8), torch.float32),
                    f"{ap}.beta": ((2, 8), torch.float32),
                    f"{fp}.norm.weight": ((8,), torch.float32),
                    f"{fp}.norm.bias": ((8,), torch.float32),
                    f"{fp}.fc1.weight": ((12, 8), torch.float32),
                    f"{fp}.fc2.weight": ((8, 12), torch.float32),
                    f"{fp}.fc3.weight": ((12, 8), torch.float32),
                    f"{fp}.alpha": ((8,), torch.float32),
                }
            )

        exported = export_upstream_state_dict(model)
        assert set(exported) == set(manifest)
        for name, (shape, dtype) in manifest.items():
            assert tuple(exported[name].shape) == shape, name
            assert exported[name].dtype == dtype, name

        generator = torch.Generator().manual_seed(1729)
        source_state: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in manifest.items():
            if dtype == torch.int64 or name.endswith("prior_mean") or name.endswith("prior_logv"):
                source_state[name] = torch.zeros(shape, dtype=dtype)
            elif name == "rope.freqs":
                source_state[name] = torch.tensor([1.0, 0.01], dtype=torch.float32)
            else:
                source_state[name] = torch.randn(shape, dtype=dtype, generator=generator)
        loaded = load_upstream_state_dict(
            MegalodonForCausalLM(config, key=jax.random.PRNGKey(1)),
            source_state,
        )
        roundtripped = export_upstream_state_dict(loaded)
        for name in manifest:
            assert torch.equal(roundtripped[name], source_state[name]), name

    def test_native_v2_roundtrip_is_exact(self, tmp_path: Path) -> None:
        """Native save/reload preserves every tensor and model output."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        path = tmp_path / "model.safetensors"

        save_checkpoint(model, path)
        loaded = load_checkpoint(path, key=jax.random.PRNGKey(1))

        expected = model_state_dict(model)
        actual = model_state_dict(loaded)
        assert expected.keys() == actual.keys()
        for name in expected:
            np.testing.assert_array_equal(np.asarray(actual[name]), np.asarray(expected[name]))

        tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
        expected_logits, _ = model(tokens)
        actual_logits, _ = loaded(tokens)
        np.testing.assert_array_equal(np.asarray(actual_logits), np.asarray(expected_logits))

    def test_metadata_free_and_legacy_files_are_rejected(self, tmp_path: Path) -> None:
        """SafeTensors without explicit v2 semantics must never be guessed."""
        from safetensors.flax import save_file

        path = tmp_path / "legacy.safetensors"
        save_file({"weight": jnp.ones((2, 2), dtype=jnp.float32)}, str(path))
        with pytest.raises(ValueError, match="legacy JAX checkpoint"):
            load_checkpoint(path, key=jax.random.PRNGKey(0))

    @pytest.mark.parametrize(
        ("field", "invalid", "message"),
        [
            ("initializer_schema", "unknown", "initializer schema"),
            ("dtype_policy", "float16-everywhere", "dtype policy"),
        ],
    )
    def test_native_metadata_semantics_are_strict(
        self,
        tmp_path: Path,
        field: str,
        invalid: str,
        message: str,
    ) -> None:
        """Required native metadata must carry recognized values, not just keys."""
        from safetensors import safe_open
        from safetensors.flax import load_file, save_file

        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        valid = tmp_path / "valid.safetensors"
        save_checkpoint(model, valid)
        with safe_open(str(valid), framework="flax") as handle:
            metadata = dict(handle.metadata() or {})
        metadata[field] = invalid
        corrupted = tmp_path / f"invalid-{field}.safetensors"
        save_file(load_file(str(valid)), str(corrupted), metadata=metadata)
        with pytest.raises(ValueError, match=message):
            load_checkpoint(corrupted, key=jax.random.PRNGKey(1))

    def test_partial_restore_requires_explicit_names(self, tmp_path: Path) -> None:
        """Partial restore reports restored and freshly initialized leaves."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        path = tmp_path / "partial.safetensors"
        save_checkpoint(model, path)

        loaded, report = load_partial_checkpoint(
            path,
            config,
            include={"model.embed.weight"},
            key=jax.random.PRNGKey(1),
        )
        np.testing.assert_array_equal(
            np.asarray(loaded.model.embed.weight),
            np.asarray(model.model.embed.weight),
        )
        assert report["restored"] == ["model.embed.weight"]
        assert report["initialized"]
        assert report["exact_config_match"] is True
        assert report["source_config_fingerprint"] == report["target_config_fingerprint"]

        changed_config = replace(config, dropout=0.1)
        _, changed_report = load_partial_checkpoint(
            path,
            changed_config,
            include={"model.embed.weight"},
            key=jax.random.PRNGKey(2),
        )
        assert changed_report["exact_config_match"] is False
        assert (
            changed_report["source_config_fingerprint"]
            != changed_report["target_config_fingerprint"]
        )
        with pytest.raises(ValueError, match="partial restore selection"):
            load_partial_checkpoint(
                path,
                config,
                include={"not.a.parameter"},
                key=jax.random.PRNGKey(1),
            )

    @pytest.mark.torch_ref
    def test_original_upstream_roundtrip_and_keyspace(self) -> None:
        """Exact released keys round-trip without transposes or schema aliases."""
        pytest.importorskip("torch")
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state = export_upstream_state_dict(model)

        assert "embed.weight" in state
        assert "rope.freqs" in state
        assert "layers.0.mega.cema.gamma" in state
        assert "layers.0.nffn.fc1.weight" in state
        assert "output.final_norm.weight" in state
        assert "output.output.weight" in state
        assert "model.embed.weight" not in state
        assert "layers.0.mega.wh2.bias" not in state
        assert "layers.0.nffn.fc1.bias" not in state

        loaded = load_upstream_state_dict(
            MegalodonForCausalLM(config, key=jax.random.PRNGKey(1)),
            state,
        )
        tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
        expected_logits, _ = model(tokens)
        actual_logits, _ = loaded(tokens)
        np.testing.assert_array_equal(np.asarray(actual_logits), np.asarray(expected_logits))

    @pytest.mark.torch_ref
    def test_original_upstream_strict_failure(self) -> None:
        """Missing, unexpected, and Hugging-Face-shaped keys fail closed."""
        pytest.importorskip("torch")
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state = export_upstream_state_dict(model)
        state.pop("layers.0.mega.wh2.weight")
        state["model.embed.weight"] = state["embed.weight"]
        with pytest.raises(ValueError, match="strict original-upstream key mismatch"):
            load_upstream_state_dict(model, state)

    @pytest.mark.torch_ref
    def test_tied_output_contract(self) -> None:
        """Tied upstream copies must remain bit-identical, without tolerance."""
        torch = pytest.importorskip("torch")
        config = replace(small_config(), share_emb=True)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state = export_upstream_state_dict(model)
        assert state["output.output.weight"].data_ptr() != state["embed.weight"].data_ptr()
        assert state["output.output.weight"].equal(state["embed.weight"])

        value = state["output.output.weight"][0, 0]
        state["output.output.weight"][0, 0] = torch.nextafter(value, value + 1.0)
        with pytest.raises(ValueError, match="bit-identical"):
            load_upstream_state_dict(model, state)

    @pytest.mark.torch_ref
    def test_consolidated_model_parallel_shards(self, tmp_path: Path) -> None:
        """Declared model-parallel axes merge back into a world-size-one model."""
        torch = pytest.importorskip("torch")
        from megalodon_jax.convert import _merge_axis

        config = replace(small_config(), rescale_nffn=True)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state = export_upstream_state_dict(model)
        shards: list[dict[str, torch.Tensor]] = [{}, {}]
        for name, value in state.items():
            axis = _merge_axis(name)
            if name.endswith(".nffn.alpha"):
                assert axis is None
            if axis is None:
                shards[0][name] = value.clone()
                shards[1][name] = value.clone()
            else:
                pieces = torch.chunk(value, 2, dim=axis)
                assert len(pieces) == 2
                shards[0][name] = pieces[0].clone()
                shards[1][name] = pieces[1].clone()

        (tmp_path / "consolidate_config.json").write_text(
            '{"model_parallel_size": 2, "dtype": "fp32"}',
            encoding="utf-8",
        )
        torch.save(shards[0], tmp_path / "consolidated.00.pth")
        torch.save(shards[1], tmp_path / "consolidated.01.pth")
        loaded = load_upstream_checkpoint(
            tmp_path,
            config,
            key=jax.random.PRNGKey(1),
        )
        tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
        expected, _ = model(tokens)
        actual, _ = loaded(tokens)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert loaded.model.layers[0].ffn.alpha is not None
        np.testing.assert_array_equal(
            np.asarray(loaded.model.layers[0].ffn.alpha),
            np.asarray(model.model.layers[0].ffn.alpha),
        )

    @pytest.mark.parametrize("compute_dtype", [jnp.float32, jnp.bfloat16])
    def test_cache_roundtrip_and_config_binding(
        self,
        tmp_path: Path,
        compute_dtype: jnp.dtype,
    ) -> None:
        """Serialized continuation state resumes exactly and rejects other configs."""
        config = replace(small_config(), compute_dtype=compute_dtype)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        _, cache = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32), return_cache=True)
        assert cache is not None
        path = tmp_path / "cache.safetensors"
        save_inference_cache(cache, path, config)
        loaded = load_inference_cache(path, config)

        for expected, actual in zip(
            jax.tree_util.tree_leaves(cache),
            jax.tree_util.tree_leaves(loaded),
        ):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

        continuation = jnp.asarray([[4, 5]], dtype=jnp.int32)
        expected_logits, _ = model(continuation, cache=cache, return_cache=True)
        actual_logits, _ = model(continuation, cache=loaded, return_cache=True)
        np.testing.assert_array_equal(np.asarray(actual_logits), np.asarray(expected_logits))

        incompatible = replace(config, attention_window=4)
        with pytest.raises(ValueError, match="fingerprint"):
            load_inference_cache(path, incompatible)

    def test_cache_schema_and_position_invariants_fail_closed(self, tmp_path: Path) -> None:
        """Presence metadata and redundant attention positions must remain self-consistent."""
        from safetensors import safe_open
        from safetensors.flax import load_file, save_file

        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        _, cache = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32), return_cache=True)
        assert cache is not None
        valid = tmp_path / "valid-cache.safetensors"
        save_inference_cache(cache, valid, config)
        tensors = load_file(str(valid))
        with safe_open(str(valid), framework="flax") as handle:
            metadata = dict(handle.metadata() or {})

        bad_position = dict(tensors)
        bad_position["layers.0.position"] = bad_position["layers.0.position"] + 1
        position_path = tmp_path / "bad-position.safetensors"
        save_file(bad_position, str(position_path), metadata=metadata)
        with pytest.raises(ValueError, match="does not equal attention count"):
            load_inference_cache(position_path, config)

        bad_presence = dict(metadata)
        present = json.loads(bad_presence["present_json"])
        bad_presence["present_json"] = json.dumps([*present, "layers.0.unknown"])
        presence_path = tmp_path / "bad-presence.safetensors"
        save_file(tensors, str(presence_path), metadata=bad_presence)
        with pytest.raises(ValueError, match="unknown entries"):
            load_inference_cache(presence_path, config)

    def test_sparse_cache_layer_presence_uses_exact_layer_names(self, tmp_path: Path) -> None:
        """Layer 10 presence must not imply that similarly prefixed layer 1 exists."""
        config = replace(small_config(), num_layers=11)
        initialized = init_cache(config, batch_size=1, allocate_kv=True)
        sparse = ModelCache(
            layer_caches=(*([None] * 10), initialized.layer_caches[10]),
            final_norm=None,
        )
        path = tmp_path / "sparse-cache.safetensors"

        save_inference_cache(sparse, path, config)
        loaded = load_inference_cache(path, config)

        assert loaded.layer_caches[:10] == (None,) * 10
        assert loaded.layer_caches[10] is not None

    def test_ambiguous_legacy_apis_refuse(self, tmp_path: Path) -> None:
        """Historical schema-guessing entry points provide migration guidance."""
        model = MegalodonForCausalLM(small_config(), key=jax.random.PRNGKey(0))
        with pytest.raises(RuntimeError, match="export_upstream_state_dict"):
            convert_jax_to_torch(model)
        with pytest.raises(RuntimeError, match="save_checkpoint"):
            save_safetensors(model, tmp_path / "model.safetensors")
        with pytest.raises(RuntimeError, match="load_checkpoint"):
            load_from_pretrained(tmp_path / "model.safetensors")
