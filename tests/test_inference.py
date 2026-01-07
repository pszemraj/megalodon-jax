"""Phase 5 inference and conversion tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax import (
    MegalodonConfig,
    MegalodonForCausalLM,
    convert_jax_to_torch,
    load_from_pretrained,
    load_weights_from_torch,
    save_safetensors,
)
from megalodon_jax.inference import generate, index_cache, init_cache, sample_token, trim_cache
from megalodon_jax.types import AttentionCache, LayerCache, ModelCache


def small_config() -> MegalodonConfig:
    """Create a tiny config for fast inference tests."""
    return MegalodonConfig(
        vocab_size=64,
        model_dim=64,
        num_layers=1,
        num_heads=2,
        z_dim=32,
        value_dim=64,
        ffn_hidden_dim=128,
        cema_ndim=4,
        chunk_size=8,
        norm_num_groups=8,
    )


class TestCacheUtilities:
    """Cache init/index/trim helpers."""

    def test_init_and_trim_cache_shapes(self):
        config = small_config()
        cache = init_cache(config, batch_size=2, allocate_kv=True)

        layer0 = cache.layer_caches[0]
        assert layer0 is not None and layer0.attn is not None
        assert layer0.attn.k.shape == (
            2,
            config.effective_max_cache_len,
            config.num_heads,
            config.head_dim,
        )

        # Extend cache artificially then trim back
        attn = layer0.attn
        pad = 2
        extended_attn = AttentionCache(
            k=jnp.pad(attn.k, ((0, 0), (0, pad), (0, 0), (0, 0))),
            v=jnp.pad(attn.v, ((0, 0), (0, pad), (0, 0), (0, 0))),
            count=attn.count,
        )
        extended_layer = LayerCache(
            attn=extended_attn,
            norm=layer0.norm,
            ema=layer0.ema,
            position=layer0.position,
        )
        extended_cache = ModelCache(
            layer_caches=(extended_layer,),
            final_norm=cache.final_norm,
        )

        trimmed = trim_cache(extended_cache, config.effective_max_cache_len)
        trimmed_attn = trimmed.layer_caches[0].attn
        assert trimmed_attn is not None
        assert trimmed_attn.k.shape[1] == config.effective_max_cache_len

    def test_index_cache_slices_batch(self):
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

    def test_trim_cache_preserves_ring_order(self):
        cache_size = 6
        max_len = 4
        batch = 1
        heads = 1
        head_dim = 1
        value_dim = 1

        k = jnp.zeros((batch, cache_size, heads, head_dim), dtype=jnp.float32)
        v = jnp.zeros((batch, cache_size, heads, value_dim), dtype=jnp.float32)
        for token in range(4, 10):
            idx = token % cache_size
            k = k.at[:, idx, 0, 0].set(float(token))
            v = v.at[:, idx, 0, 0].set(float(token))

        attn_cache = AttentionCache(
            k=k,
            v=v,
            count=jnp.array(10, dtype=jnp.int32),
        )
        layer_cache = LayerCache(
            attn=attn_cache,
            norm=None,
            ema=None,
            position=jnp.array(0, dtype=jnp.int32),
        )
        cache = ModelCache(layer_caches=(layer_cache,), final_norm=None)

        trimmed = trim_cache(cache, max_len)
        trimmed_attn = trimmed.layer_caches[0].attn
        assert trimmed_attn is not None
        assert trimmed_attn.k.shape == (batch, max_len, heads, head_dim)
        assert trimmed_attn.count == attn_cache.count

        expected_k = jnp.zeros((batch, max_len, heads, head_dim), dtype=jnp.float32)
        expected_v = jnp.zeros((batch, max_len, heads, value_dim), dtype=jnp.float32)
        for token in range(6, 10):
            idx = token % max_len
            expected_k = expected_k.at[:, idx, 0, 0].set(float(token))
            expected_v = expected_v.at[:, idx, 0, 0].set(float(token))

        np.testing.assert_allclose(np.array(trimmed_attn.k), np.array(expected_k))
        np.testing.assert_allclose(np.array(trimmed_attn.v), np.array(expected_v))


class TestSamplingAndGeneration:
    """Sampling primitives and generation loop."""

    def test_sample_token_top_k(self):
        logits = jnp.array([[0.1, 2.0, -1.0]], dtype=jnp.float32)
        token = sample_token(
            logits,
            jax.random.PRNGKey(0),
            temperature=0.0,
            top_k=1,
        )
        assert int(token[0]) == 1

    def test_generate_shapes_and_determinism(self):
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

    def test_generate_single_token_updates_cache(self):
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

    def test_generate_multi_token_updates_cache(self):
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

    def test_generate_zero_new_tokens_raises(self):
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

    def test_generate_uses_last_valid_token_with_padding(self):
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        prompt = jnp.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 0, 0],
            ],
            dtype=jnp.int32,
        )
        attention_mask = jnp.array(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
            ]
        )

        logits, _ = model(prompt, attention_mask=attention_mask, return_cache=False)
        positions = jnp.arange(prompt.shape[1], dtype=jnp.int32)
        masked_positions = jnp.where(attention_mask, positions, -1)
        last_idx = masked_positions.max(axis=1)
        gather_idx = jnp.broadcast_to(last_idx[:, None, None], (2, 1, logits.shape[-1]))
        last_logits = jnp.take_along_axis(logits, gather_idx, axis=1)[:, 0, :]
        expected = jnp.argmax(last_logits, axis=-1)

        out, _, _ = generate(
            model,
            prompt,
            max_new_tokens=1,
            key=jax.random.PRNGKey(123),
            temperature=0.0,
            attention_mask=attention_mask,
            return_cache=False,
        )

        np.testing.assert_array_equal(np.array(out[:, -1]), np.array(expected))

    def test_generate_uses_last_valid_token_with_left_padding(self):
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        prompt = jnp.array(
            [
                [0, 0, 1, 2, 3],
                [0, 4, 5, 6, 7],
            ],
            dtype=jnp.int32,
        )
        attention_mask = jnp.array(
            [
                [False, False, True, True, True],
                [False, True, True, True, True],
            ]
        )

        logits, _ = model(prompt, attention_mask=attention_mask, return_cache=False)
        positions = jnp.arange(prompt.shape[1], dtype=jnp.int32)
        masked_positions = jnp.where(attention_mask, positions, -1)
        last_idx = masked_positions.max(axis=1)
        gather_idx = jnp.broadcast_to(last_idx[:, None, None], (2, 1, logits.shape[-1]))
        last_logits = jnp.take_along_axis(logits, gather_idx, axis=1)[:, 0, :]
        expected = jnp.argmax(last_logits, axis=-1)

        out, _, _ = generate(
            model,
            prompt,
            max_new_tokens=1,
            key=jax.random.PRNGKey(123),
            temperature=0.0,
            attention_mask=attention_mask,
            return_cache=False,
        )

        np.testing.assert_array_equal(np.array(out[:, -1]), np.array(expected))

    def test_generate_empty_prompt_uses_bos(self):
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

    def test_generate_sampling_requires_key_or_seed(self):
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

    def test_generate_grouped_left_padding_multi_token(self):
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        prompt = jnp.array(
            [
                [0, 0, 1, 2],
                [0, 3, 4, 5],
            ],
            dtype=jnp.int32,
        )
        attention_mask = jnp.array(
            [
                [False, False, True, True],
                [False, True, True, True],
            ]
        )

        out, _, _ = generate(
            model,
            prompt,
            max_new_tokens=2,
            temperature=0.0,
            attention_mask=attention_mask,
            return_cache=False,
        )

        trimmed0 = prompt[0:1, 2:]
        trimmed1 = prompt[1:2, 1:]
        out0, _, _ = generate(
            model,
            trimmed0,
            max_new_tokens=2,
            temperature=0.0,
        )
        out1, _, _ = generate(
            model,
            trimmed1,
            max_new_tokens=2,
            temperature=0.0,
        )

        np.testing.assert_array_equal(np.array(out[0, 4:]), np.array(out0[0, 2:]))
        np.testing.assert_array_equal(np.array(out[1, 4:]), np.array(out1[0, 3:]))

    def test_generate_right_padding_raises(self):
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        prompt = jnp.array([[1, 2, 0, 0]], dtype=jnp.int32)
        attention_mask = jnp.array([[True, True, False, False]])

        with pytest.raises(ValueError, match="Right padding"):
            generate(
                model,
                prompt,
                max_new_tokens=2,
                temperature=0.0,
                attention_mask=attention_mask,
            )

    def test_generate_padded_return_cache_raises(self):
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        prompt = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        attention_mask = jnp.array([[False, True, True, True]])

        with pytest.raises(ValueError, match="return_cache"):
            generate(
                model,
                prompt,
                max_new_tokens=2,
                temperature=0.0,
                attention_mask=attention_mask,
                return_cache=True,
            )


class TestConversion:
    """Weight conversion + SafeTensors roundtrip."""

    def test_torch_roundtrip_matches_logits(self):
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state_dict = convert_jax_to_torch(model)

        model_copy = MegalodonForCausalLM(config, key=jax.random.PRNGKey(1))
        model_copy = load_weights_from_torch(model_copy, state_dict)

        input_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        logits1, _ = model(input_ids, deterministic=True)
        logits2, _ = model_copy(input_ids, deterministic=True)

        np.testing.assert_allclose(np.array(logits1), np.array(logits2), rtol=1e-5, atol=1e-5)

    def test_safetensors_roundtrip(self, tmp_path):
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        path = tmp_path / "model.safetensors"
        save_safetensors(model, path)

        loaded = load_from_pretrained(
            str(path),
            config=config,
            key=jax.random.PRNGKey(1),
        )

        input_ids = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        logits_orig, _ = model(input_ids, deterministic=True)
        logits_loaded, _ = loaded(input_ids, deterministic=True)

        np.testing.assert_allclose(
            np.array(logits_orig),
            np.array(logits_loaded),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_load_weights_requires_lm_head_for_untied(self):
        config = MegalodonConfig(
            vocab_size=64,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=8,
            output_size=32,
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state_dict = convert_jax_to_torch(model)
        state_dict.pop("lm_head.weight")

        with pytest.raises(KeyError, match="lm_head.weight"):
            load_weights_from_torch(
                MegalodonForCausalLM(config, key=jax.random.PRNGKey(1)),
                state_dict,
            )

    def test_load_weights_rejects_swiglu_mismatch(self):
        config_swiglu = MegalodonConfig(
            vocab_size=64,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=8,
            swiglu=True,
        )
        model_swiglu = MegalodonForCausalLM(config_swiglu, key=jax.random.PRNGKey(0))
        state_dict = convert_jax_to_torch(model_swiglu)

        config_no_swiglu = MegalodonConfig(
            vocab_size=64,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=8,
            swiglu=False,
        )

        with pytest.raises(ValueError, match="swiglu"):
            load_weights_from_torch(
                MegalodonForCausalLM(config_no_swiglu, key=jax.random.PRNGKey(1)),
                state_dict,
            )

    def test_tied_model_export_has_lm_head(self):
        """Tied model should export lm_head.weight for PyTorch strict loading."""
        config = small_config()  # default is tied (output_size=-1)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        assert model.tied, "Default model should be tied"

        state_dict = convert_jax_to_torch(model)

        # Must have lm_head.weight for PyTorch strict loading compatibility
        assert "lm_head.weight" in state_dict
        assert "model.embed.weight" in state_dict

        # For tied models, lm_head.weight should equal embed.weight
        np.testing.assert_array_equal(
            state_dict["lm_head.weight"].numpy(),
            state_dict["model.embed.weight"].numpy(),
        )

    def test_untied_model_export(self):
        """Untied model should export separate lm_head.weight."""
        config = MegalodonConfig(
            vocab_size=64,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=8,
            output_size=32,  # Different from vocab_size -> untied
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        assert not model.tied, "Model should be untied with output_size != vocab_size"

        state_dict = convert_jax_to_torch(model)

        assert "lm_head.weight" in state_dict
        assert "model.embed.weight" in state_dict

        # For untied, shapes should differ
        assert state_dict["lm_head.weight"].shape[0] == 32
        assert state_dict["model.embed.weight"].shape[0] == 64

    def test_shape_validation_error(self):
        """Shape mismatch between config and checkpoint should raise ValueError."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state_dict = convert_jax_to_torch(model)

        # Create model with different model_dim
        config_wrong = MegalodonConfig(
            vocab_size=64,
            model_dim=128,  # Different from state_dict
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=8,
        )
        model_wrong = MegalodonForCausalLM(config_wrong, key=jax.random.PRNGKey(1))

        with pytest.raises(ValueError, match="Shape mismatch"):
            load_weights_from_torch(model_wrong, state_dict)

    def test_layer_count_mismatch_error(self):
        """Layer count mismatch should raise ValueError."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state_dict = convert_jax_to_torch(model)

        # Create model with different layer count
        config_wrong = MegalodonConfig(
            vocab_size=64,
            model_dim=64,
            num_layers=2,  # Different from state_dict (1 layer)
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=8,
        )
        model_wrong = MegalodonForCausalLM(config_wrong, key=jax.random.PRNGKey(1))

        with pytest.raises(ValueError, match="Layer count mismatch"):
            load_weights_from_torch(model_wrong, state_dict)

    def test_missing_key_error(self):
        """Missing required key should raise KeyError."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state_dict = convert_jax_to_torch(model)

        # Remove a required key
        del state_dict["model.embed.weight"]

        model_copy = MegalodonForCausalLM(config, key=jax.random.PRNGKey(1))

        with pytest.raises(KeyError, match="model.embed.weight"):
            load_weights_from_torch(model_copy, state_dict)

    def test_dtype_casting(self, tmp_path):
        """load_from_pretrained should cast to requested dtype."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        path = tmp_path / "model.safetensors"
        save_safetensors(model, path)

        loaded = load_from_pretrained(
            str(path),
            config=config,
            dtype=jnp.bfloat16,
            key=jax.random.PRNGKey(1),
        )

        # Check that floating point params are bf16
        assert loaded.model.embed.weight.dtype == jnp.bfloat16

    def test_missing_file_error(self):
        """load_from_pretrained should raise FileNotFoundError for missing file."""
        config = small_config()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_from_pretrained(
                "/nonexistent/path/model.safetensors",
                config=config,
                key=jax.random.PRNGKey(0),
            )
