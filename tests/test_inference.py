"""Phase 5 inference and conversion tests."""

import jax
import jax.numpy as jnp
import numpy as np

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
        out1, _ = generate(
            model,
            prompt,
            max_new_tokens=4,
            key=key,
            temperature=0.0,  # greedy for determinism
        )
        out2, _ = generate(
            model,
            prompt,
            max_new_tokens=4,
            key=key,
            temperature=0.0,
        )

        assert out1.shape == (1, 7)
        np.testing.assert_array_equal(np.array(out1), np.array(out2))
        assert jnp.all(out1 >= 0)
        assert jnp.all(out1 < config.vocab_size)


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
