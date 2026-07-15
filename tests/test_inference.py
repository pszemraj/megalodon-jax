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
from megalodon_jax.cache import CACHE_INVARIANT_MESSAGE, validate_model_cache_host
from megalodon_jax.checkpoint import (
    BF16_DTYPE_POLICY,
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
    load_weights_from_torch,
    save_safetensors,
)
from megalodon_jax.inference import generate, index_cache, init_cache, sample_token
from megalodon_jax.types import AttentionCache, LayerCache, ModelCache, NormState
from tests.factories import tiny_config


@pytest.mark.fast
class TestCacheUtilities:
    """Cache initialization and batch-indexing helpers."""

    def test_init_cache_is_sparse(self) -> None:
        """A pristine cache carries structure but allocates no history buffers.

        :return None: None.
        """
        config = tiny_config()
        cache = init_cache(config)

        assert cache.layer_caches == (None,) * config.num_layers
        assert cache.final_norm is None

    def test_pristine_cache_is_valid(self) -> None:
        """The public sparse zero-history representation satisfies cache invariants."""
        config = tiny_config()
        validate_model_cache_host(init_cache(config), config)

    def test_noncanonical_sparse_layer_is_invalid(self) -> None:
        """Top-level sparse caches use None rather than empty LayerCache objects."""
        config = tiny_config()
        malformed = ModelCache(
            layer_caches=(LayerCache(), *([None] * (config.num_layers - 1))),
            final_norm=None,
        )

        with pytest.raises(ValueError, match=CACHE_INVARIANT_MESSAGE):
            validate_model_cache_host(malformed, config)

    def test_index_cache_slices_batch(self) -> None:
        """Ensure index_cache slices the batch dimension correctly.

        :return None: None.
        """
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        _, cache = model(
            jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
            return_cache=True,
        )
        assert cache is not None
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

        reordered = index_cache(cache_mod, jnp.asarray([1, 0, 1], dtype=jnp.int32))
        assert reordered.layer_caches[0] is not None
        assert reordered.layer_caches[0].attn is not None
        np.testing.assert_allclose(
            np.asarray(reordered.layer_caches[0].attn.k[:, 0, 0, 0]),
            np.asarray([2.0, 1.0, 2.0]),
        )

        empty = index_cache(cache_mod, jnp.asarray([], dtype=jnp.int32))
        assert empty.layer_caches[0] is not None
        assert empty.layer_caches[0].attn is not None
        assert empty.layer_caches[0].attn.k.shape[0] == 0

    def test_index_cache_rejects_invalid_indices(self) -> None:
        """Beam-parent selection rejects ambiguous or out-of-range indices."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        _, cache = model(
            jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
            return_cache=True,
        )
        assert cache is not None

        with pytest.raises(ValueError, match="rank one"):
            index_cache(cache, jnp.asarray([[0]], dtype=jnp.int32))
        for invalid_dtype in (
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray([True], dtype=jnp.bool_),
            jnp.asarray([0], dtype=jnp.int16),
            np.asarray([2**32], dtype=np.int64),
        ):
            with pytest.raises(TypeError, match="dtype int32"):
                index_cache(cache, invalid_dtype)

        for invalid in (
            jnp.asarray([-1], dtype=jnp.int32),
            jnp.asarray([2], dtype=jnp.int32),
        ):
            with pytest.raises(Exception, match="cache indices must be in"):
                jax.block_until_ready(index_cache(cache, invalid))

        with pytest.raises(ValueError, match="without allocated batch state"):
            index_cache(init_cache(config), jnp.asarray([0], dtype=jnp.int32))
        empty_sparse = index_cache(init_cache(config), jnp.asarray([], dtype=jnp.int32))
        assert empty_sparse == init_cache(config)

    def test_explicit_pristine_and_none_cache_match(self) -> None:
        """The public sparse initializer is equivalent to an omitted cache."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
        pristine = init_cache(config)

        expected_logits, expected_cache = model(tokens, return_cache=True)
        actual_logits, actual_cache = model(
            tokens,
            cache=pristine,
            return_cache=True,
        )

        np.testing.assert_array_equal(np.asarray(actual_logits), np.asarray(expected_logits))
        assert expected_cache is not None and actual_cache is not None
        for expected, actual in zip(
            jax.tree.leaves(expected_cache),
            jax.tree.leaves(actual_cache),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.fast
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

    def test_top_k_ties_keep_exactly_k_candidates(self) -> None:
        """Tied kth logits do not expand the stochastic candidate set."""
        logits = jnp.zeros((1, 8), dtype=jnp.float32)
        keys = jax.random.split(jax.random.PRNGKey(7), 32)
        tokens = jax.vmap(lambda key: sample_token(logits, key, temperature=1.0, top_k=1))(keys)

        assert np.unique(np.asarray(tokens)).size == 1

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"temperature": float("nan")}, "temperature must be finite"),
            ({"temperature": float("inf")}, "temperature must be finite"),
            ({"temperature": -1.0}, "temperature must be finite"),
            ({"temperature": 0.0, "top_k": -1}, "top_k must be"),
            ({"temperature": 0.0, "top_k": 4}, "top_k must be"),
            ({"temperature": 0.0, "top_k": 1.5}, "top_k must be an integer"),
            ({"temperature": 0.0, "top_k": True}, "top_k must be an integer"),
            ({"temperature": 0.0, "top_p": 0.0}, "top_p must be finite"),
            ({"temperature": 0.0, "top_p": float("nan")}, "top_p must be finite"),
            ({"temperature": True}, "temperature must be a non-boolean"),
            ({"temperature": 0.0, "top_p": True}, "top_p must be a non-boolean"),
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
        config = tiny_config()
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

    @pytest.mark.parametrize("max_new_tokens", [1, 3])
    def test_generate_updates_cache(self, max_new_tokens: int) -> None:
        """Generation advances cache counts by the requested number of tokens."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        out, cache, _ = generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            key=jax.random.PRNGKey(123),
            temperature=0.0,
            return_cache=True,
        )

        assert out.shape == (1, prompt.shape[1] + max_new_tokens)
        assert cache is not None
        layer0 = cache.layer_caches[0]
        assert layer0 is not None and layer0.attn is not None
        assert int(layer0.attn.count) == prompt.shape[1] + max_new_tokens

    @pytest.mark.parametrize("max_new_tokens", [1, 2])
    @pytest.mark.parametrize("return_cache", [False, True])
    def test_generate_canonicalizes_all_true_mask(
        self, max_new_tokens: int, return_cache: bool
    ) -> None:
        """An all-valid mask is exactly equivalent to omitting mask metadata."""
        config = tiny_config()
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

    @pytest.mark.parametrize("mask_shape", [(1, 1), (2, 3), (3,)])
    def test_generate_rejects_wrong_shape_all_true_mask(self, mask_shape: tuple[int, ...]) -> None:
        """Mask shape is validated before an all-valid mask is discarded."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="attention_mask shape.*prompt_ids shape"):
            generate(
                model,
                prompt,
                max_new_tokens=1,
                temperature=0.0,
                attention_mask=jnp.ones(mask_shape, dtype=jnp.bool_),
            )

    def test_generate_rejects_non_vocabulary_output_space(self) -> None:
        """Autoregressive output IDs must always be valid embedding IDs."""
        config = replace(tiny_config(), output_size=tiny_config().vocab_size + 1)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="effective_output_size.*vocab_size"):
            generate(
                model,
                jnp.asarray([[1, 2]], dtype=jnp.int32),
                max_new_tokens=1,
                temperature=0.0,
            )

    @pytest.mark.parametrize("max_new_tokens", [-1, 0, 1.5, True])
    def test_generate_rejects_invalid_max_new_tokens(self, max_new_tokens: object) -> None:
        """Generation lengths must be positive integer values, excluding booleans."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="positive integer"):
            generate(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                key=jax.random.PRNGKey(0),
                temperature=0.0,
            )

    @pytest.mark.parametrize("name", ["bos_token_id", "eos_token_id"])
    @pytest.mark.parametrize("token_id", [-1, 1.5, True, 64])
    def test_generate_rejects_invalid_special_token_override(
        self, name: str, token_id: object
    ) -> None:
        """Explicit special-token overrides must be integer vocabulary IDs."""
        config = tiny_config()
        assert config.vocab_size == 64
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        with pytest.raises(ValueError, match=name):
            generate(
                model,
                prompt,
                max_new_tokens=1,
                temperature=0.0,
                **{name: token_id},
            )

    def test_generate_accepts_integer_protocol_values(self) -> None:
        """NumPy integer scalars are accepted through the integer protocol."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        output, _, _ = generate(
            model,
            jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
            max_new_tokens=np.int64(1),
            temperature=0.0,
            bos_token_id=np.int64(config.bos_token_id),
            eos_token_id=np.int64(config.eos_token_id),
        )
        assert output.shape == (1, 4)

    def test_generate_uses_last_valid_token_with_padding(
        self,
    ) -> None:
        """Greedy generation selects logits from the last valid right-padded token."""
        config = tiny_config()
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
        model = MegalodonForCausalLM(tiny_config(), key=jax.random.PRNGKey(0))
        with pytest.raises(Exception, match="contiguous valid prefix.*right padding"):
            generate(
                model,
                jnp.asarray([[0, 0, 1, 2]], dtype=jnp.int32),
                max_new_tokens=1,
                temperature=0.0,
                attention_mask=jnp.asarray([[False, False, True, True]]),
            )

    @pytest.mark.parametrize("max_new_tokens", [1, 2])
    @pytest.mark.parametrize("return_cache", [False, True])
    def test_generate_empty_prompt_matches_explicit_bos(
        self, max_new_tokens: int, return_cache: bool
    ) -> None:
        """Empty prompts behave exactly like an explicit, unpadded BOS prompt."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        empty_prompt = jnp.empty((1, 0), dtype=jnp.int32)
        explicit_bos = jnp.asarray([[config.bos_token_id]], dtype=jnp.int32)

        expected, expected_cache, _ = generate(
            model,
            explicit_bos,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            return_cache=return_cache,
        )
        actual, actual_cache, _ = generate(
            model,
            empty_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            return_cache=return_cache,
        )

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert (actual_cache is not None) is return_cache
        assert (expected_cache is not None) is return_cache
        if return_cache:
            actual_layer = actual_cache.layer_caches[0]
            expected_layer = expected_cache.layer_caches[0]
            assert actual_layer is not None and expected_layer is not None
            assert actual_layer.attn is not None and expected_layer.attn is not None
            assert int(actual_layer.attn.count) == int(expected_layer.attn.count)

    @pytest.mark.parametrize("advanced", [False, True])
    def test_generate_rejects_empty_prompt_continuation(self, advanced: bool) -> None:
        """An empty prompt cannot silently append BOS to an existing timeline."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        if advanced:
            _, cache = model(
                jnp.asarray([[1, 2]], dtype=jnp.int32),
                return_cache=True,
            )
        else:
            cache = init_cache(config)
        assert cache is not None

        with pytest.raises(ValueError, match="empty-prompt continuation"):
            generate(
                model,
                jnp.empty((1, 0), dtype=jnp.int32),
                max_new_tokens=1,
                temperature=0.0,
                cache=cache,
            )

    def test_generate_sampling_requires_key(self) -> None:
        """Ensure sampling requires a PRNG key.

        :return None: None.
        """
        config = tiny_config()
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
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        prompt = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        cache = init_cache(config) if with_cache else None

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
    @pytest.mark.fast
    def test_original_upstream_manifest_is_source_transcribed(self, tmp_path: Path) -> None:
        """Check converter schema and sharding against a hand-authored source manifest."""
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

        # Literal dimensions and partition axes below are transcribed from the constructors in
        # the released moving-average attention, CEMA, TimestepNorm, NFFN,
        # rotary, and output-layer sources. Do not derive this mapping through
        # either converter direction or through JAX parameter leaves.
        manifest: dict[str, tuple[tuple[int, ...], torch.dtype, int | None]] = {
            "embed.weight": ((17, 8), torch.float32, 1),
            "rope.freqs": ((2,), torch.float32, None),
            "output.final_norm.prior_count": ((), torch.int64, None),
            "output.final_norm.prior_mean": ((2,), torch.float32, 0),
            "output.final_norm.prior_logv": ((2,), torch.float32, 0),
            "output.final_norm.weight": ((8,), torch.float32, 0),
            "output.final_norm.bias": ((8,), torch.float32, 0),
            "output.output.weight": ((19, 8), torch.float32, 1),
        }
        for index in range(2):
            ap = f"layers.{index}.mega"
            fp = f"layers.{index}.nffn"
            manifest.update(
                {
                    f"{ap}.timenorm.prior_count": ((), torch.int64, None),
                    f"{ap}.timenorm.prior_mean": ((2,), torch.float32, 0),
                    f"{ap}.timenorm.prior_logv": ((2,), torch.float32, 0),
                    f"{ap}.timenorm.weight": ((8,), torch.float32, 0),
                    f"{ap}.timenorm.bias": ((8,), torch.float32, 0),
                    f"{ap}.cema.alpha": ((8, 2, 1), torch.float32, 0),
                    f"{ap}.cema.delta": ((8, 2, 1), torch.float32, 0),
                    f"{ap}.cema.theta": ((8, 1, 1), torch.float32, 0),
                    f"{ap}.cema.gamma": ((8, 2, 2), torch.float32, 0),
                    f"{ap}.cema.omega": ((8, 1), torch.float32, 0),
                    f"{ap}.rmsnorm.weight": ((8,), torch.float32, None),
                    f"{ap}.wz.weight": ((8, 8), torch.float32, 0),
                    f"{ap}.wz.bias": ((8,), torch.float32, 0),
                    f"{ap}.wv.weight": ((8, 8), torch.float32, 0),
                    f"{ap}.wv.bias": ((8,), torch.float32, 0),
                    f"{ap}.wr.weight": ((8, 8), torch.float32, 0),
                    f"{ap}.wr.bias": ((8,), torch.float32, 0),
                    f"{ap}.wh1.weight": ((8, 8), torch.float32, 0),
                    f"{ap}.wh1.bias": ((8,), torch.float32, 0),
                    f"{ap}.wh2.weight": ((8, 8), torch.float32, 1),
                    f"{ap}.gamma": ((2, 8), torch.float32, 1),
                    f"{ap}.beta": ((2, 8), torch.float32, 1),
                    f"{fp}.norm.weight": ((8,), torch.float32, None),
                    f"{fp}.norm.bias": ((8,), torch.float32, None),
                    f"{fp}.fc1.weight": ((12, 8), torch.float32, 0),
                    f"{fp}.fc2.weight": ((8, 12), torch.float32, 1),
                    f"{fp}.fc3.weight": ((12, 8), torch.float32, 0),
                    f"{fp}.alpha": ((8,), torch.float32, None),
                }
            )

        exported = export_upstream_state_dict(model)
        assert set(exported) == set(manifest)
        for name, (shape, dtype, _) in manifest.items():
            assert tuple(exported[name].shape) == shape, name
            assert exported[name].dtype == dtype, name

        generator = torch.Generator().manual_seed(1729)
        source_state: dict[str, torch.Tensor] = {}
        for name, (shape, dtype, _) in manifest.items():
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

        shards: list[dict[str, torch.Tensor]] = [{}, {}]
        for name, value in source_state.items():
            axis = manifest[name][2]
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
        consolidated = load_upstream_checkpoint(
            tmp_path,
            config,
            key=jax.random.PRNGKey(2),
        )
        consolidated_state = export_upstream_state_dict(consolidated)
        for name in manifest:
            assert torch.equal(consolidated_state[name], source_state[name]), name

    @pytest.mark.fast
    def test_native_v2_roundtrip_is_exact(self, tmp_path: Path) -> None:
        """Native save/reload preserves every tensor and model output."""
        config = tiny_config()
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

    @pytest.mark.fast
    def test_native_bf16_storage_roundtrip_restores_policy(self, tmp_path: Path) -> None:
        """BF16 ordinary storage and softmax selection reload without caller flags."""
        from safetensors import safe_open

        config = replace(
            tiny_config(),
            param_dtype=jnp.bfloat16,
            compute_dtype=jnp.bfloat16,
            attention_softmax_dtype=jnp.bfloat16,
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        path = tmp_path / "model-bf16.safetensors"

        save_checkpoint(model, path)
        loaded = load_checkpoint(path, key=jax.random.PRNGKey(1))

        assert loaded.config == config
        assert loaded.model.embed.weight.dtype == jnp.bfloat16
        assert loaded.model.layers[0].attn.cema.alpha.dtype == jnp.float32
        with safe_open(str(path), framework="flax") as handle:
            assert handle.metadata()["dtype_policy"] == BF16_DTYPE_POLICY
        expected = model_state_dict(model)
        actual = model_state_dict(loaded)
        for name in expected:
            assert actual[name].dtype == expected[name].dtype, name
            np.testing.assert_array_equal(np.asarray(actual[name]), np.asarray(expected[name]))

    def test_native_roundtrip_normalizes_numpy_config_scalars(self, tmp_path: Path) -> None:
        """NumPy config scalars become Python metadata before model construction."""
        config = replace(
            tiny_config(),
            vocab_size=np.int64(64),
            dropout=np.float32(0.1),
        )
        assert type(config.vocab_size) is int
        assert type(config.dropout) is float

        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        path = tmp_path / "numpy-scalars.safetensors"

        save_checkpoint(model, path)
        loaded = load_checkpoint(path, key=jax.random.PRNGKey(1))

        assert type(loaded.config.vocab_size) is int
        assert loaded.config.vocab_size == int(config.vocab_size)
        assert type(loaded.config.dropout) is float
        assert loaded.config.dropout == float(config.dropout)

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

        config = tiny_config()
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
        config = tiny_config()
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
    @pytest.mark.parametrize("norm_affine", [True, False])
    def test_original_upstream_roundtrip_and_keyspace(
        self,
        monkeypatch: pytest.MonkeyPatch,
        norm_affine: bool,
    ) -> None:
        """Exact released keys round-trip without transposes or schema aliases."""
        pytest.importorskip("torch")
        config = tiny_config(norm_affine=norm_affine)
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
        assert ("layers.0.mega.rmsnorm.weight" in state) is norm_affine
        assert ("layers.0.nffn.norm.weight" in state) is norm_affine

        def fail_export(*args: object, **kwargs: object) -> None:
            pytest.fail("strict loading must enumerate keys without exporting tensors")

        monkeypatch.setattr(
            "megalodon_jax.convert.export_upstream_state_dict",
            fail_export,
        )

        loaded = load_upstream_state_dict(
            MegalodonForCausalLM(config, key=jax.random.PRNGKey(1)),
            state,
        )
        tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
        expected_logits, _ = model(tokens)
        actual_logits, _ = loaded(tokens)
        np.testing.assert_array_equal(np.asarray(actual_logits), np.asarray(expected_logits))

    @pytest.mark.torch_ref
    def test_original_upstream_roundtrip_preserves_bf16_ordinary_storage(self) -> None:
        """Upstream transport follows BF16 ordinary and FP32-sensitive storage."""
        torch = pytest.importorskip("torch")
        config = replace(
            tiny_config(),
            param_dtype=jnp.bfloat16,
            compute_dtype=jnp.bfloat16,
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))

        state = export_upstream_state_dict(model)
        assert state["embed.weight"].dtype == torch.bfloat16
        assert state["layers.0.mega.wz.weight"].dtype == torch.bfloat16
        assert state["output.output.weight"].dtype == torch.bfloat16
        assert state["layers.0.mega.cema.alpha"].dtype == torch.float32
        assert state["output.final_norm.weight"].dtype == torch.float32

        loaded = load_upstream_state_dict(
            MegalodonForCausalLM(config, key=jax.random.PRNGKey(1)),
            state,
        )
        expected = model_state_dict(model)
        actual = model_state_dict(loaded)
        for name in expected:
            assert actual[name].dtype == expected[name].dtype, name
            np.testing.assert_array_equal(np.asarray(actual[name]), np.asarray(expected[name]))

    @pytest.mark.torch_ref
    def test_original_upstream_strict_failure(self) -> None:
        """Missing, unexpected, and Hugging-Face-shaped keys fail closed."""
        pytest.importorskip("torch")
        config = tiny_config()
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
        config = replace(tiny_config(), share_emb=True)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        state = export_upstream_state_dict(model)
        assert state["output.output.weight"].data_ptr() != state["embed.weight"].data_ptr()
        assert state["output.output.weight"].equal(state["embed.weight"])

        value = state["output.output.weight"][0, 0]
        state["output.output.weight"][0, 0] = torch.nextafter(value, value + 1.0)
        with pytest.raises(ValueError, match="bit-identical"):
            load_upstream_state_dict(model, state)

    @pytest.mark.torch_ref
    @pytest.mark.parametrize(
        ("metadata", "message"),
        [
            ("{", "invalid consolidate_config.json"),
            ("[]", "must contain a JSON object"),
            ("{}", "missing required model_parallel_size"),
            ('{"model_parallel_size": true}', "must be a positive integer"),
            ('{"model_parallel_size": 1.5}', "must be a positive integer"),
            ('{"model_parallel_size": 0}', "must be a positive integer"),
        ],
    )
    def test_consolidation_metadata_errors_are_descriptive(
        self,
        tmp_path: Path,
        metadata: str,
        message: str,
    ) -> None:
        """Malformed consolidation metadata fails before shard discovery."""
        pytest.importorskip("torch")
        (tmp_path / "consolidate_config.json").write_text(metadata, encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            load_upstream_checkpoint(
                tmp_path,
                tiny_config(),
                key=jax.random.PRNGKey(0),
            )

    @pytest.mark.parametrize("compute_dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.fast
    def test_cache_roundtrip_and_config_binding(
        self,
        tmp_path: Path,
        compute_dtype: jnp.dtype,
    ) -> None:
        """Serialized continuation state resumes exactly and rejects other configs."""
        config = replace(tiny_config(), compute_dtype=compute_dtype)
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

    def test_save_rejects_partial_nonzero_cache(self, tmp_path: Path) -> None:
        """Persistence cannot legitimize a destructive partial continuation."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        _, cache = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32), return_cache=True)
        assert cache is not None
        first = cache.layer_caches[0]
        assert first is not None and first.attn is not None
        malformed = replace(
            cache,
            layer_caches=(replace(first, ema=None), *cache.layer_caches[1:]),
        )

        with pytest.raises(ValueError, match=CACHE_INVARIANT_MESSAGE):
            save_inference_cache(malformed, tmp_path / "partial.safetensors", config)

        nonfinite_attn = replace(
            first.attn,
            k=first.attn.k.at[0, 0, 0, 0].set(jnp.nan),
        )
        nonfinite = replace(
            cache,
            layer_caches=(replace(first, attn=nonfinite_attn), *cache.layer_caches[1:]),
        )
        with pytest.raises(ValueError, match=CACHE_INVARIANT_MESSAGE):
            save_inference_cache(nonfinite, tmp_path / "nonfinite.safetensors", config)

        partial_zero = ModelCache(
            layer_caches=(
                LayerCache(
                    norm=NormState(
                        count=jnp.zeros((1,), dtype=jnp.int32),
                        mean=jnp.zeros((1, config.norm_num_groups), dtype=jnp.float32),
                        var=jnp.ones((1, config.norm_num_groups), dtype=jnp.float32),
                    ),
                    position=jnp.asarray(0, dtype=jnp.int32),
                ),
                *([None] * (config.num_layers - 1)),
            ),
            final_norm=None,
        )
        with pytest.raises(ValueError, match=CACHE_INVARIANT_MESSAGE):
            save_inference_cache(partial_zero, tmp_path / "partial-zero.safetensors", config)

    def test_pristine_cache_save_load_and_execute(self, tmp_path: Path) -> None:
        """Every persisted pristine cache remains executable after loading."""
        config = tiny_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        valid = tmp_path / "sparse-zero.safetensors"
        save_inference_cache(init_cache(config), valid, config)
        restored = load_inference_cache(valid, config)

        logits, updated = model(
            jnp.asarray([[1]], dtype=jnp.int32),
            cache=restored,
            return_cache=True,
        )
        assert bool(jnp.all(jnp.isfinite(logits)))
        assert updated is not None

    @pytest.mark.fast
    def test_cache_schema_and_position_invariants_fail_closed(self, tmp_path: Path) -> None:
        """Presence metadata and redundant attention positions must remain self-consistent."""
        from safetensors import safe_open
        from safetensors.flax import load_file, save_file

        config = tiny_config()
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

        partial_presence = dict(metadata)
        partial_presence["present_json"] = json.dumps(
            [item for item in present if item not in {"layers.0.attn", "layers.0.ema"}]
        )
        partial_path = tmp_path / "partial-presence.safetensors"
        save_file(tensors, str(partial_path), metadata=partial_presence)
        with pytest.raises(ValueError, match=CACHE_INVARIANT_MESSAGE):
            load_inference_cache(partial_path, config)

    def test_atomic_writer_is_unique_durable_and_cleans_failures(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Atomic persistence avoids shared temp names and removes failed writes."""
        import megalodon_jax.checkpoint as checkpoint_module

        model = MegalodonForCausalLM(tiny_config(), key=jax.random.PRNGKey(0))
        destination = tmp_path / "model.safetensors"
        legacy_temporary = destination.with_suffix(".safetensors.tmp")
        legacy_temporary.write_bytes(b"unrelated writer")
        real_fsync = checkpoint_module.os.fsync
        fsync_calls: list[int] = []

        def record_fsync(file_descriptor: int) -> None:
            fsync_calls.append(file_descriptor)
            real_fsync(file_descriptor)

        monkeypatch.setattr(checkpoint_module.os, "fsync", record_fsync)

        save_checkpoint(model, destination)

        assert destination.is_file()
        assert legacy_temporary.read_bytes() == b"unrelated writer"
        assert len(fsync_calls) == 2  # temporary contents, then containing directory
        original = destination.read_bytes()

        def fail_after_partial_write(
            tensors: dict[str, object],
            path: str,
            *,
            metadata: dict[str, str],
        ) -> None:
            del tensors, metadata
            Path(path).write_bytes(b"partial")
            raise RuntimeError("simulated persistence failure")

        monkeypatch.setattr(checkpoint_module, "save_file", fail_after_partial_write)
        with pytest.raises(RuntimeError, match="simulated persistence failure"):
            save_checkpoint(model, destination)

        assert destination.read_bytes() == original
        assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []

    def test_missing_persistence_files_raise_file_not_found(self, tmp_path: Path) -> None:
        """Model and cache readers expose the same missing-file exception contract."""
        missing = tmp_path / "missing.safetensors"
        with pytest.raises(FileNotFoundError):
            load_checkpoint(missing, key=jax.random.PRNGKey(0))
        with pytest.raises(FileNotFoundError):
            load_inference_cache(missing, tiny_config())

    def test_ambiguous_legacy_apis_refuse(self, tmp_path: Path) -> None:
        """Historical schema-guessing entry points provide migration guidance."""
        model = MegalodonForCausalLM(tiny_config(), key=jax.random.PRNGKey(0))
        with pytest.raises(RuntimeError, match="export_upstream_state_dict"):
            convert_jax_to_torch(model)
        with pytest.raises(RuntimeError, match="load_upstream_state_dict"):
            load_weights_from_torch({})
        with pytest.raises(RuntimeError, match="save_checkpoint"):
            save_safetensors(model, tmp_path / "model.safetensors")
        with pytest.raises(RuntimeError, match="load_checkpoint"):
            load_from_pretrained(tmp_path / "model.safetensors")
