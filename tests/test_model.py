"""Phase 4 Model Assembly tests - MegalodonBlock, MegalodonModel, MegalodonForCausalLM."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax import MegalodonBlock, MegalodonConfig, MegalodonForCausalLM, MegalodonModel
from megalodon_jax.cache import CACHE_INVARIANT_MESSAGE, cache_invariant_violation
from megalodon_jax.config import InitMode
from megalodon_jax.inference import init_cache
from megalodon_jax.layers import TimestepNorm
from megalodon_jax.precision import audit_sensitive_param_dtypes, ensure_sensitive_param_dtype
from megalodon_jax.types import EMAState, LayerCache, ModelCache, NormState
from megalodon_jax.utils import get_initializer
from tests.factories import floating_to_bf16, tiny_config


def small_config(**overrides: Any) -> MegalodonConfig:
    """Create a small config for fast testing.

    :return MegalodonConfig: Minimal model configuration for tests.
    """
    return tiny_config(**{"vocab_size": 256, "num_layers": 2, "chunk_size": 16, **overrides})


# -----------------------------------------------------------------------------
# MegalodonBlock Tests
# -----------------------------------------------------------------------------


class TestMegalodonBlock:
    """Tests for MegalodonBlock decoder layer."""

    def test_forward_shapes(self, random_seed: int) -> None:
        """Test that MegalodonBlock produces correct output shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 32

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        block = MegalodonBlock(config, layer_id=0, key=k1)
        x = jax.random.normal(k2, (batch, seq, config.model_dim))

        out, cache = block(x, return_cache=True)

        assert out.shape == (batch, seq, config.model_dim)
        assert cache is not None
        assert cache.attn is not None
        assert cache.norm is not None
        assert cache.ema is not None

    def test_forward_no_cache(self, random_seed: int) -> None:
        """Test forward pass without returning cache.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 32

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        block = MegalodonBlock(config, layer_id=0, key=k1)
        x = jax.random.normal(k2, (batch, seq, config.model_dim))

        out, cache = block(x, return_cache=False)

        assert out.shape == (batch, seq, config.model_dim)
        assert cache is None

    @pytest.mark.fast
    def test_direct_cache_requires_complete_aligned_state(self, random_seed: int) -> None:
        """Direct block calls cannot mix reset and continued layer components."""
        config = small_config()
        block = MegalodonBlock(config, layer_id=0, key=jax.random.PRNGKey(random_seed))
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 3, config.model_dim))
        _, cache = block(x, return_cache=True)
        assert cache is not None and cache.attn is not None and cache.norm is not None

        malformed = (
            replace(cache, attn=None),
            replace(cache, norm=None),
            replace(cache, ema=None),
            replace(cache, position=cache.position + 1),
            replace(cache, norm=replace(cache.norm, count=cache.norm.count + 1)),
            replace(
                cache,
                norm=replace(cache.norm, mean=cache.norm.mean.at[0, 0].set(jnp.nan)),
            ),
        )
        for invalid in malformed:
            with pytest.raises(Exception, match=CACHE_INVARIANT_MESSAGE):
                jax.block_until_ready(block(x[:, :1], cache=invalid, return_cache=True))

        compiled = eqx.filter_jit(
            lambda values, state: block(values, cache=state, return_cache=True)
        )
        with pytest.raises(Exception, match=CACHE_INVARIANT_MESSAGE):
            jax.block_until_ready(compiled(x[:, :1], malformed[0]))

        expected, _ = block(x, return_cache=True)
        actual, _ = block(x, cache=LayerCache(), return_cache=True)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    @pytest.mark.fast
    def test_direct_cache_requires_exact_array_schema(self, random_seed: int) -> None:
        """Direct block caches fail before compute when shapes or dtypes drift."""
        config = replace(small_config(), compute_dtype=jnp.bfloat16)
        block = MegalodonBlock(config, layer_id=0, key=jax.random.PRNGKey(random_seed))
        x = jnp.zeros((2, 3, config.model_dim), dtype=jnp.bfloat16)
        _, cache = block(x, return_cache=True)
        assert cache is not None and cache.attn is not None
        assert cache.norm is not None and cache.ema is not None

        malformed = (
            replace(cache, attn=replace(cache.attn, k=cache.attn.k.astype(jnp.float32))),
            replace(cache, attn=replace(cache.attn, v=cache.attn.v[:, :-1])),
            replace(cache, norm=replace(cache.norm, mean=cache.norm.mean[:, :-1])),
            replace(cache, ema=replace(cache.ema, h=cache.ema.h[:, :, :-1])),
        )
        for invalid in malformed:
            with pytest.raises(ValueError, match="must have (shape|dtype)"):
                block(x[:, :1], cache=invalid, return_cache=True)

    @pytest.mark.fast
    def test_direct_cache_is_inference_only(self, random_seed: int) -> None:
        """Direct block calls enforce the model's inference-only cache contract."""
        config = small_config()
        block = MegalodonBlock(config, layer_id=0, key=jax.random.PRNGKey(random_seed))
        x = jnp.zeros((1, 1, config.model_dim), dtype=jnp.float32)

        with pytest.raises(ValueError, match="inference-only"):
            block(
                x,
                return_cache=True,
                deterministic=False,
                key=jax.random.PRNGKey(1),
            )

    def test_streaming_with_cache(self, random_seed: int) -> None:
        """Test that streaming with cache produces consistent outputs.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 1, 32

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        block = MegalodonBlock(config, layer_id=0, key=k1)
        x = jax.random.normal(k2, (batch, seq, config.model_dim))

        # Process full sequence and an uneven partition spanning chunk boundaries.
        out_full, cache_full = block(x, return_cache=True)
        out_noncached, _ = block(x, return_cache=False)
        pieces = []
        cache_streamed = None
        start = 0
        for width in (7, 10, 5, 10):
            stop = start + width
            out, cache_streamed = block(
                x[:, start:stop],
                cache=cache_streamed,
                return_cache=True,
            )
            pieces.append(out)
            start = stop
        out_streamed = jnp.concatenate(pieces, axis=1)

        np.testing.assert_allclose(
            np.asarray(out_full),
            np.asarray(out_noncached),
            rtol=2e-5,
            atol=2e-6,
            err_msg="Pristine cached prefill must match vectorized noncached output",
        )
        np.testing.assert_allclose(
            np.array(out_full),
            np.array(out_streamed),
            rtol=2e-3,
            atol=2e-5,
            err_msg="Streaming output differs from batch output",
        )
        assert cache_full is not None and cache_streamed is not None
        for expected, actual in zip(
            jax.tree.leaves(cache_full),
            jax.tree.leaves(cache_streamed),
            strict=True,
        ):
            if jnp.issubdtype(expected.dtype, jnp.integer):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            else:
                np.testing.assert_allclose(
                    np.asarray(actual),
                    np.asarray(expected),
                    rtol=2e-4,
                    atol=2e-5,
                )

    def test_different_layer_ids(self, random_seed: int) -> None:
        """Test that different layer IDs produce different rescaling.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=4, rescale_nffn=True)

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        block0 = MegalodonBlock(config, layer_id=0, key=k1)
        block3 = MegalodonBlock(config, layer_id=3, key=k2)

        assert block0.ffn.alpha is not None
        assert block3.ffn.alpha is not None
        # alpha = 0.1 * (0.5 ** layer_id)
        np.testing.assert_allclose(block0.ffn.alpha, 0.1 * (0.5**0), rtol=1e-6)
        np.testing.assert_allclose(block3.ffn.alpha, 0.1 * (0.5**3), rtol=1e-6)


# -----------------------------------------------------------------------------
# MegalodonModel Tests
# -----------------------------------------------------------------------------


class TestMegalodonModel:
    """Tests for MegalodonModel decoder stack."""

    def test_forward_shapes(self, random_seed: int) -> None:
        """Test that MegalodonModel produces correct output shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 32

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        hidden, cache = model(input_ids, return_cache=True)

        assert hidden.shape == (batch, seq, config.model_dim)
        assert cache is not None
        assert len(cache.layer_caches) == config.num_layers
        assert cache.final_norm is not None

    def test_embedding_scale(self, random_seed: int) -> None:
        """Test that embedding scaling works correctly.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config_scaled = MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
            scale_emb=True,
        )
        config_unscaled = MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
            scale_emb=False,
        )

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        model_scaled = MegalodonModel(config_scaled, key=k1)
        model_unscaled = MegalodonModel(config_unscaled, key=k2)

        assert model_scaled.scale == float(jnp.sqrt(64))
        assert model_unscaled.scale == 1.0

    def test_cache_threading(self, random_seed: int) -> None:
        """Test that cache is properly threaded through layers.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 1, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)

        # First forward - no cache
        _, cache1 = model(input_ids, return_cache=True)

        # Second forward - with cache
        next_ids = jax.random.randint(key, (batch, 1), minval=0, maxval=config.vocab_size)
        _, cache2 = model(next_ids, cache=cache1, return_cache=True)

        # Cache should be updated
        assert cache2 is not None
        # Position should have advanced
        for layer_cache in cache2.layer_caches:
            if layer_cache is not None and layer_cache.attn is not None:
                assert layer_cache.attn.count > 0

    def test_attention_mask(self, random_seed: int) -> None:
        """Test that attention mask is properly applied.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)

        # Create mask with some padding
        mask = jnp.ones((batch, seq), dtype=bool)
        mask = mask.at[0, -4:].set(False)  # Mask last 4 tokens in first batch

        hidden_masked, _ = model(input_ids, attention_mask=mask, return_cache=False)
        hidden_unmasked, _ = model(input_ids, return_cache=False)

        # Outputs should differ where mask is applied
        assert hidden_masked.shape == hidden_unmasked.shape
        # The masked version should differ at masked positions
        assert not jnp.allclose(hidden_masked[0, -4:], hidden_unmasked[0, -4:])


# -----------------------------------------------------------------------------
# MegalodonForCausalLM Tests
# -----------------------------------------------------------------------------


class TestMegalodonForCausalLM:
    """Tests for MegalodonForCausalLM."""

    @pytest.mark.fast
    def test_bf16_cached_logits_stay_within_compute_envelope(self) -> None:
        """Live cache outputs may change BF16 GEMM association, not semantics."""
        config = small_config(compute_dtype=jnp.bfloat16)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(91))
        tokens = jnp.asarray([[1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32)
        next_token = jnp.asarray([[8]], dtype=jnp.int32)
        noncached = eqx.filter_jit(lambda candidate, values: candidate(values, return_cache=False))
        cached = eqx.filter_jit(lambda candidate, values: candidate(values, return_cache=True))
        continue_cached = eqx.filter_jit(
            lambda candidate, values, state: candidate(values, cache=state, return_cache=True)
        )

        expected_prefill, _ = noncached(model, tokens)
        actual_prefill, cache = cached(model, tokens)
        expected_decode, _ = noncached(model, jnp.concatenate((tokens, next_token), axis=1))
        actual_decode, _ = continue_cached(model, next_token, cache)

        np.testing.assert_allclose(
            np.asarray(actual_prefill),
            np.asarray(expected_prefill),
            rtol=2e-2,
            atol=8e-2,
        )
        np.testing.assert_allclose(
            np.asarray(actual_decode),
            np.asarray(expected_decode[:, -1:]),
            rtol=2e-2,
            atol=8e-2,
        )

    def test_forward_shapes(self, random_seed: int) -> None:
        """Test that MegalodonForCausalLM produces correct logit shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 32

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        logits, cache = model(input_ids, return_cache=True)

        assert logits.shape == (batch, seq, config.vocab_size)
        assert cache is not None

    def test_weight_tying(self, random_seed: int) -> None:
        """Test that LM head is tied to input embeddings.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = replace(small_config(), share_emb=True)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # The LM head should use embed weights transposed
        # Verify by checking that logits = hidden @ embed.weight.T
        batch, seq = 1, 8
        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)

        logits, _ = model(input_ids, return_cache=False)

        # Get hidden states manually
        hidden, _ = model.model(input_ids, return_cache=False)

        # Compute logits manually using weight tying
        manual_logits = jnp.matmul(
            hidden,
            model.model.embed.weight.T,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        np.testing.assert_allclose(
            np.array(logits),
            np.array(manual_logits),
            rtol=1e-5,
            atol=1e-6,
            err_msg="Weight tying is not working correctly",
        )

    def test_compute_loss(self, random_seed: int) -> None:
        """Test that loss computation works correctly.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        labels = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)

        loss = model.compute_loss(input_ids, labels)

        # Loss should be a scalar
        assert loss.shape == ()
        # Loss should be positive and finite
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_compute_loss_validates_label_schema(self, random_seed: int) -> None:
        """Loss labels must match inputs and use discrete token IDs."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(random_seed))
        input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="same shape as input_ids"):
            model.compute_loss(input_ids, jnp.asarray([[1, 2]], dtype=jnp.int32))
        with pytest.raises(TypeError, match="labels must have integer dtype"):
            model.compute_loss(input_ids, input_ids.astype(jnp.float32))
        with pytest.raises(ValueError, match="reduction must be"):
            model.compute_loss(input_ids, input_ids, reduction="median")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# JIT Compilation Tests
# -----------------------------------------------------------------------------


class TestJIT:
    """Tests for JIT compilation and tracing behavior."""

    def test_model_jit_compilation(self, random_seed: int) -> None:
        """Test that model compiles without errors.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        @eqx.filter_jit
        def forward(
            model: MegalodonForCausalLM,
            input_ids: jnp.ndarray,
        ) -> tuple[jnp.ndarray, Any]:
            """Run a JIT-compiled forward pass.

            :param MegalodonForCausalLM model: Model under test.
            :param jnp.ndarray input_ids: Token IDs.
            :return tuple[jnp.ndarray, Any]: Logits and cache.
            """
            return model(input_ids, return_cache=False)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)

        # First call compiles
        logits1, _ = forward(model, input_ids)

        # Second call uses cached compilation
        logits2, _ = forward(model, input_ids)

        np.testing.assert_array_equal(
            np.array(logits1),
            np.array(logits2),
            err_msg="JIT compilation produced different results",
        )

    def test_no_recompilation_on_cache_update(self, random_seed: int) -> None:
        """Test that streaming with cache doesn't cause recompilation.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch = 1

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        @eqx.filter_jit
        def step(
            model: MegalodonForCausalLM,
            input_ids: jnp.ndarray,
            cache: Any,
        ) -> tuple[jnp.ndarray, Any]:
            """Run a streaming step with cache.

            :param MegalodonForCausalLM model: Model under test.
            :param jnp.ndarray input_ids: Token IDs.
            :param Any cache: Optional cache.
            :return tuple[jnp.ndarray, Any]: Logits and updated cache.
            """
            return model(input_ids, cache=cache, return_cache=True)

        # Initial prompt
        prompt = jax.random.randint(key, (batch, 8), minval=0, maxval=config.vocab_size)

        # First step - compiles
        logits1, cache = step(model, prompt, None)

        # Subsequent steps should reuse compilation
        for _ in range(3):
            next_token = jnp.argmax(logits1[:, -1:], axis=-1)
            logits1, cache = step(model, next_token, cache)

        # Should complete without error (recompilation would cause shape mismatch)
        assert logits1.shape == (batch, 1, config.vocab_size)


# -----------------------------------------------------------------------------
# Gradient Flow Tests
# -----------------------------------------------------------------------------


class TestGradients:
    """Tests for gradient computation and flow."""

    @pytest.mark.fast
    def test_checkpointed_loss_and_gradients_match_plain(self, random_seed: int) -> None:
        """Gradient checkpointing preserves stochastic training loss and gradients."""
        config = MegalodonConfig(
            vocab_size=17,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            z_dim=8,
            value_dim=8,
            ffn_hidden_dim=12,
            cema_ndim=2,
            chunk_size=4,
            norm_num_groups=2,
            dropout=0.1,
            attention_dropout=0.1,
            hidden_dropout=0.1,
        )
        model_key = jax.random.PRNGKey(random_seed)
        plain = MegalodonForCausalLM(config, key=model_key)
        checkpointed = MegalodonForCausalLM(
            replace(config, use_checkpoint=True),
            key=model_key,
        )
        tokens = jnp.asarray([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32)
        dropout_key = jax.random.PRNGKey(random_seed + 1)

        def loss_fn(candidate: MegalodonForCausalLM) -> jax.Array:
            return candidate.compute_loss(
                tokens,
                tokens,
                deterministic=False,
                key=dropout_key,
            )

        plain_loss, plain_grads = eqx.filter_value_and_grad(loss_fn)(plain)
        checkpointed_loss, checkpointed_grads = eqx.filter_value_and_grad(loss_fn)(checkpointed)

        np.testing.assert_allclose(checkpointed_loss, plain_loss, rtol=1e-6, atol=1e-6)
        plain_leaves = [
            leaf for leaf in jax.tree_util.tree_leaves(plain_grads) if eqx.is_inexact_array(leaf)
        ]
        checkpointed_leaves = [
            leaf
            for leaf in jax.tree_util.tree_leaves(checkpointed_grads)
            if eqx.is_inexact_array(leaf)
        ]
        assert plain_leaves
        for checkpointed_grad, plain_grad in zip(
            checkpointed_leaves,
            plain_leaves,
            strict=True,
        ):
            np.testing.assert_allclose(checkpointed_grad, plain_grad, rtol=1e-6, atol=1e-6)

    def test_gradient_flow_all_params(self, random_seed: int) -> None:
        """Test that gradients flow to all trainable parameters.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        labels = input_ids  # Self-supervision

        def loss_fn(model: MegalodonForCausalLM) -> jnp.ndarray:
            """Compute a scalar loss for gradient checks.

            :param MegalodonForCausalLM model: Model under test.
            :return jnp.ndarray: Scalar loss value.
            """
            return model.compute_loss(input_ids, labels)

        # Compute gradients
        grads = eqx.filter_grad(loss_fn)(model)

        # Apply check to all leaves
        grad_leaves = jax.tree_util.tree_leaves(
            eqx.filter(grads, eqx.is_array), is_leaf=eqx.is_array
        )
        assert len(grad_leaves) > 0, "No gradients computed"

        for leaf in grad_leaves:
            assert jnp.all(jnp.isfinite(leaf)), "Gradient contains NaN or Inf"

        embed_grad = grads.model.embed.weight
        assert embed_grad is not None
        assert jnp.any(embed_grad != 0)

    def test_gradient_dtype_matches_param_dtype_bf16_compute(self, random_seed: int) -> None:
        """Test that gradients keep param_dtype when compute dtype is bf16.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = replace(small_config(), compute_dtype=jnp.bfloat16)
        batch, seq = 2, 8

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        labels = input_ids

        def loss_fn(model: MegalodonForCausalLM) -> jnp.ndarray:
            """Compute a scalar loss for gradient checks.

            :param MegalodonForCausalLM model: Model under test.
            :return jnp.ndarray: Scalar loss value.
            """
            return model.compute_loss(input_ids, labels)

        grads = eqx.filter_grad(loss_fn)(model)
        embed_grad = grads.model.embed.weight
        assert embed_grad is not None
        assert embed_grad.dtype == jnp.float32

    def test_gradient_dtype_matches_bf16_param_storage(self, random_seed: int) -> None:
        """Ordinary gradients are BF16 while sensitive gradients remain FP32."""
        config = replace(
            small_config(num_layers=1),
            param_dtype=jnp.bfloat16,
            compute_dtype=jnp.bfloat16,
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(random_seed))
        input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

        grads = eqx.filter_grad(lambda candidate: candidate.compute_loss(input_ids, input_ids))(
            model
        )

        assert grads.model.embed.weight.dtype == jnp.bfloat16
        assert grads.model.layers[0].attn.cema.alpha.dtype == jnp.float32


# -----------------------------------------------------------------------------
# ModelCache Tests
# -----------------------------------------------------------------------------


class TestModelCache:
    """Tests for ModelCache pytree registration and behavior."""

    def test_cache_is_pytree(self, random_seed: int) -> None:
        """Test that ModelCache is properly registered as a pytree.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 1, 8

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        _, cache = model(input_ids, return_cache=True)

        assert isinstance(cache.layer_caches, tuple)
        # Cache should be a valid pytree
        leaves = jax.tree_util.tree_leaves(cache)
        assert len(leaves) > 0

        # Should be able to map over cache
        def double(x: Any) -> Any:
            """Double numeric cache leaves for pytree mapping.

            :param Any x: Leaf value to transform.
            :return Any: Doubled numeric leaf or original value.
            """
            if isinstance(x, jnp.ndarray):
                return x * 2
            return x

        doubled = jax.tree_util.tree_map(double, cache)
        assert doubled is not None

    def test_returned_cache_stops_parameter_gradients(self, random_seed: int) -> None:
        """A cache-only objective cannot backpropagate through its recorded history."""
        config = MegalodonConfig(
            vocab_size=17,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            z_dim=8,
            value_dim=8,
            ffn_hidden_dim=12,
            cema_ndim=2,
            chunk_size=4,
            norm_num_groups=2,
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(random_seed))
        tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)

        def cache_objective(candidate: MegalodonForCausalLM) -> jax.Array:
            _, cache = candidate(tokens, return_cache=True)
            assert cache is not None
            total = jnp.asarray(0.0, dtype=jnp.float32)
            for leaf in jax.tree_util.tree_leaves(cache):
                if eqx.is_inexact_array(leaf):
                    total = total + jnp.sum(jnp.real(leaf).astype(jnp.float32))
            return total

        gradients = eqx.filter_grad(cache_objective)(model)
        gradient_leaves = [
            leaf for leaf in jax.tree_util.tree_leaves(gradients) if eqx.is_inexact_array(leaf)
        ]
        assert gradient_leaves
        assert max(float(jnp.max(jnp.abs(leaf))) for leaf in gradient_leaves) == 0.0

    def test_nonzero_cache_requires_complete_state(self, random_seed: int) -> None:
        """Every component must continue together once the timeline advances."""
        config = small_config()
        model = MegalodonModel(config, key=jax.random.PRNGKey(random_seed))
        tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
        _, cache = model(tokens, return_cache=True)
        assert cache is not None
        first = cache.layer_caches[0]
        assert first is not None

        malformed: dict[str, ModelCache] = {
            "layer": replace(cache, layer_caches=(None, *cache.layer_caches[1:])),
            "attention": replace(
                cache,
                layer_caches=(replace(first, attn=None), *cache.layer_caches[1:]),
            ),
            "norm": replace(
                cache,
                layer_caches=(replace(first, norm=None), *cache.layer_caches[1:]),
            ),
            "ema": replace(
                cache,
                layer_caches=(replace(first, ema=None), *cache.layer_caches[1:]),
            ),
            "final_norm": replace(cache, final_norm=None),
        }
        next_token = jnp.asarray([[4]], dtype=jnp.int32)
        for invalid in malformed.values():
            with pytest.raises(Exception, match="sparse zero-history initializer"):
                jax.block_until_ready(model(next_token, cache=invalid, return_cache=True))

        compiled = eqx.filter_jit(
            lambda values, state: model(values, cache=state, return_cache=True)
        )
        with pytest.raises(Exception, match="sparse zero-history initializer"):
            jax.block_until_ready(compiled(next_token, malformed["attention"]))
        with pytest.raises(ValueError, match="non-empty input_ids"):
            jax.block_until_ready(
                model(
                    jnp.empty((1, 0), dtype=jnp.int32),
                    cache=cache,
                    return_cache=True,
                )
            )

    def test_nonzero_cache_requires_one_timeline(self, random_seed: int) -> None:
        """Layer, component, and final-normalization counters cannot diverge."""
        config = small_config()
        model = MegalodonModel(config, key=jax.random.PRNGKey(random_seed))
        tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
        _, cache = model(tokens, return_cache=True)
        assert cache is not None and cache.final_norm is not None
        first, second = cache.layer_caches
        assert first is not None and second is not None
        assert second.attn is not None and first.norm is not None

        advanced_attn = replace(second.attn, count=second.attn.count + 1)
        second_advanced = replace(
            second,
            position=second.position + 1,
            attn=advanced_attn,
        )
        norm_advanced = replace(first.norm, count=first.norm.count + 1)
        malformed = (
            replace(cache, layer_caches=(first, second_advanced)),
            replace(cache, layer_caches=(replace(first, norm=norm_advanced), second)),
            replace(
                cache,
                final_norm=replace(cache.final_norm, count=cache.final_norm.count + 1),
            ),
        )
        for invalid in malformed:
            with pytest.raises(Exception, match=CACHE_INVARIANT_MESSAGE):
                jax.block_until_ready(model(jnp.asarray([[4]], dtype=jnp.int32), cache=invalid))

    @pytest.mark.fast
    def test_model_cache_rejects_invalid_compact_state_values(self, random_seed: int) -> None:
        """Model entry always validates normalization and EMA state payloads."""
        config = small_config()
        model = MegalodonModel(config, key=jax.random.PRNGKey(random_seed))
        _, cache = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32), return_cache=True)
        assert cache is not None and cache.final_norm is not None
        first = cache.layer_caches[0]
        assert first is not None and first.norm is not None and first.ema is not None

        invalid_layer_norms = (
            replace(first.norm, mean=first.norm.mean.at[0, 0].set(jnp.nan)),
            replace(first.norm, var=first.norm.var.at[0, 0].set(jnp.nan)),
            replace(first.norm, var=first.norm.var.at[0, 0].set(-1.0)),
        )
        malformed = [
            replace(
                cache,
                layer_caches=(replace(first, norm=state), *cache.layer_caches[1:]),
            )
            for state in invalid_layer_norms
        ]
        malformed.append(
            replace(
                cache,
                final_norm=replace(
                    cache.final_norm,
                    mean=cache.final_norm.mean.at[0, 0].set(jnp.nan),
                ),
            )
        )
        malformed_ema = replace(
            cache,
            layer_caches=(
                replace(
                    first,
                    ema=replace(
                        first.ema,
                        h=first.ema.h.at[0, 0, 0].set(jnp.nan + 0.0j),
                    ),
                ),
                *cache.layer_caches[1:],
            ),
        )
        malformed.append(malformed_ema)

        for invalid in malformed:
            with pytest.raises(Exception, match=CACHE_INVARIANT_MESSAGE):
                jax.block_until_ready(model(jnp.asarray([[4]], dtype=jnp.int32), cache=invalid))

        compiled_violation = eqx.filter_jit(lambda state: cache_invariant_violation(state, config))
        assert bool(np.asarray(compiled_violation(malformed_ema)))

        assert bool(np.asarray(cache_invariant_violation(malformed_ema, config)))
        assert first.attn is not None
        malformed_kv = replace(
            cache,
            layer_caches=(
                replace(
                    first,
                    attn=replace(
                        first.attn,
                        k=first.attn.k.at[0, 0, 0, 0].set(jnp.nan),
                    ),
                ),
                *cache.layer_caches[1:],
            ),
        )
        assert not bool(np.asarray(cache_invariant_violation(malformed_kv, config)))
        assert bool(
            np.asarray(
                cache_invariant_violation(
                    malformed_kv,
                    config,
                    check_full_payload=True,
                )
            )
        )

    def test_zero_layer_complete_cache_requires_pristine_final_norm(self) -> None:
        """A final-norm-only zero-layer cache still validates its zero timeline."""
        config = small_config(num_layers=0)
        malformed = ModelCache(
            layer_caches=(),
            final_norm=NormState(
                count=jnp.zeros((1,), dtype=jnp.int32),
                mean=jnp.ones((1, config.norm_num_groups), dtype=jnp.float32),
                var=jnp.ones((1, config.norm_num_groups), dtype=jnp.float32),
            ),
        )
        assert bool(np.asarray(cache_invariant_violation(malformed, config)))

    @pytest.mark.fast
    def test_model_cache_rejects_coherent_count_overflow(self, random_seed: int) -> None:
        """Model-entry validation checks one shared timeline before all layers."""
        config = small_config()
        model = MegalodonModel(config, key=jax.random.PRNGKey(random_seed))
        _, cache = model(jnp.asarray([[1]], dtype=jnp.int32), return_cache=True)
        assert cache is not None and cache.final_norm is not None
        maximum = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        exhausted_layers = []
        for layer in cache.layer_caches:
            assert layer is not None and layer.attn is not None and layer.norm is not None
            exhausted_layers.append(
                replace(
                    layer,
                    position=maximum,
                    attn=replace(layer.attn, count=maximum),
                    norm=replace(
                        layer.norm,
                        count=jnp.full_like(layer.norm.count, maximum),
                    ),
                )
            )
        exhausted = replace(
            cache,
            layer_caches=tuple(exhausted_layers),
            final_norm=replace(
                cache.final_norm,
                count=jnp.full_like(cache.final_norm.count, maximum),
            ),
        )

        with pytest.raises(Exception, match=CACHE_INVARIANT_MESSAGE):
            jax.block_until_ready(
                model(jnp.asarray([[2]], dtype=jnp.int32), cache=exhausted, return_cache=True)
            )

    def test_pristine_cache_rejects_history_buffers(self, random_seed: int) -> None:
        """A zero timeline cannot carry zero-filled or active history buffers."""
        config = small_config()
        model = MegalodonModel(config, key=jax.random.PRNGKey(random_seed))
        cache = init_cache(config)
        first = LayerCache(
            attn=None,
            norm=None,
            ema=None,
            position=jnp.asarray(0, dtype=jnp.int32),
        )
        zero_history = jnp.zeros((1, config.model_dim, config.cema_ndim), dtype=jnp.complex64)
        for history in (zero_history, zero_history.at[0, 0, 0].set(1.0 + 0.0j)):
            invalid = replace(
                cache,
                layer_caches=(
                    replace(first, ema=EMAState(h=history)),
                    *cache.layer_caches[1:],
                ),
            )
            with pytest.raises(Exception, match=CACHE_INVARIANT_MESSAGE):
                jax.block_until_ready(model(jnp.asarray([[1]], dtype=jnp.int32), cache=invalid))


# -----------------------------------------------------------------------------
# Parity Tests (PyTorch Reference)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------


class TestCacheExecutionMode:
    """Tests for inference-only cache population."""

    @pytest.mark.parametrize("deterministic", [False, True])
    def test_cache_population_respects_execution_mode(
        self,
        random_seed: int,
        deterministic: bool,
    ) -> None:
        """Only deterministic inference may populate caches."""
        config = small_config()
        batch, seq = 2, 16
        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)
        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        dropout_key = jax.random.PRNGKey(42)
        if not deterministic:
            with pytest.raises(ValueError, match="inference-only"):
                model(
                    input_ids,
                    return_cache=True,
                    deterministic=False,
                    key=dropout_key,
                )
        else:
            _, cache = model(
                input_ids,
                return_cache=True,
                deterministic=True,
                key=dropout_key,
            )
            assert cache is not None
            assert all(layer is not None for layer in cache.layer_caches)


class TestFixDropoutKeyGuard:
    """Tests for dropout key validation in MegalodonModel."""

    def test_requires_key_when_dropout_enabled(self, random_seed: int) -> None:
        """Ensure deterministic=False requires a PRNG key when dropout is active.

        :param int random_seed: Random seed fixture.
        """
        config = small_config(
            vocab_size=128,
            num_layers=1,
            dropout=0.1,
        )
        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)
        input_ids = jax.random.randint(key, (2, 8), minval=0, maxval=config.vocab_size)

        with pytest.raises(ValueError, match="PRNG key required"):
            model(input_ids, deterministic=False, key=None)

    def test_embedding_dropout_uses_independent_deterministic_key(self, random_seed: int) -> None:
        """Embedding dropout is active before a zero-layer model's final norm."""
        config = small_config(vocab_size=128, num_layers=0, dropout=0.5)
        model = MegalodonModel(config, key=jax.random.PRNGKey(random_seed))
        input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
        key_a = jax.random.PRNGKey(100)
        key_b = jax.random.PRNGKey(101)

        train_a, _ = model(input_ids, deterministic=False, key=key_a)
        train_a_repeat, _ = model(input_ids, deterministic=False, key=key_a)
        train_b, _ = model(input_ids, deterministic=False, key=key_b)
        inference, _ = model(input_ids, deterministic=True)

        np.testing.assert_array_equal(np.asarray(train_a), np.asarray(train_a_repeat))
        assert not np.array_equal(np.asarray(train_a), np.asarray(train_b))
        assert not np.array_equal(np.asarray(train_a), np.asarray(inference))


class TestPaddingContract:
    """Tests for explicit mask-driven padding semantics."""

    def test_pad_id_is_metadata_not_an_embedding_rule(self, random_seed: int) -> None:
        """A token id remains learnable and valid unless an explicit mask excludes it.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1, pad_token_id=0)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jnp.array([[5, 0]], dtype=jnp.int32)
        valid_output, _ = model(input_ids, attention_mask=jnp.array([[True, True]]))
        masked_output, _ = model(input_ids, attention_mask=jnp.array([[True, False]]))

        assert np.any(np.asarray(valid_output[0, 1]) != 0.0)
        np.testing.assert_array_equal(np.asarray(masked_output[0, 1]), np.zeros(64))

        replacement = model.embed.weight.at[0].add(jnp.linspace(-1.0, 1.0, 64))
        modified = eqx.tree_at(lambda current: current.embed.weight, model, replacement)
        modified_output, _ = modified(input_ids, attention_mask=jnp.array([[True, True]]))
        assert not np.allclose(np.asarray(valid_output), np.asarray(modified_output))

    def test_explicit_masking_preserves_bf16_dtype(self, random_seed: int) -> None:
        """Explicit masking does not upcast bf16 activations.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(
            num_layers=1,
            pad_token_id=0,
            compute_dtype=jnp.bfloat16,
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        # Create input with a valid prefix and trailing pad tokens.
        input_ids = jnp.array([[5, 10, 15, 0, 0]])

        # Forward pass should preserve bf16 dtype
        hidden, _ = model(
            input_ids,
            attention_mask=jnp.array([[True, True, True, False, False]]),
            return_cache=False,
        )

        assert hidden.dtype == jnp.bfloat16, f"Expected bf16 output, got {hidden.dtype}."

    @pytest.mark.parametrize(
        "attention_mask",
        [
            pytest.param([[False, False, True, True]], id="left-padding"),
            pytest.param([[True, True, False, True]], id="interior-hole"),
        ],
    )
    def test_non_right_padded_masks_are_rejected(
        self,
        random_seed: int,
        attention_mask: list[list[bool]],
    ) -> None:
        """Physical padding cannot interrupt or precede the valid token prefix."""
        model = MegalodonModel(small_config(num_layers=1), key=jax.random.PRNGKey(random_seed))
        input_ids = jnp.asarray([[5, 10, 15, 20]], dtype=jnp.int32)

        with pytest.raises(Exception, match="contiguous valid prefix.*right padding"):
            model(input_ids, attention_mask=jnp.asarray(attention_mask))


class TestFix3UntiedLMHead:
    """Tests for untied LM head support (Fix 3)."""

    def test_tied_head_requires_explicit_flag(self, random_seed: int) -> None:
        """Vocabulary width and weight sharing are independent settings.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1, output_size=-1, share_emb=True)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        assert model.tied is True
        assert model.lm_head is None

    def test_vocab_sized_head_is_untied_by_default(self, random_seed: int) -> None:
        """Source default keeps a separate vocabulary-sized output matrix."""
        config = replace(small_config(), output_size=-1, share_emb=False)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(random_seed))
        assert model.tied is False
        assert model.lm_head is not None
        assert model.lm_head.weight.shape == (config.vocab_size, config.model_dim)

    def test_invalid_share_width_is_rejected(self) -> None:
        """Tied output cannot use a non-vocabulary width."""
        with pytest.raises(ValueError, match="share_emb"):
            replace(small_config(), output_size=128, share_emb=True)

    def test_untied_head_structure_and_forward_shape(self, random_seed: int) -> None:
        """Test that a non-vocabulary output uses a correctly shaped separate head.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1, output_size=512)
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        assert model.tied is False
        assert model.lm_head is not None
        assert model.lm_head.weight.shape == (512, config.model_dim)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        logits, _ = model(input_ids, return_cache=False)

        # Output should be (batch, seq, output_size)
        assert logits.shape == (batch, seq, 512)


class TestFix4CacheValidation:
    """Tests for cache length validation (Fix 4)."""

    def test_cache_length_mismatch_raises(self, random_seed: int) -> None:
        """Test that mismatched cache length raises ValueError.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        from megalodon_jax import ModelCache

        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)

        # Create a bad cache with wrong number of layers
        bad_cache = ModelCache(layer_caches=(None,) * 5)  # Wrong: 5 instead of 2

        with pytest.raises(ValueError, match="expected 2"):
            model(input_ids, cache=bad_cache)


class TestFix6InitMode:
    """Tests for init_mode initialization (Fix 6)."""

    @pytest.mark.parametrize("mode", ["gaussian", "xavier", "he", "bert"])
    def test_bf16_initialization_is_fp32_sample_then_cast(
        self,
        mode: InitMode,
        random_seed: int,
    ) -> None:
        """Every supported BF16 initializer retains the FP32 sampling grid."""
        initializer = get_initializer(mode, dim=16)
        key = jax.random.PRNGKey(random_seed)

        full = initializer(key, (32, 16), jnp.float32)
        compact = initializer(key, (32, 16), jnp.bfloat16)

        np.testing.assert_array_equal(
            np.asarray(compact),
            np.asarray(full.astype(jnp.bfloat16)),
        )

    def test_gaussian_linear_init_uses_unit_scale(self, random_seed: int) -> None:
        """Test that gaussian init for Linear layers uses std ~ 1.0 (dim=None).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1, init_mode="gaussian")

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Linear weights should use std ~ 1.0 (truncated normal), not 1/sqrt(dim).
        linear_weight = model.model.layers[0].attn.wz.weight
        var = jnp.var(linear_weight)
        assert var > 0.2, f"Gaussian Linear init variance too small: {var}"

    def test_he_init_applied(self, random_seed: int) -> None:
        """Test that He initialization is applied when init_mode='he'.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1, init_mode="he")

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        weight = np.asarray(model.model.layers[0].attn.wz.weight)
        expected_std = 1.0 / np.sqrt(3.0 * weight.shape[-1])
        np.testing.assert_allclose(weight.std(), expected_std, rtol=0.12)

    def test_internal_modes_do_not_change_embedding_policy(self, random_seed: int) -> None:
        """Internal modes change projections but never boundary tensors.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        base_config = dict(
            vocab_size=256,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
        )

        key = jax.random.PRNGKey(random_seed)

        # Create models with different init modes
        model_he = MegalodonForCausalLM(MegalodonConfig(**base_config, init_mode="he"), key=key)
        model_xavier = MegalodonForCausalLM(
            MegalodonConfig(**base_config, init_mode="xavier"), key=key
        )
        model_bert = MegalodonForCausalLM(MegalodonConfig(**base_config, init_mode="bert"), key=key)

        np.testing.assert_array_equal(
            np.asarray(model_he.model.embed.weight),
            np.asarray(model_xavier.model.embed.weight),
        )
        np.testing.assert_array_equal(
            np.asarray(model_he.model.embed.weight),
            np.asarray(model_bert.model.embed.weight),
        )

        he_weight = np.asarray(model_he.model.layers[0].attn.wz.weight)
        xavier_weight = np.asarray(model_xavier.model.layers[0].attn.wz.weight)
        bert_weight = np.asarray(model_bert.model.layers[0].attn.wz.weight)
        np.testing.assert_allclose(
            he_weight.std(), 1.0 / np.sqrt(3.0 * he_weight.shape[-1]), rtol=0.12
        )
        np.testing.assert_allclose(
            xavier_weight.std(),
            np.sqrt(2.0 / sum(xavier_weight.shape[-2:])),
            rtol=0.12,
        )
        np.testing.assert_allclose(bert_weight.std(), 0.02, rtol=0.12)


class TestComputeLossMasking:
    """Tests for compute_loss attention mask handling."""

    def test_masked_positions_excluded_from_loss(self, random_seed: int) -> None:
        """Test that masked positions don't contribute to loss.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Create input with last 4 positions masked
        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)
        labels = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)

        # Mask last 4 positions as invalid
        attention_mask = jnp.ones((batch, seq), dtype=bool)
        attention_mask = attention_mask.at[:, -4:].set(False)

        # Loss with mask should differ from unmasked loss
        masked_loss = model.compute_loss(input_ids, labels, attention_mask=attention_mask)
        unmasked_loss = model.compute_loss(input_ids, labels, attention_mask=None)

        # Losses should be different (unless extremely unlikely coincidence)
        assert not jnp.allclose(masked_loss, unmasked_loss), (
            "Masked and unmasked loss should differ"
        )

    def test_all_masked_returns_zero_loss(self, random_seed: int) -> None:
        """Test that fully masked input returns zero loss (no valid positions).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)
        labels = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)

        # All positions masked
        attention_mask = jnp.zeros((batch, seq), dtype=bool)

        loss = model.compute_loss(input_ids, labels, attention_mask=attention_mask)

        # With all positions masked, loss should be 0 (sum of 0 / max(0,1) = 0)
        assert loss == 0.0, f"Fully masked loss should be 0, got {loss}"

    def test_ignore_index_excludes_positions(self, random_seed: int) -> None:
        """Test that labels with ignore_index are excluded from loss.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)
        labels = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)

        # Set some labels to ignore_index (-100)
        labels_with_ignore = labels.at[:, -4:].set(-100)

        # Should not raise (previously would fail on bounds check)
        loss = model.compute_loss(input_ids, labels_with_ignore)
        assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"

        # All labels ignored should return 0 loss
        all_ignored = jnp.full_like(labels, -100)
        loss_all_ignored = model.compute_loss(input_ids, all_ignored)
        assert loss_all_ignored == 0.0, f"All ignored should be 0 loss, got {loss_all_ignored}"

    def test_custom_ignore_index(self, random_seed: int) -> None:
        """Test that custom ignore_index value works.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)
        labels = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)

        # Use -1 as ignore index instead of -100
        labels_custom = labels.at[:, -4:].set(-1)
        loss = model.compute_loss(input_ids, labels_custom, ignore_index=-1)
        assert jnp.isfinite(loss), "Loss should be finite with custom ignore_index"

    def test_bounds_check_only_on_valid_labels(self, random_seed: int) -> None:
        """Test that bounds check only applies to non-ignored labels.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)
        labels = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)

        # Out of bounds label at ignored position should NOT raise
        labels_ignored_oob = labels.at[0, 5].set(-100)  # This is fine
        loss = model.compute_loss(input_ids, labels_ignored_oob)
        assert jnp.isfinite(loss)

        # Out of bounds label at valid position SHOULD raise
        labels_valid_oob = labels.at[0, 5].set(999)
        with pytest.raises(Exception):  # eqx.error_if raises various exception types
            model.compute_loss(input_ids, labels_valid_oob)


class TestUntiedHeadInit:
    """Tests for untied LM head initialization."""

    def test_untied_head_uses_model_dim_scale(self, random_seed: int) -> None:
        """Untied heads use the boundary policy based on model width.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(
            num_layers=1,
            output_size=1024,
            init_mode="gaussian",
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        lm_head_var = jnp.var(model.lm_head.weight)
        # Variance of a standard normal truncated to [-3, 3].
        truncated_variance = 0.9733369246625415
        expected_var = truncated_variance / config.model_dim

        np.testing.assert_allclose(
            np.array(lm_head_var),
            expected_var,
            rtol=0.08,
            atol=5e-4,
            err_msg=(
                f"Untied LM head variance {lm_head_var} deviates from expected "
                f"{expected_var} (truncated std=1/sqrt(model_dim))."
            ),
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_compute_loss_empty_sequence_seq0(self, random_seed: int) -> None:
        """Test compute_loss returns 0.0 for empty sequences (seq=0).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Empty sequence (seq=0)
        input_ids = jnp.zeros((2, 0), dtype=jnp.int32)
        labels = jnp.zeros((2, 0), dtype=jnp.int32)

        loss = model.compute_loss(input_ids, labels)

        # Should return 0.0, not NaN
        assert loss == 0.0, f"Empty sequence loss should be 0.0, got {loss}"
        assert not jnp.isnan(loss), "Empty sequence loss should not be NaN"

    def test_compute_loss_single_token_seq1(self, random_seed: int) -> None:
        """Test compute_loss returns 0.0 for single-token sequences (seq=1).

        After shifting, a single-token sequence becomes empty (no predictions to make).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1, compute_dtype=jnp.bfloat16)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Single token sequence (seq=1) - after shift, becomes empty
        input_ids = jnp.ones((2, 1), dtype=jnp.int32)
        labels = jnp.ones((2, 1), dtype=jnp.int32)

        loss = model.compute_loss(input_ids, labels)

        # Should return 0.0, not NaN
        assert loss == 0.0, f"Single-token sequence loss should be 0.0, got {loss}"
        assert not jnp.isnan(loss), "Single-token sequence loss should not be NaN"
        assert loss.dtype == config.loss_softmax_dtype

    def test_cache_rejects_attention_mask_metadata(self, random_seed: int) -> None:
        """Cached calls require an unmasked generation batch.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        batch, seq = 2, 16
        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)
        # Even an all-True mask is rejected so the cached API never silently
        # discards validity metadata that cannot be represented in the cache.
        attention_mask = jnp.ones((batch, seq), dtype=bool)

        with pytest.raises(ValueError, match="attention_mask is unsupported with cached calls"):
            model(input_ids, attention_mask=attention_mask, return_cache=True)

        logits, cache = model(input_ids, return_cache=True)
        assert logits.shape == (batch, seq, config.vocab_size)
        assert cache is not None

    @pytest.mark.parametrize("supply_cache", [False, True])
    def test_nondeterministic_calls_reject_streaming_state(
        self, random_seed: int, supply_cache: bool
    ) -> None:
        """Training cannot return or consume a partially advanced inference cache."""
        config = small_config(num_layers=1)
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(random_seed))
        tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
        cache = None
        return_cache = True
        if supply_cache:
            _, cache = model(tokens[:, :1], return_cache=True)
            return_cache = False

        with pytest.raises(ValueError, match="inference-only"):
            model(
                tokens,
                cache=cache,
                return_cache=return_cache,
                deterministic=False,
                key=jax.random.PRNGKey(1),
            )

    def test_empty_batch_handling(self, random_seed: int) -> None:
        """Test that empty batch (B=0) is handled gracefully.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Empty batch (B=0)
        input_ids = jnp.zeros((0, 16), dtype=jnp.int32)

        logits, cache = model(input_ids, return_cache=True)

        assert logits.shape == (0, 16, config.vocab_size)
        assert cache is not None
        assert len(cache.layer_caches) == config.num_layers

    def test_empty_sequence_handling(self, random_seed: int) -> None:
        """Test that empty sequence (L=0) is handled gracefully.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Empty sequence (L=0)
        input_ids = jnp.zeros((2, 0), dtype=jnp.int32)

        logits, cache = model(input_ids, return_cache=True)

        assert logits.shape == (2, 0, config.vocab_size)
        assert cache is not None
        assert len(cache.layer_caches) == config.num_layers

    def test_empty_input_dtype_matches_model(self, random_seed: int) -> None:
        """Empty input should return output with same dtype as model (not hardcoded float32).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1, compute_dtype=jnp.bfloat16)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Empty batch (B=0)
        empty_batch = jnp.zeros((0, 16), dtype=jnp.int32)
        logits_b0, _ = model(empty_batch, return_cache=False)
        assert logits_b0.dtype == jnp.float32, f"Expected float32, got {logits_b0.dtype}"

        # Empty sequence (L=0)
        empty_seq = jnp.zeros((2, 0), dtype=jnp.int32)
        logits_l0, _ = model(empty_seq, return_cache=False)
        assert logits_l0.dtype == jnp.float32, f"Expected float32, got {logits_l0.dtype}"

        logits, _ = model(jnp.asarray([[1, 2]], dtype=jnp.int32))
        assert logits.dtype == jnp.float32

    def test_vocab_bounds_input_ids_raises(self, random_seed: int) -> None:
        """Test that out-of-bounds input_ids raises an error.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # input_ids with value >= vocab_size
        bad_input_ids = jnp.array([[1, 2, 300, 4]])  # 300 >= 256

        with pytest.raises(Exception):  # eqx.error_if raises EquinoxRuntimeError
            model(bad_input_ids, return_cache=False)

    def test_vocab_bounds_labels_raises(self, random_seed: int) -> None:
        """Test that out-of-bounds labels in compute_loss raises an error.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(num_layers=1)

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jnp.array([[1, 2, 3, 4]])
        bad_labels = jnp.array([[2, 300, 4, 5]])  # 300 >= 256

        with pytest.raises(Exception):  # eqx.error_if raises EquinoxRuntimeError
            model.compute_loss(input_ids, bad_labels)

    def test_loss_close_bf16_vs_fp32(self, random_seed: int) -> None:
        """Test bf16 compute loss stays close to fp32 loss.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config_fp32 = small_config(num_layers=1, compute_dtype=jnp.float32)
        config_bf16 = replace(config_fp32, compute_dtype=jnp.bfloat16)

        key = jax.random.PRNGKey(random_seed)
        k_model, k_data = jax.random.split(key)

        model_fp32 = MegalodonForCausalLM(config_fp32, key=k_model)
        model_bf16 = MegalodonForCausalLM(config_bf16, key=k_model)

        k_input, k_labels = jax.random.split(k_data)
        batch, seq = 2, 8
        input_ids = jax.random.randint(
            k_input, (batch, seq), minval=1, maxval=config_fp32.vocab_size
        )
        labels = jax.random.randint(k_labels, (batch, seq), minval=1, maxval=config_fp32.vocab_size)

        loss_fp32 = model_fp32.compute_loss(input_ids, labels)
        loss_bf16 = model_bf16.compute_loss(input_ids, labels)

        assert jnp.isfinite(loss_fp32)
        assert jnp.isfinite(loss_bf16)
        assert loss_bf16.dtype == config_bf16.loss_softmax_dtype
        np.testing.assert_allclose(
            np.array(loss_fp32),
            np.array(loss_bf16),
            rtol=5e-2,
            atol=5e-2,
            err_msg="bf16 compute loss should be close to fp32 loss",
        )

    def test_loss_softmax_dtype_bf16(self, random_seed: int) -> None:
        """Test that loss can be computed in bf16 softmax dtype.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(
            num_layers=1,
            compute_dtype=jnp.bfloat16,
            loss_softmax_dtype=jnp.bfloat16,
        )

        key = jax.random.PRNGKey(random_seed)
        k_model, k_input, k_labels = jax.random.split(key, 3)
        model = MegalodonForCausalLM(config, key=k_model)
        batch, seq = 2, 8
        input_ids = jax.random.randint(k_input, (batch, seq), minval=1, maxval=config.vocab_size)
        labels = jax.random.randint(k_labels, (batch, seq), minval=1, maxval=config.vocab_size)

        loss = model.compute_loss(input_ids, labels)

        assert loss.dtype == jnp.bfloat16
        assert jnp.isfinite(loss)


class TestPrecisionAudit:
    """Tests for precision policy audits."""

    def test_sensitive_params_fp32(self, random_seed: int) -> None:
        """Ensure sensitive params remain fp32 by default and audit detects drift.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config(
            vocab_size=128,
            num_layers=1,
            chunk_size=8,
            compute_dtype=jnp.bfloat16,
            rescale_nffn=True,
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)
        layer_prior = TimestepNorm(
            config.model_dim,
            config.norm_num_groups,
            prior_count=2,
            eps=config.norm_eps,
        )
        final_prior = TimestepNorm(
            config.model_dim,
            config.norm_num_groups,
            prior_count=2,
            eps=config.norm_eps,
        )
        core = eqx.tree_at(
            lambda m: (m.layers[0].attn.timenorm, m.norm),
            model.model,
            (layer_prior, final_prior),
        )
        model = eqx.tree_at(lambda m: m.model, model, core)

        mismatches = audit_sensitive_param_dtypes(model)
        assert mismatches == {}

        model_bf16 = jax.tree.map(floating_to_bf16, model)
        mismatches = audit_sensitive_param_dtypes(model_bf16)
        assert "layers.0.attn.cema.alpha" in mismatches
        assert "layers.0.ffn.alpha" in mismatches
        assert "layers.0.attn.timenorm.prior_mean" in mismatches
        assert "layers.0.attn.timenorm.prior_logv" in mismatches
        assert "norm.prior_mean" in mismatches
        assert "norm.prior_logv" in mismatches

        restored = ensure_sensitive_param_dtype(model_bf16)
        assert audit_sensitive_param_dtypes(restored) == {}

    def test_bf16_storage_applies_only_to_ordinary_parameters(self, random_seed: int) -> None:
        """Compact storage keeps embeddings and projections BF16 and sensitive state FP32."""
        config = small_config(
            vocab_size=128,
            num_layers=1,
            chunk_size=8,
            swiglu=True,
            rescale_nffn=True,
            param_dtype=jnp.bfloat16,
            compute_dtype=jnp.bfloat16,
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(random_seed))
        reference = MegalodonForCausalLM(
            replace(config, param_dtype=jnp.float32),
            key=jax.random.PRNGKey(random_seed),
        )
        layer = model.model.layers[0]
        reference_layer = reference.model.layers[0]
        assert model.lm_head is not None
        assert reference.lm_head is not None

        ordinary_pairs = [
            (model.model.embed.weight, reference.model.embed.weight),
            (model.lm_head.weight, reference.lm_head.weight),
            (layer.attn.wz.weight, reference_layer.attn.wz.weight),
            (layer.attn.wz.bias, reference_layer.attn.wz.bias),
            (layer.attn.wv.weight, reference_layer.attn.wv.weight),
            (layer.attn.wv.bias, reference_layer.attn.wv.bias),
            (layer.attn.wr.weight, reference_layer.attn.wr.weight),
            (layer.attn.wr.bias, reference_layer.attn.wr.bias),
            (layer.attn.wh1.weight, reference_layer.attn.wh1.weight),
            (layer.attn.wh1.bias, reference_layer.attn.wh1.bias),
            (layer.attn.wh2.weight, reference_layer.attn.wh2.weight),
            (layer.ffn.fc1.weight, reference_layer.ffn.fc1.weight),
            (layer.ffn.fc2.weight, reference_layer.ffn.fc2.weight),
            (layer.ffn.fc3.weight, reference_layer.ffn.fc3.weight),
        ]
        for compact, full in ordinary_pairs:
            assert compact is not None and full is not None
            assert compact.dtype == jnp.bfloat16
            np.testing.assert_array_equal(
                np.asarray(compact), np.asarray(full.astype(jnp.bfloat16))
            )
        assert audit_sensitive_param_dtypes(model) == {}


class TestComplexEMAMask:
    """Tests for ComplexEMA mask handling."""

    def test_mask_prevents_state_contamination(self, random_seed: int) -> None:
        """Test that masked positions don't contaminate EMA hidden state.

        When masked positions have large values, masking them out should prevent
        those values from affecting the final hidden state. Without masking, the
        large values would propagate through the EMA and cause a very different
        final state.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        from megalodon_jax.layers import ComplexEMA

        dim, ndim = 32, 4
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        ema = ComplexEMA(dim, ndim, key=key)

        # Input with large values at masked positions
        x_clean = jnp.ones((batch, dim, seq))
        x_contaminated = x_clean.at[:, :, 8:].set(1e6)  # Last 8 positions have huge values

        # Mask: first 8 valid, last 8 masked
        mask = jnp.concatenate(
            [jnp.ones((batch, 8), dtype=bool), jnp.zeros((batch, 8), dtype=bool)],
            axis=1,
        )

        # With mask on contaminated input
        _, h_masked_contaminated = ema(x_contaminated, mask=mask, return_state=True)

        # Without mask on clean input (same as if contaminated values were zeros)
        x_zeroed = x_clean.at[:, :, 8:].set(0.0)
        _, h_clean_zeroed = ema(x_zeroed, mask=None, return_state=True)

        # Hidden states should match - mask effectively zeros those positions
        np.testing.assert_allclose(
            np.array(h_masked_contaminated),
            np.array(h_clean_zeroed),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Masking should produce same state as zeroing those positions",
        )

        # Also verify: without mask, contamination would cause very different state
        _, h_unmasked_contaminated = ema(x_contaminated, mask=None, return_state=True)
        assert not np.allclose(
            np.array(h_masked_contaminated), np.array(h_unmasked_contaminated)
        ), "Without mask, contamination should cause different hidden state"

    def test_mask_zeros_contribution_from_masked_positions(self, random_seed: int) -> None:
        """Test that masked positions contribute zero to EMA state.

        The EMA output at masked positions is NOT zero (due to causal convolution),
        but the CONTRIBUTION from masked positions to the state should be zero.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        from megalodon_jax.layers import ComplexEMA

        dim, ndim = 16, 4
        batch, seq = 1, 8

        key = jax.random.PRNGKey(random_seed)
        ema = ComplexEMA(dim, ndim, key=key)

        # Random input
        x = jax.random.normal(key, (batch, dim, seq))

        # Mask middle positions
        mask = jnp.array([[True, True, False, False, False, True, True, True]])

        # With mask
        y_masked, h_masked = ema(x, mask=mask, return_state=True)

        # Same as zeroing those positions
        x_zeroed = x.at[:, :, 2:5].set(0.0)
        y_zeroed, h_zeroed = ema(x_zeroed, mask=None, return_state=True)

        # Outputs and states should match
        np.testing.assert_allclose(
            np.array(y_masked),
            np.array(y_zeroed),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Masking should produce same output as zeroing those positions",
        )
        np.testing.assert_allclose(
            np.array(h_masked),
            np.array(h_zeroed),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Masking should produce same state as zeroing those positions",
        )


class TestAttentionMasking:
    """Tests for attention masking edge cases."""

    def test_fully_masked_query_outputs_zeros(self) -> None:
        """Fully masked queries (all keys invalid) should output zeros, not value averages.

        Regression test for bug where using finfo.min instead of -inf caused
        softmax to produce uniform distribution instead of NaN, so fully masked
        queries returned an average of values rather than zeros.

        :return None: None.
        """
        from megalodon_jax.layers.attention import attention_single_chunk

        B, L_q, L_kv, H, Dh, Dv = 2, 4, 8, 2, 16, 32
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (B, L_q, H, Dh))
        k = jax.random.normal(k2, (B, L_kv, H, Dh))
        v = jax.random.normal(k3, (B, L_kv, H, Dv))

        # Fully masked: all keys invalid for batch 0, some valid for batch 1
        kv_mask = jnp.array(
            [
                [False] * L_kv,  # Batch 0: all masked
                [True] * 4 + [False] * 4,  # Batch 1: first 4 valid
            ]
        )

        out = attention_single_chunk(q, k, v, kv_mask=kv_mask, causal=False)

        # Batch 0 should be all zeros (no valid keys to attend to)
        np.testing.assert_allclose(
            np.array(out[0]),
            np.zeros((L_q, H, Dv)),
            atol=1e-6,
            err_msg="Fully masked queries should output zeros, not value averages",
        )

        # Batch 1 should be non-zero (has valid keys)
        assert jnp.any(out[1] != 0), "Partially masked batch should have non-zero output"


class TestPackedMetadata:
    """Tests for strict packed metadata plumbing in model forward/loss."""

    def test_compute_loss_parity_without_effective_segmentation(self, random_seed: int) -> None:
        """Passing monotonic position_ids and single segment should match default loss."""
        config = small_config()
        key = jax.random.PRNGKey(random_seed)
        k_model, k_data = jax.random.split(key)
        model = MegalodonForCausalLM(config, key=k_model)

        batch, seq = 2, 12
        input_ids = jax.random.randint(k_data, (batch, seq), 0, config.vocab_size)
        labels = input_ids
        attention_mask = jnp.ones((batch, seq), dtype=jnp.bool_)
        segment_ids = jnp.ones((batch, seq), dtype=jnp.int32)
        position_ids = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, :], (batch, seq))

        loss_default = model.compute_loss(input_ids, labels, attention_mask=attention_mask)
        loss_meta = model.compute_loss(
            input_ids,
            labels,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        np.testing.assert_allclose(
            np.array(loss_default), np.array(loss_meta), rtol=1e-5, atol=1e-5
        )

    def test_segment_ids_change_outputs_for_packed_sequence(self, random_seed: int) -> None:
        """Strict packed metadata should alter outputs on later packed segments."""
        config = replace(small_config(), chunk_size=8)
        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jnp.asarray([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
        segment_ids = jnp.asarray([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=jnp.int32)
        position_ids = jnp.asarray([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=jnp.int32)

        logits_default, _ = model(input_ids, attention_mask=attention_mask, return_cache=False)
        logits_strict, _ = model(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            return_cache=False,
        )
        assert not np.allclose(
            np.array(logits_default[:, 4:, :]),
            np.array(logits_strict[:, 4:, :]),
        )

    def test_cache_or_return_cache_with_segment_ids_raises(self, random_seed: int) -> None:
        """Strict packed metadata must be rejected on any streaming path."""
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(random_seed))

        input_ids = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
        segment_ids = jnp.ones_like(input_ids, dtype=jnp.int32)

        with pytest.raises(ValueError, match="non-cached training"):
            model(input_ids, segment_ids=segment_ids, return_cache=True)

    def test_return_cache_with_segment_ids_raises_nondeterministic(self, random_seed: int) -> None:
        """The cache rejection must hold under deterministic=False too.

        Regression: the guard was gated on ``return_cache and deterministic``,
        so dropout-mode calls silently returned a ModelCache carrying
        segmented final-norm state instead of raising.
        """
        config = small_config()
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(random_seed))

        input_ids = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
        segment_ids = jnp.ones_like(input_ids, dtype=jnp.int32)

        with pytest.raises(ValueError, match="inference-only"):
            model(
                input_ids,
                segment_ids=segment_ids,
                return_cache=True,
                deterministic=False,
                key=jax.random.PRNGKey(0),
            )

    def test_supports_segment_reset_flag_present(self, random_seed: int) -> None:
        """Capability flag must be exposed on classes and instances."""
        assert MegalodonModel.supports_segment_reset is True
        assert MegalodonForCausalLM.supports_segment_reset is True

        model = MegalodonForCausalLM(small_config(), key=jax.random.PRNGKey(random_seed))
        assert getattr(model, "supports_segment_reset", False) is True
        assert getattr(model.model, "supports_segment_reset", False) is True

    def test_packed_isolation_matches_standalone_doc(self, random_seed: int) -> None:
        """Doc B packed after doc A must produce the same logits as doc B alone.

        This is the issue #7 "done" criterion: with strict metadata, no state
        (attention, ComplexEMA, TimestepNorm) may leak across packed documents.
        """
        config = small_config()
        key = jax.random.PRNGKey(random_seed)
        k_model, k_a, k_b = jax.random.split(key, 3)
        model = MegalodonForCausalLM(config, key=k_model)

        # doc A (6) + doc B (7) + padding (3), single 16-token chunk
        ids_a = jax.random.randint(k_a, (1, 6), 0, config.vocab_size)
        ids_b = jax.random.randint(k_b, (1, 7), 0, config.vocab_size)
        input_ids = jnp.concatenate([ids_a, ids_b, jnp.zeros((1, 3), dtype=ids_a.dtype)], axis=1)
        attention_mask = jnp.asarray([[True] * 13 + [False] * 3])
        segment_ids = jnp.asarray([[1] * 6 + [2] * 7 + [0] * 3], dtype=jnp.int32)
        position_ids = jnp.asarray([list(range(6)) + list(range(7)) + [0, 0, 0]], dtype=jnp.int32)

        logits_packed, _ = model(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        logits_b, _ = model(ids_b)

        np.testing.assert_allclose(
            np.array(logits_packed[:, 6:13, :]),
            np.array(logits_b),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Packed doc B logits should match doc B run alone",
        )

    def test_position_ids_derived_from_segment_ids(self, random_seed: int) -> None:
        """Omitting position_ids must derive per-document RoPE positions.

        Regression: segment_ids without position_ids silently kept continuous
        global RoPE phases across document boundaries while attention was
        still isolated, so packed docs no longer matched running alone.
        """
        config = small_config()
        key = jax.random.PRNGKey(random_seed)
        k_model, k_a, k_b = jax.random.split(key, 3)
        model = MegalodonForCausalLM(config, key=k_model)

        # doc A (6) + doc B (7) + padding (3), single 16-token chunk
        ids_a = jax.random.randint(k_a, (1, 6), 0, config.vocab_size)
        ids_b = jax.random.randint(k_b, (1, 7), 0, config.vocab_size)
        input_ids = jnp.concatenate([ids_a, ids_b, jnp.zeros((1, 3), dtype=ids_a.dtype)], axis=1)
        attention_mask = jnp.asarray([[True] * 13 + [False] * 3])
        segment_ids = jnp.asarray([[1] * 6 + [2] * 7 + [0] * 3], dtype=jnp.int32)
        position_ids = jnp.asarray([list(range(6)) + list(range(7)) + [0, 0, 0]], dtype=jnp.int32)

        logits_derived, _ = model(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        logits_explicit, _ = model(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        logits_b, _ = model(ids_b)

        # Valid region matches the explicit-position_ids call (padding may
        # differ: derived pad positions are run-local, the explicit ones are 0)
        np.testing.assert_allclose(
            np.array(logits_derived[:, :13, :]),
            np.array(logits_explicit[:, :13, :]),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Derived position_ids should reproduce explicit per-doc positions",
        )
        np.testing.assert_allclose(
            np.array(logits_derived[:, 6:13, :]),
            np.array(logits_b),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Packed doc B logits should match doc B alone without explicit position_ids",
        )

    def test_sequential_segment_scan_config_matches_associative(self, random_seed: int) -> None:
        """The config-selected sequential CEMA fallback must match the default path.

        use_associative_segment_scan=False threads from MegalodonConfig down to
        every layer's CEMA call, selecting the low-memory sequential scan; on a
        packed batch it must agree with the associative default.
        """
        key = jax.random.PRNGKey(random_seed)
        k_model, k_data = jax.random.split(key)
        config_assoc = small_config()
        config_seq = replace(config_assoc, use_associative_segment_scan=False)
        # Same key -> identical weights; only the segmented scan impl differs
        model_assoc = MegalodonForCausalLM(config_assoc, key=k_model)
        model_seq = MegalodonForCausalLM(config_seq, key=k_model)
        assert model_seq.model.layers[0].attn.use_associative_segment_scan is False

        input_ids = jax.random.randint(k_data, (1, 16), 0, config_assoc.vocab_size)
        attention_mask = jnp.asarray([[True] * 13 + [False] * 3])
        segment_ids = jnp.asarray([[1] * 6 + [2] * 7 + [0] * 3], dtype=jnp.int32)

        logits_assoc, _ = model_assoc(
            input_ids, attention_mask=attention_mask, segment_ids=segment_ids
        )
        logits_seq, _ = model_seq(input_ids, attention_mask=attention_mask, segment_ids=segment_ids)

        np.testing.assert_allclose(
            np.array(logits_assoc[:, :13, :]),
            np.array(logits_seq[:, :13, :]),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Sequential segmented CEMA (via config) should match associative default",
        )

    def test_gradient_isolation_no_leak_from_doc_a_to_doc_b(self, random_seed: int) -> None:
        """Loss on doc B tokens must produce zero gradients on doc A embeddings.

        Uses an untied LM head: with tied weights the softmax gradient is dense
        over every embedding row regardless of state leaks, which would mask the
        signal this test is after.
        """
        config = replace(small_config(), output_size=small_config().vocab_size + 1)
        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)
        assert not model.tied

        # Disjoint token-id ranges so embedding rows are attributable per doc
        ids_a = jnp.arange(10, 16, dtype=jnp.int32)[None, :]  # doc A: 6 tokens
        ids_b = jnp.arange(100, 107, dtype=jnp.int32)[None, :]  # doc B: 7 tokens
        input_ids = jnp.concatenate([ids_a, ids_b, jnp.zeros((1, 3), dtype=jnp.int32)], axis=1)
        attention_mask = jnp.asarray([[True] * 13 + [False] * 3])
        segment_ids = jnp.asarray([[1] * 6 + [2] * 7 + [0] * 3], dtype=jnp.int32)
        position_ids = jnp.asarray([list(range(6)) + list(range(7)) + [0, 0, 0]], dtype=jnp.int32)

        # Ignore all doc A predictions plus the A->B boundary pair
        # (logit at A's last position predicts B's first token)
        labels = jnp.where(
            (jnp.arange(16)[None, :] < 7) | (jnp.arange(16)[None, :] >= 13),
            -100,
            input_ids,
        )

        def loss_fn(m: MegalodonForCausalLM) -> jnp.ndarray:
            """Compute loss restricted to doc B's tokens.

            :param MegalodonForCausalLM m: Model under test.
            :return jnp.ndarray: Scalar loss value.
            """
            return m.compute_loss(
                input_ids,
                labels,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                position_ids=position_ids,
            )

        grads = eqx.filter_grad(loss_fn)(model)
        embed_grads = np.array(grads.model.embed.weight)

        np.testing.assert_allclose(
            embed_grads[10:16],
            0.0,
            atol=1e-5,
            err_msg="Doc A embedding rows received gradient from doc B's loss",
        )
        # Guard against a vacuously-zero-everywhere pass
        assert np.max(np.abs(embed_grads[100:107])) > 1e-6

    def test_trivial_segment_ids_close_to_unsegmented(self, random_seed: int) -> None:
        """A single segment spanning the valid region must match segment_ids=None.

        Forces every CEMA call in the stack onto the segmented (associative
        scan) path and every norm onto the segment-local path; a trivial
        segmentation must agree with the FFT/global baseline at valid positions.
        """
        config = replace(small_config(), chunk_size=8)
        key = jax.random.PRNGKey(random_seed)
        k_model, k_data = jax.random.split(key)
        model = MegalodonForCausalLM(config, key=k_model)

        input_ids = jax.random.randint(k_data, (2, 16), 0, config.vocab_size)
        attention_mask = jnp.asarray([[True] * 12 + [False] * 4] * 2)
        segment_ids = attention_mask.astype(jnp.int32)
        position_ids = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32)[None, :], (2, 16))

        logits_base, _ = model(input_ids, attention_mask=attention_mask)
        logits_seg, _ = model(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )

        # Padding positions legitimately differ (segment-0 queries attend to
        # nothing); the valid region must agree. Tolerance is slightly looser
        # than usual: the segmented scan and cumsum sum in different association
        # orders, leaving O(1e-4) dust on later-layer activations.
        np.testing.assert_allclose(
            np.array(logits_seg[:, :12, :]),
            np.array(logits_base[:, :12, :]),
            rtol=1e-3,
            atol=5e-4,
            err_msg="Trivial segmentation should match the unsegmented baseline",
        )

    def test_repeated_segment_ids_isolate_docs(self, random_seed: int) -> None:
        """A reused positive segment id must not link non-adjacent documents.

        Packers may legally recycle ids within a row (e.g. ``[1..., 2..., 1...]``);
        isolation is defined by contiguous runs, not raw id values, so doc C
        must match a standalone run even though it shares doc A's id.
        """
        config = small_config()
        key = jax.random.PRNGKey(random_seed)
        k_model, k_a, k_b, k_c = jax.random.split(key, 4)
        model = MegalodonForCausalLM(config, key=k_model)

        # doc A (id 1, 5) + doc B (id 2, 4) + doc C (id 1 again, 5) + padding (2)
        ids_a = jax.random.randint(k_a, (1, 5), 0, config.vocab_size)
        ids_b = jax.random.randint(k_b, (1, 4), 0, config.vocab_size)
        ids_c = jax.random.randint(k_c, (1, 5), 0, config.vocab_size)
        input_ids = jnp.concatenate(
            [ids_a, ids_b, ids_c, jnp.zeros((1, 2), dtype=ids_a.dtype)], axis=1
        )
        attention_mask = jnp.asarray([[True] * 14 + [False] * 2])
        segment_ids = jnp.asarray([[1] * 5 + [2] * 4 + [1] * 5 + [0] * 2], dtype=jnp.int32)
        position_ids = jnp.asarray(
            [list(range(5)) + list(range(4)) + list(range(5)) + [0, 0]], dtype=jnp.int32
        )

        logits_packed, _ = model(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        logits_c, _ = model(ids_c)

        np.testing.assert_allclose(
            np.array(logits_packed[:, 9:14, :]),
            np.array(logits_c),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Packed doc C logits should match doc C alone despite reused id",
        )

    @pytest.mark.parametrize("doc_b_len", [10, 18])
    def test_packed_isolation_unaligned_doc_crossing_chunks(
        self, random_seed: int, doc_b_len: int
    ) -> None:
        """A doc packed at a non-chunk-aligned offset must match itself run alone.

        chunk_size=8, doc A is 6 tokens, so doc B starts mid-chunk and crosses
        one (len 10) or two (len 18) global chunk boundaries. Attention chunk
        boundaries must be re-anchored at the document start, or the packed
        doc gets a different block-diagonal pattern than it would alone.
        """
        config = replace(small_config(), chunk_size=8)
        key = jax.random.PRNGKey(random_seed)
        k_model, k_a, k_b = jax.random.split(key, 3)
        model = MegalodonForCausalLM(config, key=k_model)

        ids_a = jax.random.randint(k_a, (1, 6), 0, config.vocab_size)
        ids_b = jax.random.randint(k_b, (1, doc_b_len), 0, config.vocab_size)
        input_ids = jnp.concatenate([ids_a, ids_b], axis=1)
        seq = 6 + doc_b_len
        attention_mask = jnp.ones((1, seq), dtype=jnp.bool_)
        segment_ids = jnp.asarray([[1] * 6 + [2] * doc_b_len], dtype=jnp.int32)
        position_ids = jnp.asarray([list(range(6)) + list(range(doc_b_len))], dtype=jnp.int32)

        logits_packed, _ = model(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        logits_b, _ = model(ids_b)

        np.testing.assert_allclose(
            np.array(logits_packed[:, 6:, :]),
            np.array(logits_b),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Unaligned packed doc B logits should match doc B run alone",
        )

    def test_compute_loss_excludes_cross_segment_label_pairs(self, random_seed: int) -> None:
        """Packed loss must equal the valid-pair-weighted mean of per-doc losses.

        Without boundary masking, doc A's last token would be scored on
        predicting doc B's first token, so packed loss/gradients would differ
        from running each document alone.
        """
        config = small_config()
        key = jax.random.PRNGKey(random_seed)
        k_model, k_a, k_b = jax.random.split(key, 3)
        model = MegalodonForCausalLM(config, key=k_model)

        # doc A (6) + doc B (7) + padding (3); labels are raw input_ids with
        # no manual ignore_index at the A->B boundary or padding
        ids_a = jax.random.randint(k_a, (1, 6), 0, config.vocab_size)
        ids_b = jax.random.randint(k_b, (1, 7), 0, config.vocab_size)
        input_ids = jnp.concatenate([ids_a, ids_b, jnp.zeros((1, 3), dtype=ids_a.dtype)], axis=1)
        attention_mask = jnp.asarray([[True] * 13 + [False] * 3])
        segment_ids = jnp.asarray([[1] * 6 + [2] * 7 + [0] * 3], dtype=jnp.int32)
        position_ids = jnp.asarray([list(range(6)) + list(range(7)) + [0, 0, 0]], dtype=jnp.int32)

        loss_packed = model.compute_loss(
            input_ids,
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        token_loss = model.compute_loss(
            input_ids,
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            reduction="none",
        )
        loss_sum, valid_count = model.compute_loss(
            input_ids,
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            reduction="sum",
            return_valid_count=True,
        )
        loss_a = model.compute_loss(ids_a, ids_a)
        loss_b = model.compute_loss(ids_b, ids_b)

        # 5 valid shifted pairs inside doc A, 6 inside doc B; the boundary
        # pair and padding must be excluded automatically
        expected_sum = 5 * np.array(loss_a) + 6 * np.array(loss_b)
        expected = expected_sum / 11
        np.testing.assert_allclose(
            np.array(loss_packed),
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Packed loss should be the valid-pair-weighted mean of per-doc losses",
        )
        assert token_loss.shape == (1, 15)
        assert int(valid_count) == 11
        np.testing.assert_allclose(np.asarray(loss_sum), expected_sum, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(token_loss).sum(), loss_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(
            np.asarray(token_loss)[0, [5, 12, 13, 14]],
            np.zeros(4, dtype=np.float32),
        )
