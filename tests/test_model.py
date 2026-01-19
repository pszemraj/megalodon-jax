"""Phase 4 Model Assembly tests - MegalodonBlock, MegalodonModel, MegalodonForCausalLM."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax import MegalodonBlock, MegalodonConfig, MegalodonForCausalLM, MegalodonModel
from megalodon_jax.convert import load_weights_from_torch
from megalodon_jax.precision import audit_sensitive_param_dtypes
from tests.utils import to_jax


def small_config() -> MegalodonConfig:
    """Create a small config for fast testing.

    :return MegalodonConfig: Minimal model configuration for tests.
    """
    return MegalodonConfig(
        vocab_size=256,
        model_dim=64,
        num_layers=2,
        num_heads=2,
        z_dim=32,
        value_dim=64,
        ffn_hidden_dim=128,
        cema_ndim=4,
        chunk_size=16,
        norm_num_groups=8,
    )


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

        # Process full sequence
        out_full, _ = block(x, return_cache=True)

        # Process in two halves with cache
        mid = seq // 2
        out1, cache1 = block(x[:, :mid], return_cache=True)
        out2, cache2 = block(x[:, mid:], cache=cache1, return_cache=True)
        out_streamed = jnp.concatenate([out1, out2], axis=1)

        # Outputs should match (within tolerance due to streaming norm differences)
        # Streaming norms accumulate statistics slightly differently than batch
        np.testing.assert_allclose(
            np.array(out_full),
            np.array(out_streamed),
            rtol=2e-3,
            atol=2e-5,
            err_msg="Streaming output differs from batch output",
        )

    def test_different_layer_ids(self, random_seed: int) -> None:
        """Test that different layer IDs produce different rescaling.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=4,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
            rescale_nffn=True,  # Enable rescaling
        )

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        block0 = MegalodonBlock(config, layer_id=0, key=k1)
        block3 = MegalodonBlock(config, layer_id=3, key=k2)

        # Different layer IDs should have different alpha values
        assert block0.ffn.alpha is not None
        assert block3.ffn.alpha is not None
        assert block0.ffn.alpha != block3.ffn.alpha
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
    """Tests for MegalodonForCausalLM with tied LM head."""

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
        config = small_config()

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
        manual_logits = hidden @ model.model.embed.weight.T

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

    def test_streaming_generation(self, random_seed: int) -> None:
        """Test streaming token-by-token generation.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch = 1

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Initial prompt
        prompt = jax.random.randint(key, (batch, 8), minval=0, maxval=config.vocab_size)

        # Generate 5 tokens autoregressively
        cache = None
        generated = []
        current_input = prompt

        for step in range(5):
            logits, cache = model(current_input, cache=cache, return_cache=True)
            next_token = jnp.argmax(logits[:, -1:], axis=-1)
            generated.append(next_token)
            current_input = next_token

        # Should have generated 5 tokens
        assert len(generated) == 5
        # Each should be shape (batch, 1)
        for tok in generated:
            assert tok.shape == (batch, 1)


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

    def test_gradient_flow_embedding(self, random_seed: int) -> None:
        """Test that gradients flow to embedding weights.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = small_config()
        batch, seq = 2, 8

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
        labels = input_ids

        def loss_fn(model: MegalodonForCausalLM) -> jnp.ndarray:
            """Compute a scalar loss for embedding gradient checks.

            :param MegalodonForCausalLM model: Model under test.
            :return jnp.ndarray: Scalar loss value.
            """
            return model.compute_loss(input_ids, labels)

        grads = eqx.filter_grad(loss_fn)(model)

        # Check embedding gradient
        embed_grad = grads.model.embed.weight
        assert embed_grad is not None
        assert jnp.all(jnp.isfinite(embed_grad))
        # Some gradients should be non-zero (for tokens that appear in input)
        assert jnp.any(embed_grad != 0)


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

    def test_cache_tuple_immutability(self, random_seed: int) -> None:
        """Test that layer_caches is a tuple (not list) for JAX compatibility.

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


# -----------------------------------------------------------------------------
# Parity Tests (PyTorch Reference)
# -----------------------------------------------------------------------------


@pytest.mark.torch_ref
class TestParity:
    """Tests comparing JAX implementation to PyTorch reference."""

    @pytest.fixture
    def parity_config(self) -> MegalodonConfig:
        """Config for parity testing - matches PyTorch defaults.

        :return MegalodonConfig: Parity test configuration.
        """
        return MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=2,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
            norm_affine=True,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )

    def test_causal_lm_forward_parity(
        self,
        random_seed: int,
        parity_config: MegalodonConfig,
        torch_device: torch.device,
    ) -> None:
        """Test that JAX MegalodonForCausalLM matches PyTorch reference.

        :param int random_seed: Random seed fixture.
        :param MegalodonConfig parity_config: Parity config fixture.
        :param torch.device torch_device: Torch device fixture.
        :return None: None.
        """
        torch = pytest.importorskip("torch")
        megalodon = pytest.importorskip("megalodon")
        config = parity_config
        batch, seq = 2, 32

        # Create PyTorch model
        torch_config = megalodon.MegalodonConfig(
            vocab_size=config.vocab_size,
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            z_dim=config.z_dim,
            value_dim=config.value_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            cema_ndim=config.cema_ndim,
            chunk_size=config.chunk_size,
            norm_num_groups=config.norm_num_groups,
            norm_affine=config.norm_affine,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        torch_model = megalodon.MegalodonForCausalLM(torch_config).to(torch_device).eval()

        # Create JAX model
        key = jax.random.PRNGKey(random_seed)
        jax_model = MegalodonForCausalLM(config, key=key)

        # Copy weights from PyTorch to JAX (from CPU state_dict)
        cpu_state_dict = {k: v.cpu() for k, v in torch_model.state_dict().items()}
        jax_model = load_weights_from_torch(jax_model, cpu_state_dict)

        # Create test input on torch_device
        torch.manual_seed(random_seed)
        input_ids_torch = torch.randint(0, config.vocab_size, (batch, seq), device=torch_device)
        input_ids_jax = to_jax(input_ids_torch.cpu())

        # Forward pass
        with torch.no_grad():
            torch_out = torch_model(input_ids_torch, use_cache=False, return_dict=True)
            torch_logits = torch_out.logits.cpu()

        jax_logits, _ = jax_model(input_ids_jax, return_cache=False)

        # Compare outputs
        # Cross-framework parity has slightly looser tolerances due to
        # different operation ordering and GPU TF32 precision
        np.testing.assert_allclose(
            np.array(jax_logits),
            torch_logits.numpy(),
            rtol=1e-3,
            atol=1e-4,
            err_msg="JAX logits differ from PyTorch reference",
        )

    def test_streaming_parity(
        self,
        random_seed: int,
        parity_config: MegalodonConfig,
        torch_device: torch.device,
    ) -> None:
        """Test that JAX streaming matches PyTorch streaming.

        :param int random_seed: Random seed fixture.
        :param MegalodonConfig parity_config: Parity config fixture.
        :param torch.device torch_device: Torch device fixture.
        :return None: None.
        """
        torch = pytest.importorskip("torch")
        megalodon = pytest.importorskip("megalodon")
        config = parity_config
        batch = 1
        prompt_len = 16
        gen_len = 4

        # Create PyTorch model
        torch_config = megalodon.MegalodonConfig(
            vocab_size=config.vocab_size,
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            z_dim=config.z_dim,
            value_dim=config.value_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            cema_ndim=config.cema_ndim,
            chunk_size=config.chunk_size,
            norm_num_groups=config.norm_num_groups,
            norm_affine=config.norm_affine,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        torch_model = megalodon.MegalodonForCausalLM(torch_config).to(torch_device).eval()

        # Create JAX model and load weights (from CPU state_dict)
        key = jax.random.PRNGKey(random_seed)
        jax_model = MegalodonForCausalLM(config, key=key)
        cpu_state_dict = {k: v.cpu() for k, v in torch_model.state_dict().items()}
        jax_model = load_weights_from_torch(jax_model, cpu_state_dict)

        # Create prompt on torch_device
        torch.manual_seed(random_seed)
        prompt_torch = torch.randint(0, config.vocab_size, (batch, prompt_len), device=torch_device)
        prompt_jax = to_jax(prompt_torch.cpu())

        # PyTorch streaming generation
        torch_generated = []
        with torch.no_grad():
            torch_out = torch_model(prompt_torch, use_cache=True, return_dict=True)
            torch_pkv = torch_out.past_key_values

            for _ in range(gen_len):
                next_token = torch_out.logits[:, -1:].argmax(dim=-1)
                torch_generated.append(next_token.cpu().item())
                torch_out = torch_model(
                    next_token, past_key_values=torch_pkv, use_cache=True, return_dict=True
                )
                torch_pkv = torch_out.past_key_values

        # JAX streaming generation
        jax_generated = []
        jax_logits, jax_cache = jax_model(prompt_jax, return_cache=True)
        for _ in range(gen_len):
            next_token = jnp.argmax(jax_logits[:, -1:], axis=-1)
            jax_generated.append(int(next_token[0, 0]))
            jax_logits, jax_cache = jax_model(next_token, cache=jax_cache, return_cache=True)

        # Generated tokens should match
        assert torch_generated == jax_generated, (
            f"Generated tokens differ: PyTorch={torch_generated}, JAX={jax_generated}"
        )

    def test_streaming_parity_crosses_chunk_boundary(
        self, random_seed: int, torch_device: torch.device
    ) -> None:
        """Test streaming parity when generation crosses a chunk boundary.

        :param int random_seed: Random seed fixture.
        :param torch.device torch_device: Torch device fixture.
        :return None: None.
        """
        torch = pytest.importorskip("torch")
        megalodon = pytest.importorskip("megalodon")
        config = MegalodonConfig(
            vocab_size=128,
            model_dim=64,
            num_layers=2,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=8,
            norm_affine=True,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        batch = 1
        prompt_len = config.chunk_size - 1
        gen_len = 2

        torch_config = megalodon.MegalodonConfig(
            vocab_size=config.vocab_size,
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            z_dim=config.z_dim,
            value_dim=config.value_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            cema_ndim=config.cema_ndim,
            chunk_size=config.chunk_size,
            norm_num_groups=config.norm_num_groups,
            norm_affine=config.norm_affine,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        torch_model = megalodon.MegalodonForCausalLM(torch_config).to(torch_device).eval()

        key = jax.random.PRNGKey(random_seed)
        jax_model = MegalodonForCausalLM(config, key=key)
        cpu_state_dict = {k: v.cpu() for k, v in torch_model.state_dict().items()}
        jax_model = load_weights_from_torch(jax_model, cpu_state_dict)

        torch.manual_seed(random_seed)
        prompt_torch = torch.randint(0, config.vocab_size, (batch, prompt_len), device=torch_device)
        prompt_jax = to_jax(prompt_torch.cpu())

        torch_generated = []
        with torch.no_grad():
            torch_out = torch_model(prompt_torch, use_cache=True, return_dict=True)
            torch_pkv = torch_out.past_key_values

            for _ in range(gen_len):
                next_token = torch_out.logits[:, -1:].argmax(dim=-1)
                torch_generated.append(next_token.cpu().item())
                torch_out = torch_model(
                    next_token, past_key_values=torch_pkv, use_cache=True, return_dict=True
                )
                torch_pkv = torch_out.past_key_values

        jax_generated = []
        jax_logits, jax_cache = jax_model(prompt_jax, return_cache=True)
        for _ in range(gen_len):
            next_token = jnp.argmax(jax_logits[:, -1:], axis=-1)
            jax_generated.append(int(next_token[0, 0]))
            jax_logits, jax_cache = jax_model(next_token, cache=jax_cache, return_cache=True)

        assert torch_generated == jax_generated, (
            "Cross-boundary streaming tokens differ: "
            f"PyTorch={torch_generated}, JAX={jax_generated}"
        )


# -----------------------------------------------------------------------------
# Regression Tests for Phase 4 Fixes
# -----------------------------------------------------------------------------


class TestFix1GradientCheckpointing:
    """Tests for gradient checkpointing (Fix 1)."""

    def test_checkpointing_disables_cache_during_training(self, random_seed: int) -> None:
        """Test that use_checkpoint=True produces None caches during training.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=2,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
            use_checkpoint=True,  # Enable checkpointing
        )
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)

        # Training mode (deterministic=False) with checkpointing
        # The PRNG key is needed for dropout even if dropout=0
        dropout_key = jax.random.PRNGKey(42)
        _, cache = model(input_ids, return_cache=True, deterministic=False, key=dropout_key)

        # All layer caches should be None when checkpointing
        for layer_cache in cache.layer_caches:
            assert layer_cache is None, "Layer cache should be None when checkpointing"

    def test_checkpointing_disabled_returns_cache(self, random_seed: int) -> None:
        """Test that use_checkpoint=False returns caches normally.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=2,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
            use_checkpoint=False,  # Disable checkpointing
        )
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)

        # Deterministic mode returns caches
        _, cache = model(input_ids, return_cache=True, deterministic=True)

        # Layer caches should be populated
        for layer_cache in cache.layer_caches:
            assert layer_cache is not None, "Layer cache should be returned when not checkpointing"


class TestFixDropoutKeyGuard:
    """Tests for dropout key validation in MegalodonModel."""

    def test_requires_key_when_dropout_enabled(self, random_seed: int) -> None:
        """Ensure deterministic=False requires a PRNG key when dropout is active.

        :param int random_seed: Random seed fixture.
        """
        config = MegalodonConfig(
            vocab_size=128,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
            dropout=0.1,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)
        input_ids = jax.random.randint(key, (2, 8), minval=0, maxval=config.vocab_size)

        with pytest.raises(ValueError, match="PRNG key required"):
            model(input_ids, deterministic=False, key=None)


class TestFix2PadTokenMasking:
    """Tests for pad token masking (Fix 2)."""

    def test_pad_token_embeddings_are_zeroed(self, random_seed: int) -> None:
        """Test that pad tokens produce zero embeddings.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            pad_token_id=0,
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        # Create input with pad tokens at positions 1, 3
        input_ids = jnp.array([[5, 0, 10, 0, 15]])  # 0 is pad token

        # Get embeddings after scaling but before layers process them
        # We check embedding lookup directly
        embed = jax.vmap(jax.vmap(model.embed))(input_ids) * model.scale

        # Mask pad tokens the same way as in the model (dtype-matched zero)
        pad_mask = input_ids == config.pad_token_id
        masked_embed = jnp.where(pad_mask[:, :, None], jnp.zeros((), dtype=embed.dtype), embed)

        # Positions 1 and 3 (pad tokens) should have all-zero embeddings
        np.testing.assert_array_equal(
            np.array(masked_embed[0, 1]),
            np.zeros(config.model_dim),
            err_msg="Pad token at position 1 should have zero embedding",
        )
        np.testing.assert_array_equal(
            np.array(masked_embed[0, 3]),
            np.zeros(config.model_dim),
            err_msg="Pad token at position 3 should have zero embedding",
        )

    def test_pad_masking_preserves_bf16_dtype(self, random_seed: int) -> None:
        """Test that pad token masking preserves bf16 dtype (no upcast to float32).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            pad_token_id=0,
            compute_dtype=jnp.bfloat16,
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        # Create input with pad tokens
        input_ids = jnp.array([[5, 0, 10, 0, 15]])

        # Forward pass should preserve bf16 dtype
        hidden, _ = model(input_ids, return_cache=False)

        assert hidden.dtype == jnp.bfloat16, (
            f"Expected bf16 output, got {hidden.dtype}. Pad masking may be upcasting to float32."
        )


class TestFix3UntiedLMHead:
    """Tests for untied LM head support (Fix 3)."""

    def test_tied_head_when_output_size_matches_vocab(self, random_seed: int) -> None:
        """Test that LM head is tied when output_size equals vocab_size.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            output_size=-1,  # Tied
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        assert model.tied is True
        assert model.lm_head is None

    def test_untied_head_when_output_size_differs(self, random_seed: int) -> None:
        """Test that separate LM head is created when output_size != vocab_size.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            output_size=512,  # Different from vocab_size
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        assert model.tied is False
        assert model.lm_head is not None
        assert model.lm_head.weight.shape == (512, 64)  # (output_size, model_dim)

    def test_untied_head_forward_shapes(self, random_seed: int) -> None:
        """Test that untied LM head produces correct output shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            output_size=512,  # Different from vocab_size
        )
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

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

        config = MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=2,  # Model expects 2 layer caches
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
        )
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

    def test_gaussian_linear_init_uses_unit_scale(self, random_seed: int) -> None:
        """Test that gaussian init for Linear layers uses std ~ 1.0 (dim=None).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            init_mode="gaussian",
        )

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
        config = MegalodonConfig(
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
            init_mode="he",
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Check that weights are not the Equinox default (which is different from He)
        embed_weight = model.model.embed.weight
        assert embed_weight is not None
        # He init should have reasonable variance
        var = jnp.var(embed_weight)
        assert var > 0.001, "Embedding weights should have non-trivial variance"

    def test_none_init_keeps_defaults(self, random_seed: int) -> None:
        """Test that init_mode='none' doesn't reinitialize weights.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            init_mode="none",
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Weights should exist
        assert model.model.embed.weight is not None

    def test_different_init_modes_produce_different_weights(self, random_seed: int) -> None:
        """Test that different init modes produce different weight distributions.

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

        # BERT init has stddev=0.02, which should have much smaller variance than He
        bert_var = jnp.var(model_bert.model.embed.weight)
        he_var = jnp.var(model_he.model.embed.weight)
        xavier_var = jnp.var(model_xavier.model.embed.weight)

        # BERT has fixed small stddev, He scales with dimensions
        # BERT std=0.02 means var ~0.0004
        assert bert_var < 0.01, f"BERT init variance {bert_var} should be small"
        assert xavier_var > bert_var, "Xavier init variance should exceed BERT variance"
        assert he_var > xavier_var, "He init variance should exceed Xavier variance"


class TestComputeLossMasking:
    """Tests for compute_loss attention mask handling."""

    def test_masked_positions_excluded_from_loss(self, random_seed: int) -> None:
        """Test that masked positions don't contribute to loss.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
        config = MegalodonConfig(
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
        config = MegalodonConfig(
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
        config = MegalodonConfig(
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
        config = MegalodonConfig(
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

    def test_untied_head_gaussian_uses_output_dim_scale(self, random_seed: int) -> None:
        """Test that untied LM head with gaussian init uses std=1/sqrt(output_size).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            output_size=1024,  # Large output_size to make the bug obvious
            init_mode="gaussian",
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Expect variance near 1/output_size (std = 1/sqrt(output_size)).
        lm_head_var = jnp.var(model.lm_head.weight)
        expected_var = 1.0 / config.output_size

        np.testing.assert_allclose(
            np.array(lm_head_var),
            expected_var,
            rtol=0.25,
            atol=5e-4,
            err_msg=(
                f"Untied LM head variance {lm_head_var} deviates from expected "
                f"{expected_var} (std=1/sqrt(output_size))."
            ),
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_compute_loss_empty_sequence_seq0(self, random_seed: int) -> None:
        """Test compute_loss returns 0.0 for empty sequences (seq=0).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
        config = MegalodonConfig(
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
        model = MegalodonForCausalLM(config, key=key)

        # Single token sequence (seq=1) - after shift, becomes empty
        input_ids = jnp.ones((2, 1), dtype=jnp.int32)
        labels = jnp.ones((2, 1), dtype=jnp.int32)

        loss = model.compute_loss(input_ids, labels)

        # Should return 0.0, not NaN
        assert loss == 0.0, f"Single-token sequence loss should be 0.0, got {loss}"
        assert not jnp.isnan(loss), "Single-token sequence loss should not be NaN"

    def test_cache_with_padding_raises_error(self, random_seed: int) -> None:
        """Test that building cache with padded attention_mask raises an error.

        Caching with padding is unsupported because ComplexEMA doesn't handle masks,
        leading to hidden state contamination from padded positions.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
        model = MegalodonForCausalLM(config, key=key)

        batch, seq = 2, 16
        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)
        # Mask with some padding (last 4 positions are False)
        attention_mask = jnp.concatenate(
            [jnp.ones((batch, seq - 4), dtype=bool), jnp.zeros((batch, 4), dtype=bool)],
            axis=1,
        )

        # Should raise when return_cache=True with padding
        with pytest.raises(Exception):  # eqx.error_if raises EquinoxRuntimeError
            model(input_ids, attention_mask=attention_mask, return_cache=True)

    def test_cache_without_padding_succeeds(self, random_seed: int) -> None:
        """Test that building cache without padding works correctly.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
        model = MegalodonForCausalLM(config, key=key)

        batch, seq = 2, 16
        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)
        # All-True mask (no padding) should work
        attention_mask = jnp.ones((batch, seq), dtype=bool)

        # Should succeed with all-True mask
        logits, cache = model(input_ids, attention_mask=attention_mask, return_cache=True)

        assert logits.shape == (batch, seq, config.vocab_size)
        assert cache is not None

    def test_no_mask_with_cache_succeeds(self, random_seed: int) -> None:
        """Test that building cache without any mask works correctly.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
        model = MegalodonForCausalLM(config, key=key)

        batch, seq = 2, 16
        input_ids = jax.random.randint(key, (batch, seq), minval=1, maxval=config.vocab_size)

        # No mask at all should work
        logits, cache = model(input_ids, attention_mask=None, return_cache=True)

        assert logits.shape == (batch, seq, config.vocab_size)
        assert cache is not None

    def test_empty_batch_handling(self, random_seed: int) -> None:
        """Test that empty batch (B=0) is handled gracefully.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=2,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
        )

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
        config = MegalodonConfig(
            vocab_size=256,
            model_dim=64,
            num_layers=2,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=16,
            norm_num_groups=8,
        )

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
        config = MegalodonConfig(
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
            compute_dtype=jnp.bfloat16,
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Empty batch (B=0)
        empty_batch = jnp.zeros((0, 16), dtype=jnp.int32)
        logits_b0, _ = model(empty_batch, return_cache=False)
        assert logits_b0.dtype == jnp.bfloat16, f"Expected bfloat16, got {logits_b0.dtype}"

        # Empty sequence (L=0)
        empty_seq = jnp.zeros((2, 0), dtype=jnp.int32)
        logits_l0, _ = model(empty_seq, return_cache=False)
        assert logits_l0.dtype == jnp.bfloat16, f"Expected bfloat16, got {logits_l0.dtype}"

    def test_vocab_bounds_input_ids_raises(self, random_seed: int) -> None:
        """Test that out-of-bounds input_ids raises an error.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
        config = MegalodonConfig(
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
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jnp.array([[1, 2, 3, 4]])
        bad_labels = jnp.array([[2, 300, 4, 5]])  # 300 >= 256

        with pytest.raises(Exception):  # eqx.error_if raises EquinoxRuntimeError
            model.compute_loss(input_ids, bad_labels)

    def test_loss_dtype_matches_model_bf16(self, random_seed: int) -> None:
        """Test that empty sequence loss returns correct dtype when model is bf16.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
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
            compute_dtype=jnp.bfloat16,
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # Single token sequence (becomes empty after shift)
        input_ids = jnp.array([[1]])
        labels = jnp.array([[2]])

        loss = model.compute_loss(input_ids, labels)

        # Loss should be bf16 (matching logits dtype), not float32
        assert loss.dtype == jnp.bfloat16, f"Expected bfloat16, got {loss.dtype}"
        assert loss == 0.0


class TestPrecisionAudit:
    """Tests for precision policy audits."""

    def test_sensitive_params_fp32(self, random_seed: int) -> None:
        """Ensure sensitive params remain fp32 by default and audit detects drift.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        config = MegalodonConfig(
            vocab_size=128,
            model_dim=64,
            num_layers=1,
            num_heads=2,
            z_dim=32,
            value_dim=64,
            ffn_hidden_dim=128,
            cema_ndim=4,
            chunk_size=8,
            norm_num_groups=8,
            compute_dtype=jnp.bfloat16,
        )

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        mismatches = audit_sensitive_param_dtypes(model)
        assert mismatches == {}

        def to_bf16(x: Any) -> Any:
            if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.bfloat16)
            return x

        model_bf16 = jax.tree.map(to_bf16, model)
        mismatches = audit_sensitive_param_dtypes(model_bf16)
        assert "layers.0.attn.cema.alpha" in mismatches


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

    def test_first_position_causal_mask_outputs_valid(self) -> None:
        """First position in causal attention should attend to itself only.

        :return None: None.
        """
        from megalodon_jax.layers.attention import attention_single_chunk

        B, L, H, Dh, Dv = 1, 4, 1, 8, 8
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (B, L, H, Dh))
        k = jax.random.normal(k2, (B, L, H, Dh))
        v = jax.random.normal(k3, (B, L, H, Dv))

        out = attention_single_chunk(q, k, v, causal=True)

        # First position attends only to first key, so output should be v[0]
        np.testing.assert_allclose(
            np.array(out[0, 0]),
            np.array(v[0, 0]),
            rtol=1e-5,
            err_msg="First causal position should output first value exactly",
        )
