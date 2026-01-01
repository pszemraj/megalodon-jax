"""Phase 4 Model Assembly tests - MegalodonBlock, MegalodonModel, MegalodonForCausalLM."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from megalodon_jax import (
    MegalodonBlock,
    MegalodonConfig,
    MegalodonForCausalLM,
    MegalodonModel,
    ModelCache,
)
from megalodon_jax.convert import load_weights_from_torch


def to_jax(t: torch.Tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(t.detach().cpu().numpy())


def to_torch(a: jnp.ndarray) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor."""
    return torch.from_numpy(np.array(a))


def small_config() -> MegalodonConfig:
    """Create a small config for fast testing."""
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

    def test_forward_shapes(self, random_seed):
        """Test that MegalodonBlock produces correct output shapes."""
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

    def test_forward_no_cache(self, random_seed):
        """Test forward pass without returning cache."""
        config = small_config()
        batch, seq = 2, 32

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        block = MegalodonBlock(config, layer_id=0, key=k1)
        x = jax.random.normal(k2, (batch, seq, config.model_dim))

        out, cache = block(x, return_cache=False)

        assert out.shape == (batch, seq, config.model_dim)
        assert cache is None

    def test_streaming_with_cache(self, random_seed):
        """Test that streaming with cache produces consistent outputs."""
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

        # Outputs should match (within tolerance due to streaming norms)
        np.testing.assert_allclose(
            np.array(out_full),
            np.array(out_streamed),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Streaming output differs from batch output",
        )

    def test_different_layer_ids(self, random_seed):
        """Test that different layer IDs produce different rescaling."""
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
        batch, seq = 1, 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        block0 = MegalodonBlock(config, layer_id=0, key=k1)
        block3 = MegalodonBlock(config, layer_id=3, key=k2)
        x = jax.random.normal(k3, (batch, seq, config.model_dim))

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

    def test_forward_shapes(self, random_seed):
        """Test that MegalodonModel produces correct output shapes."""
        config = small_config()
        batch, seq = 2, 32

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )
        hidden, cache = model(input_ids, return_cache=True)

        assert hidden.shape == (batch, seq, config.model_dim)
        assert cache is not None
        assert len(cache.layer_caches) == config.num_layers
        assert cache.final_norm is not None

    def test_embedding_scale(self, random_seed):
        """Test that embedding scaling works correctly."""
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

    def test_cache_threading(self, random_seed):
        """Test that cache is properly threaded through layers."""
        config = small_config()
        batch, seq = 1, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )

        # First forward - no cache
        _, cache1 = model(input_ids, return_cache=True)

        # Second forward - with cache
        next_ids = jax.random.randint(
            key, (batch, 1), minval=0, maxval=config.vocab_size
        )
        _, cache2 = model(next_ids, cache=cache1, return_cache=True)

        # Cache should be updated
        assert cache2 is not None
        # Position should have advanced
        for layer_cache in cache2.layer_caches:
            if layer_cache is not None and layer_cache.attn is not None:
                assert layer_cache.attn.count > 0

    def test_attention_mask(self, random_seed):
        """Test that attention mask is properly applied."""
        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonModel(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )

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

    def test_forward_shapes(self, random_seed):
        """Test that MegalodonForCausalLM produces correct logit shapes."""
        config = small_config()
        batch, seq = 2, 32

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )
        logits, cache = model(input_ids, return_cache=True)

        assert logits.shape == (batch, seq, config.vocab_size)
        assert cache is not None

    def test_weight_tying(self, random_seed):
        """Test that LM head is tied to input embeddings."""
        config = small_config()

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        # The LM head should use embed weights transposed
        # Verify by checking that logits = hidden @ embed.weight.T
        batch, seq = 1, 8
        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )

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

    def test_compute_loss(self, random_seed):
        """Test that loss computation works correctly."""
        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )
        labels = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )

        loss = model.compute_loss(input_ids, labels)

        # Loss should be a scalar
        assert loss.shape == ()
        # Loss should be positive and finite
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_streaming_generation(self, random_seed):
        """Test streaming token-by-token generation."""
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

    def test_model_jit_compilation(self, random_seed):
        """Test that model compiles without errors."""
        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        @eqx.filter_jit
        def forward(model, input_ids):
            return model(input_ids, return_cache=False)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )

        # First call compiles
        logits1, _ = forward(model, input_ids)

        # Second call uses cached compilation
        logits2, _ = forward(model, input_ids)

        np.testing.assert_array_equal(
            np.array(logits1),
            np.array(logits2),
            err_msg="JIT compilation produced different results",
        )

    def test_no_recompilation_on_cache_update(self, random_seed):
        """Test that streaming with cache doesn't cause recompilation."""
        config = small_config()
        batch = 1

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        @eqx.filter_jit
        def step(model, input_ids, cache):
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

    def test_gradient_flow_all_params(self, random_seed):
        """Test that gradients flow to all trainable parameters."""
        config = small_config()
        batch, seq = 2, 16

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )
        labels = input_ids  # Self-supervision

        def loss_fn(model):
            return model.compute_loss(input_ids, labels)

        # Compute gradients
        grads = eqx.filter_grad(loss_fn)(model)

        # Check that gradients exist and are finite
        def check_grad(x):
            if eqx.is_array(x):
                assert jnp.all(jnp.isfinite(x)), "Gradient contains NaN or Inf"
                return True
            return False

        # Apply check to all leaves
        grad_leaves = jax.tree_util.tree_leaves(
            eqx.filter(grads, eqx.is_array), is_leaf=eqx.is_array
        )
        assert len(grad_leaves) > 0, "No gradients computed"

        for leaf in grad_leaves:
            assert jnp.all(jnp.isfinite(leaf)), "Gradient contains NaN or Inf"

    def test_gradient_flow_embedding(self, random_seed):
        """Test that gradients flow to embedding weights."""
        config = small_config()
        batch, seq = 2, 8

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )
        labels = input_ids

        def loss_fn(model):
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

    def test_cache_is_pytree(self, random_seed):
        """Test that ModelCache is properly registered as a pytree."""
        config = small_config()
        batch, seq = 1, 8

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )
        _, cache = model(input_ids, return_cache=True)

        # Cache should be a valid pytree
        leaves = jax.tree_util.tree_leaves(cache)
        assert len(leaves) > 0

        # Should be able to map over cache
        def double(x):
            if isinstance(x, jnp.ndarray):
                return x * 2
            return x

        doubled = jax.tree_util.tree_map(double, cache)
        assert doubled is not None

    def test_cache_tuple_immutability(self, random_seed):
        """Test that layer_caches is a tuple (not list) for JAX compatibility."""
        config = small_config()
        batch, seq = 1, 8

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(
            key, (batch, seq), minval=0, maxval=config.vocab_size
        )
        _, cache = model(input_ids, return_cache=True)

        assert isinstance(cache.layer_caches, tuple)


# -----------------------------------------------------------------------------
# Parity Tests (PyTorch Reference)
# -----------------------------------------------------------------------------


class TestParity:
    """Tests comparing JAX implementation to PyTorch reference."""

    @pytest.fixture
    def parity_config(self):
        """Config for parity testing - matches PyTorch defaults."""
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

    def test_causal_lm_forward_parity(self, random_seed, parity_config, torch_device):
        """Test that JAX MegalodonForCausalLM matches PyTorch reference."""
        from megalodon import MegalodonConfig as TorchConfig
        from megalodon import MegalodonForCausalLM as TorchMegalodonForCausalLM

        config = parity_config
        batch, seq = 2, 32

        # Create PyTorch model
        torch_config = TorchConfig(
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
        torch_model = TorchMegalodonForCausalLM(torch_config).eval()

        # Create JAX model
        key = jax.random.PRNGKey(random_seed)
        jax_model = MegalodonForCausalLM(config, key=key)

        # Copy weights from PyTorch to JAX
        jax_model = load_weights_from_torch(jax_model, torch_model.state_dict())

        # Create test input
        torch.manual_seed(random_seed)
        input_ids_torch = torch.randint(0, config.vocab_size, (batch, seq))
        input_ids_jax = to_jax(input_ids_torch)

        # Forward pass
        with torch.no_grad():
            torch_out = torch_model(input_ids_torch, use_cache=False, return_dict=True)
            torch_logits = torch_out.logits

        jax_logits, _ = jax_model(input_ids_jax, return_cache=False)

        # Compare outputs
        np.testing.assert_allclose(
            np.array(jax_logits),
            torch_logits.numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="JAX logits differ from PyTorch reference",
        )

    def test_streaming_parity(self, random_seed, parity_config, torch_device):
        """Test that JAX streaming matches PyTorch streaming."""
        from megalodon import MegalodonConfig as TorchConfig
        from megalodon import MegalodonForCausalLM as TorchMegalodonForCausalLM

        config = parity_config
        batch = 1
        prompt_len = 16
        gen_len = 4

        # Create PyTorch model
        torch_config = TorchConfig(
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
        torch_model = TorchMegalodonForCausalLM(torch_config).eval()

        # Create JAX model and load weights
        key = jax.random.PRNGKey(random_seed)
        jax_model = MegalodonForCausalLM(config, key=key)
        jax_model = load_weights_from_torch(jax_model, torch_model.state_dict())

        # Create prompt
        torch.manual_seed(random_seed)
        prompt_torch = torch.randint(0, config.vocab_size, (batch, prompt_len))
        prompt_jax = to_jax(prompt_torch)

        # PyTorch streaming generation
        torch_generated = []
        with torch.no_grad():
            torch_out = torch_model(prompt_torch, use_cache=True, return_dict=True)
            torch_pkv = torch_out.past_key_values
            current_torch = prompt_torch

            for _ in range(gen_len):
                next_token = torch_out.logits[:, -1:].argmax(dim=-1)
                torch_generated.append(next_token.item())
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
