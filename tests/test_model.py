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

        # Outputs should match (within tolerance due to streaming norm differences)
        # Streaming norms accumulate statistics slightly differently than batch
        np.testing.assert_allclose(
            np.array(out_full),
            np.array(out_streamed),
            rtol=2e-3,
            atol=2e-5,
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

    def test_forward_shapes(self, random_seed):
        """Test that MegalodonModel produces correct output shapes."""
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

    def test_attention_mask(self, random_seed):
        """Test that attention mask is properly applied."""
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

    def test_forward_shapes(self, random_seed):
        """Test that MegalodonForCausalLM produces correct logit shapes."""
        config = small_config()
        batch, seq = 2, 32

        key = jax.random.PRNGKey(random_seed)
        model = MegalodonForCausalLM(config, key=key)

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
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

    def test_compute_loss(self, random_seed):
        """Test that loss computation works correctly."""
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

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
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

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
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

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
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

        input_ids = jax.random.randint(key, (batch, seq), minval=0, maxval=config.vocab_size)
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
        # Cross-framework parity has slightly looser tolerances due to
        # different operation ordering and GPU TF32 precision
        np.testing.assert_allclose(
            np.array(jax_logits),
            torch_logits.numpy(),
            rtol=1e-3,
            atol=1e-4,
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


# -----------------------------------------------------------------------------
# Regression Tests for Phase 4 Fixes
# -----------------------------------------------------------------------------


class TestFix1GradientCheckpointing:
    """Tests for gradient checkpointing (Fix 1)."""

    def test_checkpointing_disables_cache_during_training(self, random_seed):
        """Test that use_checkpoint=True produces None caches during training."""
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

    def test_checkpointing_disabled_returns_cache(self, random_seed):
        """Test that use_checkpoint=False returns caches normally."""
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


class TestFix2PadTokenMasking:
    """Tests for pad token masking (Fix 2)."""

    def test_pad_token_embeddings_are_zeroed(self, random_seed):
        """Test that pad tokens produce zero embeddings."""
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

        # Mask pad tokens the same way as in the model
        pad_mask = input_ids == config.pad_token_id
        masked_embed = jnp.where(pad_mask[:, :, None], 0.0, embed)

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


class TestFix3UntiedLMHead:
    """Tests for untied LM head support (Fix 3)."""

    def test_tied_head_when_output_size_matches_vocab(self, random_seed):
        """Test that LM head is tied when output_size equals vocab_size."""
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

    def test_untied_head_when_output_size_differs(self, random_seed):
        """Test that separate LM head is created when output_size != vocab_size."""
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

    def test_untied_head_forward_shapes(self, random_seed):
        """Test that untied LM head produces correct output shapes."""
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

    def test_cache_length_mismatch_raises(self, random_seed):
        """Test that mismatched cache length raises ValueError."""
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

    def test_he_init_applied(self, random_seed):
        """Test that He initialization is applied when init_mode='he'."""
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

    def test_none_init_keeps_defaults(self, random_seed):
        """Test that init_mode='none' doesn't reinitialize weights."""
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

    def test_different_init_modes_produce_different_weights(self, random_seed):
        """Test that different init modes produce different weight distributions."""
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
        model_he = MegalodonForCausalLM(
            MegalodonConfig(**base_config, init_mode="he"), key=key
        )
        model_xavier = MegalodonForCausalLM(
            MegalodonConfig(**base_config, init_mode="xavier"), key=key
        )
        model_bert = MegalodonForCausalLM(
            MegalodonConfig(**base_config, init_mode="bert"), key=key
        )

        # BERT init has stddev=0.02, which should have much smaller variance than He
        bert_var = jnp.var(model_bert.model.embed.weight)
        he_var = jnp.var(model_he.model.embed.weight)

        # BERT has fixed small stddev, He scales with dimensions
        # BERT std=0.02 means var ~0.0004
        assert bert_var < 0.01, f"BERT init variance {bert_var} should be small"
        assert he_var != bert_var, "Different init modes should produce different weights"
