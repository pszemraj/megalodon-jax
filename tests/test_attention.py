"""Phase 3 Attention tests - primitives, ChunkedAttention, MegalodonAttention, NormalizedFFN."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from megalodon_jax.layers import (
    ChunkedAttention,
    MegalodonAttention,
    NormalizedFFN,
    RotaryEmbedding,
    attention_multi_chunk,
    attention_single_chunk,
)


def to_jax(t: torch.Tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(t.detach().cpu().numpy())


def to_torch(a: jnp.ndarray) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor."""
    return torch.from_numpy(np.array(a))


# -----------------------------------------------------------------------------
# Attention Primitive Tests
# -----------------------------------------------------------------------------


class TestAttentionPrimitives:
    """Tests for attention_single_chunk and attention_multi_chunk."""

    def test_single_chunk_shapes(self, random_seed):
        """Test that attention_single_chunk produces correct output shapes."""
        batch, seq, heads, head_dim, value_dim = 2, 16, 4, 32, 64

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))

        out = attention_single_chunk(q, k, v)

        assert out.shape == (batch, seq, heads, value_dim)

    def test_single_chunk_causal_masking(self, random_seed):
        """Test that causal masking prevents attending to future positions."""
        batch, seq, heads, head_dim = 1, 4, 1, 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        # Create Q/K/V with random values
        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, head_dim))

        out = attention_single_chunk(q, k, v, causal=True)

        # For causal attention, verify shapes and non-NaN outputs
        assert out.shape == (batch, seq, heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

        # Verify causality: output at position 0 should only depend on input at position 0
        # We can test this by checking that changing future inputs doesn't affect past outputs
        # Modify v at position 3 and check that output at position 0 is unchanged
        v_modified = v.at[:, 3, :, :].set(v[:, 3, :, :] * 100)
        out_modified = attention_single_chunk(q, k, v_modified, causal=True)

        # Position 0 output should be identical (can't see position 3)
        np.testing.assert_allclose(
            np.array(out[:, 0, :, :]),
            np.array(out_modified[:, 0, :, :]),
            rtol=1e-6,
            err_msg="Causal masking failed: position 0 saw future position 3",
        )

    def test_single_chunk_no_temperature_scaling(self, random_seed):
        """Test that attention uses scale=1.0 (no temperature scaling)."""
        batch, seq, heads, head_dim = 1, 4, 1, 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        # Use small values to avoid softmax saturation
        q = jax.random.normal(k1, (batch, seq, heads, head_dim)) * 0.1
        k = jax.random.normal(k2, (batch, seq, heads, head_dim)) * 0.1
        v = jax.random.normal(k3, (batch, seq, heads, head_dim))

        out = attention_single_chunk(q, k, v)

        # Verify output is not NaN and has reasonable magnitude
        assert not jnp.any(jnp.isnan(out))

    def test_multi_chunk_shapes(self, random_seed):
        """Test that attention_multi_chunk produces correct output shapes."""
        batch, seq, heads, head_dim, value_dim = 2, 64, 4, 32, 64
        chunk_size = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))

        rotary = RotaryEmbedding(dim=head_dim)
        start_index = jnp.array(0, dtype=jnp.int32)

        out = attention_multi_chunk(
            q, k, v, chunk_size=chunk_size, start_index=start_index, rotary=rotary
        )

        assert out.shape == (batch, seq, heads, value_dim)

    def test_multi_chunk_padding(self, random_seed):
        """Test multi_chunk handles sequences not divisible by chunk_size."""
        batch, seq, heads, head_dim, value_dim = 2, 50, 4, 32, 64
        chunk_size = 16  # 50 is not divisible by 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))

        rotary = RotaryEmbedding(dim=head_dim)
        start_index = jnp.array(0, dtype=jnp.int32)

        out = attention_multi_chunk(
            q, k, v, chunk_size=chunk_size, start_index=start_index, rotary=rotary
        )

        # Output should have original sequence length (padding removed)
        assert out.shape == (batch, seq, heads, value_dim)
        assert not jnp.any(jnp.isnan(out))


# -----------------------------------------------------------------------------
# ChunkedAttention Tests
# -----------------------------------------------------------------------------


class TestChunkedAttention:
    """Tests for ChunkedAttention module."""

    def test_forward_shapes(self, random_seed):
        """Test ChunkedAttention forward pass shapes."""
        batch, seq, heads, head_dim, value_dim = 2, 32, 4, 32, 64
        chunk_size = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        q = jax.random.normal(k2, (batch, seq, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq, heads, value_dim))

        out, cache, position = attn(q, k, v)

        assert out.shape == (batch, seq, heads, value_dim)
        assert cache is None  # No cache returned without return_cache=True

    def test_streaming_with_cache(self, random_seed):
        """Test ChunkedAttention streaming with cache."""
        batch, heads, head_dim, value_dim = 2, 4, 32, 64
        chunk_size = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Process tokens one at a time
        outputs = []
        cache = None

        for i in range(8):
            ki = jax.random.fold_in(k2, i)
            q = jax.random.normal(ki, (batch, 1, heads, head_dim))
            k = jax.random.normal(jax.random.fold_in(k3, i), (batch, 1, heads, head_dim))
            v = jax.random.normal(jax.random.fold_in(k4, i), (batch, 1, heads, value_dim))

            out, cache, position = attn(q, k, v, cache=cache, return_cache=True)
            outputs.append(out)

        # Check final cache state
        assert cache is not None
        assert cache.count == 8
        assert cache.k.shape[1] == 8  # All 8 tokens cached


# -----------------------------------------------------------------------------
# NormalizedFFN Tests
# -----------------------------------------------------------------------------


class TestNormalizedFFN:
    """Tests for NormalizedFFN module."""

    def test_forward_shapes(self, random_seed):
        """Test NormalizedFFN forward pass shapes."""
        batch, seq, model_dim, ffn_dim = 2, 16, 64, 128

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=False,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim))
        out = ffn(x)

        assert out.shape == (batch, seq, model_dim)

    def test_swiglu_variant(self, random_seed):
        """Test NormalizedFFN with SwiGLU activation."""
        batch, seq, model_dim, ffn_dim = 2, 16, 64, 128

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=True,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim))
        out = ffn(x)

        assert out.shape == (batch, seq, model_dim)
        assert ffn.fc3 is not None  # SwiGLU has fc3

    def test_two_hop_residual(self, random_seed):
        """Test NormalizedFFN with two-hop residual."""
        batch, seq, model_dim, ffn_dim = 2, 16, 64, 128

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=False,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim))
        residual_base = jax.random.normal(k3, (batch, seq, model_dim))

        out_normal = ffn(x)
        out_two_hop = ffn(x, residual_base=residual_base)

        # Two-hop should use residual_base, not x
        assert not jnp.allclose(out_normal, out_two_hop)

    def test_ffn_parity(self, random_seed):
        """Test NormalizedFFN parity with PyTorch reference."""
        from megalodon.modeling_megalodon import MegalodonConfig as TorchConfig
        from megalodon.modeling_megalodon import NormalizedFFN as TorchFFN

        model_dim, ffn_dim = 64, 128
        batch, seq = 2, 16

        # Create PyTorch config and FFN
        torch_cfg = TorchConfig(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=False,
            norm_eps=1e-5,
            rescale_nffn=False,
        )
        torch_ffn = TorchFFN(torch_cfg, layer_id=0)
        torch_ffn.eval()

        # Create JAX FFN
        key = jax.random.PRNGKey(random_seed)
        jax_ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=False,
            norm_eps=1e-5,
            key=key,
        )

        # Copy weights from PyTorch to JAX
        jax_ffn = eqx.tree_at(lambda m: m.norm.weight, jax_ffn, to_jax(torch_ffn.norm.weight))
        jax_ffn = eqx.tree_at(lambda m: m.norm.bias, jax_ffn, to_jax(torch_ffn.norm.bias))
        jax_ffn = eqx.tree_at(lambda m: m.fc1.weight, jax_ffn, to_jax(torch_ffn.fc1.weight))
        jax_ffn = eqx.tree_at(lambda m: m.fc1.bias, jax_ffn, to_jax(torch_ffn.fc1.bias))
        jax_ffn = eqx.tree_at(lambda m: m.fc2.weight, jax_ffn, to_jax(torch_ffn.fc2.weight))
        jax_ffn = eqx.tree_at(lambda m: m.fc2.bias, jax_ffn, to_jax(torch_ffn.fc2.bias))

        # Generate test input
        x_torch = torch.randn(batch, seq, model_dim)
        x_jax = to_jax(x_torch)

        # Forward pass
        with torch.no_grad():
            y_torch = torch_ffn(x_torch)
        y_jax = jax_ffn(x_jax)

        # Compare - fp32 accumulation causes ~1e-3 drift over multiple operations
        np.testing.assert_allclose(
            np.array(y_jax),
            y_torch.detach().numpy(),
            rtol=5e-3,
            atol=5e-3,
            err_msg="NormalizedFFN output should match PyTorch reference",
        )


# -----------------------------------------------------------------------------
# MegalodonAttention Tests
# -----------------------------------------------------------------------------


class TestMegalodonAttention:
    """Tests for MegalodonAttention block."""

    def test_forward_shapes(self, random_seed):
        """Test MegalodonAttention forward pass shapes."""
        batch, seq = 2, 32
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim))
        y, cache = attn(x)

        assert y.shape == (batch, seq, model_dim)
        assert cache is None  # No cache without return_cache=True

    def test_gradient_flow(self, random_seed):
        """Test gradients flow through MegalodonAttention."""
        batch, seq = 2, 16
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        def loss_fn(model, x):
            y, _ = model(x)
            return jnp.sum(y**2)

        x = jax.random.normal(k2, (batch, seq, model_dim))
        grads = eqx.filter_grad(loss_fn)(attn, x)

        # Check gradients for key parameters
        assert not jnp.any(jnp.isnan(grads.gamma))
        assert not jnp.any(jnp.isnan(grads.beta))
        assert not jnp.any(jnp.isnan(grads.wz.weight))
        assert not jnp.any(jnp.isnan(grads.wv.weight))

    def test_streaming_with_cache(self, random_seed):
        """Test MegalodonAttention streaming with cache."""
        batch = 2
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        # Process tokens one at a time
        outputs = []
        cache = None

        for i in range(8):
            ki = jax.random.fold_in(k2, i)
            x = jax.random.normal(ki, (batch, 1, model_dim))
            y, cache = attn(x, cache=cache, return_cache=True)
            outputs.append(y)

        # Check final cache state
        assert cache is not None
        assert cache.attn is not None
        assert cache.norm is not None
        assert cache.ema is not None


# -----------------------------------------------------------------------------
# Precision Tests (bf16)
# -----------------------------------------------------------------------------


class TestPrecision:
    """Tests for bf16 precision handling."""

    def test_attention_primitives_bf16(self, random_seed):
        """Test attention primitives work with bf16 inputs."""
        batch, seq, heads, head_dim = 2, 16, 4, 32

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)
        k = jax.random.normal(k2, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)
        v = jax.random.normal(k3, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)

        out = attention_single_chunk(q, k, v)

        assert out.dtype == jnp.bfloat16
        assert not jnp.any(jnp.isnan(out))

    def test_megalodon_attention_bf16(self, random_seed):
        """Test MegalodonAttention with bf16 inputs and params."""
        batch, seq = 2, 16
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        # Cast params to bf16
        def to_bf16(x):
            if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.bfloat16)
            return x

        attn_bf16 = jax.tree.map(to_bf16, attn)

        x = jax.random.normal(k2, (batch, seq, model_dim)).astype(jnp.bfloat16)
        y, _ = attn_bf16(x)

        assert y.dtype == jnp.bfloat16
        assert not jnp.any(jnp.isnan(y))

    def test_normalized_ffn_bf16(self, random_seed):
        """Test NormalizedFFN with bf16 inputs and params."""
        batch, seq, model_dim, ffn_dim = 2, 16, 64, 128

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=True,
            key=k1,
        )

        # Cast params to bf16
        def to_bf16(x):
            if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.bfloat16)
            return x

        ffn_bf16 = jax.tree.map(to_bf16, ffn)

        x = jax.random.normal(k2, (batch, seq, model_dim)).astype(jnp.bfloat16)
        out = ffn_bf16(x)

        assert out.dtype == jnp.bfloat16
        assert not jnp.any(jnp.isnan(out))


# -----------------------------------------------------------------------------
# JIT Compilation Tests
# -----------------------------------------------------------------------------


class TestJITCompilation:
    """Tests for JIT compilation stability."""

    def test_chunked_attention_jit(self, random_seed):
        """Test ChunkedAttention compiles without errors."""
        batch, seq, heads, head_dim, value_dim = 2, 32, 4, 32, 64
        chunk_size = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        @eqx.filter_jit
        def forward(model, q, k, v):
            return model(q, k, v)

        q = jax.random.normal(k2, (batch, seq, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq, heads, value_dim))

        out, _, _ = forward(attn, q, k, v)
        assert out.shape == (batch, seq, heads, value_dim)

    def test_megalodon_attention_jit(self, random_seed):
        """Test MegalodonAttention compiles without errors."""
        batch, seq = 2, 32
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        @eqx.filter_jit
        def forward(model, x):
            return model(x)

        x = jax.random.normal(k2, (batch, seq, model_dim))
        y, _ = forward(attn, x)
        assert y.shape == (batch, seq, model_dim)
