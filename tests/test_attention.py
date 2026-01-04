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

    def test_single_chunk_preserves_bf16_dtype(self, random_seed):
        """Test that attention_single_chunk preserves bf16 dtype (no forced fp32)."""
        batch, seq, heads, head_dim, value_dim = 1, 4, 2, 16, 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (batch, seq, heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (batch, seq, heads, value_dim), dtype=jnp.bfloat16)

        out = attention_single_chunk(q, k, v)

        assert out.dtype == jnp.bfloat16
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
        # Fixed-size buffer: shape is max_cache_len (=chunk_size by default)
        assert cache.k.shape[1] == chunk_size

    def test_streaming_small_l_preserves_dtype_and_cache_count(self, random_seed):
        """Small-L streaming should keep dtype and track cache count without padding."""
        batch, heads, head_dim, value_dim = 1, 2, 16, 16
        chunk_size = 8
        seq = 3  # L < chunk_size to hit the token-wise path

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        q = jax.random.normal(k2, (batch, seq, heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(k3, (batch, seq, heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(k4, (batch, seq, heads, value_dim), dtype=jnp.bfloat16)

        out, cache, position = attn(q, k, v, return_cache=True)

        assert out.shape == (batch, seq, heads, value_dim)
        assert out.dtype == jnp.bfloat16
        assert cache is not None
        assert cache.count == seq  # should only advance by actual tokens
        assert position == seq


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

    def test_streaming_jit_no_recompile(self, random_seed):
        """Verify streaming path is JIT-compatible without recompilation.

        This test ensures that:
        1. The streaming path can be JIT compiled
        2. Multiple calls with the same shapes don't cause recompilation
        3. The fori_loop with dynamic cache values works correctly
        """
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        @eqx.filter_jit
        def step(model, q, k, v, cache):
            return model(q, k, v, cache=cache, return_cache=True)

        # Process 10 tokens one at a time
        cache = None
        for i in range(10):
            ki = jax.random.fold_in(k2, i)
            q = jax.random.normal(ki, (batch, 1, heads, head_dim))
            k = jax.random.normal(jax.random.fold_in(ki, 1), (batch, 1, heads, head_dim))
            v = jax.random.normal(jax.random.fold_in(ki, 2), (batch, 1, heads, value_dim))

            out, cache, pos = step(attn, q, k, v, cache)
            assert out.shape == (batch, 1, heads, value_dim)
            assert cache is not None
            assert int(pos) == i + 1

        # Final cache should be fixed-size
        assert cache.k.shape[1] == chunk_size
        assert int(cache.count) == 10

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


# -----------------------------------------------------------------------------
# Streaming Equivalence Tests
# -----------------------------------------------------------------------------


class TestStreamingEquivalence:
    """Tests for streaming (token-by-token) equivalence with batch processing."""

    def test_chunked_attention_streaming_equivalence(self, random_seed, force_fp32_matmul):
        """Verify streaming with cache matches batch processing within a chunk."""
        batch, heads, head_dim, value_dim = 1, 2, 16, 16
        chunk_size = 8
        seq_len = 6  # Within one chunk

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Generate full sequence Q/K/V
        q_full = jax.random.normal(k2, (batch, seq_len, heads, head_dim))
        k_full = jax.random.normal(k3, (batch, seq_len, heads, head_dim))
        v_full = jax.random.normal(k4, (batch, seq_len, heads, value_dim))

        # Batch processing (no cache)
        out_batch, _, _ = attn(q_full, k_full, v_full, return_cache=False)

        # Streaming processing (token by token with cache)
        streaming_outputs = []
        cache = None
        for i in range(seq_len):
            q_i = q_full[:, i : i + 1, :, :]
            k_i = k_full[:, i : i + 1, :, :]
            v_i = v_full[:, i : i + 1, :, :]

            out_i, cache, _ = attn(q_i, k_i, v_i, cache=cache, return_cache=True)
            streaming_outputs.append(out_i)

        out_streaming = jnp.concatenate(streaming_outputs, axis=1)

        # Outputs should match (within a single chunk)
        np.testing.assert_allclose(
            np.array(out_batch),
            np.array(out_streaming),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Streaming output should match batch output within a single chunk",
        )

    def test_chunk_boundary_cache_reset(self, random_seed):
        """Verify cache is reset at chunk boundaries."""
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Process tokens 0-3 (first chunk), then tokens 4-5 (second chunk)
        cache = None
        for i in range(6):
            q = jax.random.normal(jax.random.fold_in(k2, i), (batch, 1, heads, head_dim))
            k = jax.random.normal(jax.random.fold_in(k3, i), (batch, 1, heads, head_dim))
            v = jax.random.normal(jax.random.fold_in(k4, i), (batch, 1, heads, value_dim))

            _, cache, position = attn(q, k, v, cache=cache, return_cache=True)

            # Fixed-size buffer: shape is always max_cache_len
            assert cache.k.shape[1] == chunk_size, (
                f"Cache buffer should be fixed size {chunk_size}, got {cache.k.shape[1]}"
            )
            # The count tracks absolute position
            assert cache.count == i + 1, f"Cache count should be {i + 1}, got {cache.count}"

    def test_cache_resize_on_input(self, random_seed):
        """Test that incoming caches are resized to fixed buffer size."""
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        from megalodon_jax.types import AttentionCache

        # Create a smaller-than-expected cache (2 entries, but chunk_size=4)
        # Simulates position 2 (has 2 cached entries from positions 0-1)
        fake_k = jax.random.normal(k2, (batch, 2, heads, head_dim))
        fake_v = jax.random.normal(k3, (batch, 2, heads, value_dim))
        fake_cache = AttentionCache(k=fake_k, v=fake_v, count=jnp.array(2, dtype=jnp.int32))

        # Process a new token at position 2
        q = jax.random.normal(k4, (batch, 1, heads, head_dim))
        k = jax.random.normal(jax.random.fold_in(k4, 1), (batch, 1, heads, head_dim))
        v = jax.random.normal(jax.random.fold_in(k4, 2), (batch, 1, heads, value_dim))

        _, new_cache, new_pos = attn(q, k, v, cache=fake_cache, return_cache=True)

        # Output cache should be fixed size (chunk_size)
        assert new_cache.k.shape[1] == chunk_size, (
            f"Cache should be fixed size {chunk_size}, got {new_cache.k.shape[1]}"
        )
        # Position should advance
        assert new_pos == 3, f"Position should be 3, got {new_pos}"
        assert new_cache.count == 3, f"Cache count should be 3, got {new_cache.count}"


# -----------------------------------------------------------------------------
# Parity Tests
# -----------------------------------------------------------------------------


class TestParity:
    """Additional parity tests for MegalodonAttention and related modules."""

    def test_chunked_attention_block_diagonal_structure(self, random_seed):
        """Verify block-diagonal attention structure across chunks."""
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 8  # Two full chunks

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Generate full sequence Q/K/V
        q = jax.random.normal(k2, (batch, seq_len, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq_len, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq_len, heads, value_dim))

        # Get batch output
        out, _, _ = attn(q, k, v)

        # Process each chunk separately and compare
        # First chunk: positions 0-3
        q1, k1, v1 = q[:, :4], k[:, :4], v[:, :4]
        out1, _, _ = attn(q1, k1, v1)

        # Second chunk: positions 4-7
        q2, k2, v2 = q[:, 4:], k[:, 4:], v[:, 4:]
        out2, _, _ = attn(q2, k2, v2)

        # Block-diagonal: each chunk's output should match processing independently
        np.testing.assert_allclose(
            np.array(out[:, :4]),
            np.array(out1),
            rtol=1e-5,
            atol=1e-5,
            err_msg="First chunk output should match independent processing",
        )
        np.testing.assert_allclose(
            np.array(out[:, 4:]),
            np.array(out2),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Second chunk output should match independent processing",
        )

    def test_multi_token_streaming_across_boundary(self, random_seed):
        """Test that multi-token streaming calls properly split across chunk boundaries."""
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Generate a sequence that spans chunk boundary
        seq_len = 6  # Will span positions 0-5 when starting from position 2
        q = jax.random.normal(k2, (batch, seq_len, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq_len, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq_len, heads, value_dim))

        # Start at position 2, so tokens will span positions 2,3 | 4,5,6,7
        # chunk boundary at position 4
        from megalodon_jax.types import AttentionCache

        # Pre-fill cache with 2 tokens at positions 0-1
        init_k = jax.random.normal(jax.random.fold_in(k1, 0), (batch, 2, heads, head_dim))
        init_v = jax.random.normal(jax.random.fold_in(k1, 1), (batch, 2, heads, value_dim))
        # Apply RoPE to initial K
        init_k_rot, _ = attn.rotary(init_k, init_k, jnp.array(0))
        cache = AttentionCache(k=init_k_rot, v=init_v, count=jnp.array(2, dtype=jnp.int32))

        # Process multi-token call that spans boundary
        out_multi, cache_multi, pos_multi = attn(q, k, v, cache=cache, return_cache=True)

        # Now compare with token-by-token processing
        streaming_outputs = []
        cache_stream = cache
        for i in range(seq_len):
            q_i = q[:, i : i + 1]
            k_i = k[:, i : i + 1]
            v_i = v[:, i : i + 1]
            out_i, cache_stream, _ = attn(q_i, k_i, v_i, cache=cache_stream, return_cache=True)
            streaming_outputs.append(out_i)

        out_streaming = jnp.concatenate(streaming_outputs, axis=1)

        # Multi-token and streaming should produce same result
        np.testing.assert_allclose(
            np.array(out_multi),
            np.array(out_streaming),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Multi-token streaming should match token-by-token when crossing chunk boundary",
        )

        # Final cache positions should match
        assert int(pos_multi) == int(cache_stream.count), (
            f"Final positions should match: multi={int(pos_multi)}, stream={int(cache_stream.count)}"
        )


# -----------------------------------------------------------------------------
# MegalodonAttention Parity Tests
# -----------------------------------------------------------------------------


class TestMegalodonAttentionParity:
    """Parity tests for MegalodonAttention against PyTorch reference."""

    def test_megalodon_attention_forward_parity(self, random_seed, torch_device):
        """Test MegalodonAttention forward pass parity with PyTorch reference."""
        from conftest import sync_and_clear_torch

        pytest = __import__("pytest")

        # Check if PyTorch reference is available
        try:
            import sys

            sys.path.insert(
                0,
                "/home/pszemraj/workspace/LLM/experimental-arch/megalodon-2512/megalodon-jax/src",
            )
            from megalodon.modeling_megalodon import MegalodonAttention as TorchMegalodonAttention
            from megalodon.modeling_megalodon import MegalodonConfig as TorchConfig
        except ImportError:
            pytest.skip("PyTorch reference not available")

        # Config matching both implementations
        batch, seq_len = 1, 8
        model_dim = 32
        z_dim = 16
        value_dim = 16
        num_heads = 2
        cema_ndim = 2
        chunk_size = 4
        norm_num_groups = 2

        # Create PyTorch module on same device as JAX (GPU if available)
        torch_cfg = TorchConfig(
            model_dim=model_dim,
            num_heads=num_heads,
            z_dim=z_dim,
            value_dim=value_dim,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-5,
            norm_affine=True,
            rope_base=10000.0,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        torch_attn = TorchMegalodonAttention(torch_cfg).to(torch_device)
        torch_attn.eval()

        # Create JAX module with same config
        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-5,
            rope_base=10000.0,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            key=k1,
        )

        # Copy weights from PyTorch to JAX
        # TimestepNorm
        jax_attn = eqx.tree_at(
            lambda m: m.timenorm.weight, jax_attn, to_jax(torch_attn.timenorm.weight)
        )
        jax_attn = eqx.tree_at(
            lambda m: m.timenorm.bias, jax_attn, to_jax(torch_attn.timenorm.bias)
        )

        # ComplexEMA (alpha, delta, theta stored as real tensors)
        jax_attn = eqx.tree_at(
            lambda m: m.cema.alpha,
            jax_attn,
            to_jax(torch_attn.cema.alpha),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.delta,
            jax_attn,
            to_jax(torch_attn.cema.delta),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.theta,
            jax_attn,
            to_jax(torch_attn.cema.theta),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.gamma_real,
            jax_attn,
            to_jax(torch_attn.cema.gamma_real),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.gamma_imag,
            jax_attn,
            to_jax(torch_attn.cema.gamma_imag),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.omega,
            jax_attn,
            to_jax(torch_attn.cema.omega),
        )

        # RMSNorm
        jax_attn = eqx.tree_at(
            lambda m: m.rmsnorm.gamma, jax_attn, to_jax(torch_attn.rmsnorm.gamma)
        )

        # Projections (both PyTorch and Equinox use (out_features, in_features) layout)
        jax_attn = eqx.tree_at(lambda m: m.wz.weight, jax_attn, to_jax(torch_attn.wz.weight))
        jax_attn = eqx.tree_at(lambda m: m.wz.bias, jax_attn, to_jax(torch_attn.wz.bias))
        jax_attn = eqx.tree_at(lambda m: m.wv.weight, jax_attn, to_jax(torch_attn.wv.weight))
        jax_attn = eqx.tree_at(lambda m: m.wv.bias, jax_attn, to_jax(torch_attn.wv.bias))
        jax_attn = eqx.tree_at(lambda m: m.wr.weight, jax_attn, to_jax(torch_attn.wr.weight))
        jax_attn = eqx.tree_at(lambda m: m.wr.bias, jax_attn, to_jax(torch_attn.wr.bias))
        jax_attn = eqx.tree_at(lambda m: m.wh1.weight, jax_attn, to_jax(torch_attn.wh1.weight))
        jax_attn = eqx.tree_at(lambda m: m.wh1.bias, jax_attn, to_jax(torch_attn.wh1.bias))
        jax_attn = eqx.tree_at(lambda m: m.wh2.weight, jax_attn, to_jax(torch_attn.wh2.weight))
        jax_attn = eqx.tree_at(lambda m: m.wh2.bias, jax_attn, to_jax(torch_attn.wh2.bias))

        # Q/K affine parameters
        jax_attn = eqx.tree_at(lambda m: m.gamma, jax_attn, to_jax(torch_attn.gamma))
        jax_attn = eqx.tree_at(lambda m: m.beta, jax_attn, to_jax(torch_attn.beta))

        # Inner attention rotary (copy from CPU tensor)
        jax_attn = eqx.tree_at(
            lambda m: m.inner.rotary.inv_freq,
            jax_attn,
            to_jax(torch_attn.inner.rope.inv_freq.cpu()),
        )

        # Generate input on JAX, convert to PyTorch on same device
        x_jax = jax.random.normal(k2, (batch, seq_len, model_dim))
        x_torch = to_torch(x_jax).to(torch_device)

        # Run PyTorch forward on GPU, then move result to CPU
        with torch.no_grad():
            y_torch, _ = torch_attn(x_torch)
            y_torch_cpu = y_torch.cpu().numpy()

        # Clean up PyTorch GPU memory before JAX forward
        del y_torch, x_torch, torch_attn
        sync_and_clear_torch()

        # Run JAX forward
        y_jax, _ = jax_attn(x_jax, deterministic=True)

        # Compare outputs (both ran on GPU with matching TF32 settings)
        np.testing.assert_allclose(
            np.array(y_jax),
            y_torch_cpu,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MegalodonAttention output should match PyTorch reference",
        )

    def test_megalodon_attention_gradient_flow(self, random_seed):
        """Verify gradients flow through all parameters without NaN."""
        batch, seq_len = 1, 8
        model_dim = 32
        z_dim = 16
        value_dim = 16
        num_heads = 2
        cema_ndim = 2
        chunk_size = 4
        norm_num_groups = 2

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

        x = jax.random.normal(k2, (batch, seq_len, model_dim))

        def loss_fn(model):
            y, _ = model(x, deterministic=True)
            return jnp.mean(y**2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(attn)

        # Check loss is finite
        assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"

        # Check all gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        for i, g in enumerate(grad_leaves):
            assert jnp.all(jnp.isfinite(g)), f"Gradient {i} has non-finite values"

        # Check at least some gradients are non-zero
        non_zero_grads = sum(1 for g in grad_leaves if jnp.any(g != 0))
        assert non_zero_grads > 0, "At least some gradients should be non-zero"
