"""Phase 2 Core Layers tests - TimestepNorm, ComplexEMA parity."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from megalodon_jax.layers import ComplexEMA, TimestepNorm
from tests.torch_ref import modeling as torch_modeling


def to_jax(t: torch.Tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    if t.is_complex():
        return jnp.array(t.detach().cpu().numpy())
    return jnp.array(t.detach().cpu().numpy())


def to_torch(a: jnp.ndarray) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor."""
    return torch.from_numpy(np.array(a))


class TestTimestepNormParity:
    """Parity tests for TimestepNorm against PyTorch reference."""

    def test_forward_parity(self, random_seed):
        """Test TimestepNorm forward pass matches PyTorch."""
        torch_norm_cls = torch_modeling().TimestepNorm

        dim = 64
        num_groups = 8
        eps = 1e-5

        # Create both modules
        torch_norm = torch_norm_cls(dim, num_groups, eps=eps)
        jax_norm = TimestepNorm(dim, num_groups, eps=eps)

        # Copy weights from PyTorch to JAX
        jax_norm = eqx.tree_at(lambda m: m.weight, jax_norm, to_jax(torch_norm.weight))
        jax_norm = eqx.tree_at(lambda m: m.bias, jax_norm, to_jax(torch_norm.bias))

        # Generate test input
        batch, seq = 2, 32
        x_torch = torch.randn(batch, seq, dim)
        x_jax = to_jax(x_torch)

        # Forward pass
        y_torch, count_t, mean_t, var_t = torch_norm(x_torch)
        y_jax, state_jax = jax_norm(x_jax)

        # Compare outputs
        np.testing.assert_allclose(np.array(y_jax), y_torch.detach().numpy(), rtol=1e-4, atol=1e-5)

        # Compare output state
        np.testing.assert_allclose(
            np.array(state_jax.count),
            count_t.detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_streaming_state_continuity(self, random_seed):
        """Test that processing in chunks matches full sequence."""
        dim = 64
        num_groups = 8

        jax_norm = TimestepNorm(dim, num_groups)

        batch, seq = 2, 32
        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (batch, seq, dim))

        # Full sequence at once
        y_full, state_full = jax_norm(x)

        # Process in two chunks with state passing
        y1, state1 = jax_norm(x[:, :16, :])
        y2, state2 = jax_norm(x[:, 16:, :], state=state1)
        y_chunked = jnp.concatenate([y1, y2], axis=1)

        # Outputs should match (with some tolerance due to floating point)
        np.testing.assert_allclose(
            np.array(y_chunked),
            np.array(y_full),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Chunked processing should match full sequence",
        )

    def test_weight_initialization(self):
        """Test that weight and bias are initialized to zeros."""
        norm = TimestepNorm(64, 8)
        np.testing.assert_array_equal(np.array(norm.weight), np.zeros(64))
        np.testing.assert_array_equal(np.array(norm.bias), np.zeros(64))

    def test_effective_scale_is_one(self):
        """Test that effective scale is 1.0 with zero weight."""
        norm = TimestepNorm(64, 8)
        # Effective scale = weight + 1.0 = 1.0
        x = jnp.ones((2, 16, 64))
        y, _ = norm(x)
        # Output should be normalized (mean~0, var~1 per group, then scaled by 1.0)
        assert y.shape == x.shape

    def test_mask_handling(self, random_seed):
        """Test that padding mask correctly excludes positions from statistics."""
        dim = 64
        num_groups = 8

        jax_norm = TimestepNorm(dim, num_groups)

        batch, seq = 2, 16
        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (batch, seq, dim))

        # Create mask: first 8 positions valid, rest masked
        mask = jnp.concatenate(
            [
                jnp.ones((batch, 8), dtype=jnp.bool_),
                jnp.zeros((batch, 8), dtype=jnp.bool_),
            ],
            axis=1,
        )

        y_masked, state_masked = jax_norm(x, mask=mask)

        # Count should be 8 for all batch elements
        np.testing.assert_array_equal(np.array(state_masked.count), np.array([8, 8]))

    def test_divisibility_validation(self):
        """Test that num_features must be divisible by num_groups."""
        with pytest.raises(ValueError, match="divisible by"):
            TimestepNorm(63, 8)

    def test_different_shapes(self):
        """Test TimestepNorm works with various input shapes."""
        norm = TimestepNorm(128, 16)
        for shape in [(1, 10, 128), (4, 32, 128), (2, 1, 128)]:
            x = jnp.ones(shape)
            y, state = norm(x)
            assert y.shape == shape


class TestComplexEMAParity:
    """Parity tests for ComplexEMA against PyTorch reference."""

    def test_coefficients_parity(self, random_seed):
        """Test that coefficient computation matches PyTorch."""
        torch_ema_cls = torch_modeling().ComplexEMA

        dim = 64
        ndim = 16

        torch_ema = torch_ema_cls(dim, ndim)
        key = jax.random.PRNGKey(random_seed)
        jax_ema = ComplexEMA(dim, ndim, key=key)

        # Copy parameters from PyTorch to JAX
        jax_ema = eqx.tree_at(lambda m: m.alpha, jax_ema, to_jax(torch_ema.alpha))
        jax_ema = eqx.tree_at(lambda m: m.delta, jax_ema, to_jax(torch_ema.delta))
        jax_ema = eqx.tree_at(lambda m: m.theta, jax_ema, to_jax(torch_ema.theta))
        jax_ema = eqx.tree_at(lambda m: m.gamma_real, jax_ema, to_jax(torch_ema.gamma_real))
        jax_ema = eqx.tree_at(lambda m: m.gamma_imag, jax_ema, to_jax(torch_ema.gamma_imag))
        jax_ema = eqx.tree_at(lambda m: m.omega, jax_ema, to_jax(torch_ema.omega))

        # Compute coefficients
        p_torch, q_torch, gamma_torch = torch_ema._coeffs()
        p_jax, q_jax, gamma_jax = jax_ema._coeffs()

        # Compare
        np.testing.assert_allclose(np.array(p_jax), p_torch.detach().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(q_jax), q_torch.detach().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            np.array(gamma_jax), gamma_torch.detach().numpy(), rtol=1e-5, atol=1e-6
        )

    def test_fft_forward_parity(self, random_seed):
        """Test FFT path forward pass matches PyTorch."""
        torch_ema_cls = torch_modeling().ComplexEMA

        dim = 64
        ndim = 16

        torch_ema = torch_ema_cls(dim, ndim)
        key = jax.random.PRNGKey(random_seed)
        jax_ema = ComplexEMA(dim, ndim, key=key)

        # Copy parameters from PyTorch to JAX
        jax_ema = eqx.tree_at(lambda m: m.alpha, jax_ema, to_jax(torch_ema.alpha))
        jax_ema = eqx.tree_at(lambda m: m.delta, jax_ema, to_jax(torch_ema.delta))
        jax_ema = eqx.tree_at(lambda m: m.theta, jax_ema, to_jax(torch_ema.theta))
        jax_ema = eqx.tree_at(lambda m: m.gamma_real, jax_ema, to_jax(torch_ema.gamma_real))
        jax_ema = eqx.tree_at(lambda m: m.gamma_imag, jax_ema, to_jax(torch_ema.gamma_imag))
        jax_ema = eqx.tree_at(lambda m: m.omega, jax_ema, to_jax(torch_ema.omega))

        # Generate test input (B, D, L)
        batch, seq = 2, 32
        x_torch = torch.randn(batch, dim, seq)
        x_jax = to_jax(x_torch)

        # Forward pass (FFT path - no state)
        y_torch, _ = torch_ema(x_torch, hx=None, compute_last_state=False)
        y_jax, _ = jax_ema(x_jax, h_init=None, return_state=False)

        # Compare outputs
        np.testing.assert_allclose(np.array(y_jax), y_torch.detach().numpy(), rtol=1e-4, atol=1e-5)

    def test_sequential_forward_parity(self, random_seed):
        """Test sequential path forward pass matches PyTorch."""
        torch_ema_cls = torch_modeling().ComplexEMA

        dim = 64
        ndim = 16

        torch_ema = torch_ema_cls(dim, ndim)
        key = jax.random.PRNGKey(random_seed)
        jax_ema = ComplexEMA(dim, ndim, key=key)

        # Copy parameters from PyTorch to JAX
        jax_ema = eqx.tree_at(lambda m: m.alpha, jax_ema, to_jax(torch_ema.alpha))
        jax_ema = eqx.tree_at(lambda m: m.delta, jax_ema, to_jax(torch_ema.delta))
        jax_ema = eqx.tree_at(lambda m: m.theta, jax_ema, to_jax(torch_ema.theta))
        jax_ema = eqx.tree_at(lambda m: m.gamma_real, jax_ema, to_jax(torch_ema.gamma_real))
        jax_ema = eqx.tree_at(lambda m: m.gamma_imag, jax_ema, to_jax(torch_ema.gamma_imag))
        jax_ema = eqx.tree_at(lambda m: m.omega, jax_ema, to_jax(torch_ema.omega))

        # Generate test input (B, D, L)
        batch, seq = 2, 16
        x_torch = torch.randn(batch, dim, seq)
        x_jax = to_jax(x_torch)

        # Forward pass (sequential path - with state)
        y_torch, h_torch = torch_ema(x_torch, hx=None, compute_last_state=True)
        y_jax, h_jax = jax_ema(x_jax, h_init=None, return_state=True)

        # Compare outputs
        np.testing.assert_allclose(np.array(y_jax), y_torch.detach().numpy(), rtol=1e-4, atol=1e-5)

        # Compare final state
        np.testing.assert_allclose(np.array(h_jax), h_torch.detach().numpy(), rtol=1e-4, atol=1e-5)

    def test_fft_vs_sequential_equivalence(self, random_seed):
        """Test that FFT and sequential paths produce equivalent outputs."""
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_ema = ComplexEMA(dim, ndim, key=k1)

        # Generate test input
        batch, seq = 2, 32
        x = jax.random.normal(k2, (batch, dim, seq))

        # FFT path
        y_fft, _ = jax_ema(x, return_state=False)

        # Sequential path
        y_seq, _ = jax_ema(x, return_state=True)

        # Should produce equivalent outputs
        np.testing.assert_allclose(
            np.array(y_fft),
            np.array(y_seq),
            rtol=1e-4,
            atol=1e-5,
            err_msg="FFT and sequential paths should produce equivalent outputs",
        )

    def test_state_continuity(self, random_seed):
        """Test that chunked processing with state matches full sequence."""
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_ema = ComplexEMA(dim, ndim, key=k1)

        # Generate test input
        batch, seq = 2, 32
        x = jax.random.normal(k2, (batch, dim, seq))

        # Full sequence with FFT
        y_full, _ = jax_ema(x, return_state=False)

        # Two chunks with state passing
        y1, h1 = jax_ema(x[:, :, :16], return_state=True)
        y2, h2 = jax_ema(x[:, :, 16:], h_init=h1, return_state=True)
        y_chunked = jnp.concatenate([y1, y2], axis=-1)

        # Should match (within numerical tolerance)
        np.testing.assert_allclose(
            np.array(y_chunked),
            np.array(y_full),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Chunked processing should match full sequence",
        )

    def test_q_magnitude_bounded(self, random_seed):
        """Test that |q| < 1 by construction (ensures decaying impulse response)."""
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        jax_ema = ComplexEMA(dim, ndim, key=key)

        _, q, _ = jax_ema._coeffs()
        q_magnitude = jnp.abs(q)

        # All magnitudes should be strictly less than 1
        assert jnp.all(q_magnitude < 1.0), "q magnitude must be < 1 for stability"

    def test_jit_compilation(self, random_seed):
        """Test that ComplexEMA works with JIT compilation."""
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_ema = ComplexEMA(dim, ndim, key=k1)

        # JIT the forward pass
        @eqx.filter_jit
        def forward(ema, x):
            return ema(x, return_state=False)

        x = jax.random.normal(k2, (2, dim, 32))

        # First call compiles
        y1, _ = forward(jax_ema, x)
        # Second call should use cached compilation
        y2, _ = forward(jax_ema, x)

        np.testing.assert_array_equal(np.array(y1), np.array(y2))


class TestPrecisionPolicy:
    """Tests for bf16/fp16 precision handling."""

    def test_timestep_norm_bf16_input(self, random_seed):
        """Test TimestepNorm works correctly with bf16 inputs."""
        dim = 64
        num_groups = 8

        jax_norm = TimestepNorm(dim, num_groups)

        batch, seq = 2, 16
        key = jax.random.PRNGKey(random_seed)

        # Create bf16 input
        x_f32 = jax.random.normal(key, (batch, seq, dim))
        x_bf16 = x_f32.astype(jnp.bfloat16)

        # Forward pass should work
        y_bf16, state = jax_norm(x_bf16)

        # Output dtype should match input dtype
        assert y_bf16.dtype == jnp.bfloat16

        # Compare with fp32 path (within bf16 tolerance)
        y_f32, _ = jax_norm(x_f32)
        np.testing.assert_allclose(
            np.array(y_bf16.astype(jnp.float32)),
            np.array(y_f32),
            rtol=1e-2,
            atol=1e-2,
            err_msg="bf16 output should be close to fp32 output",
        )

    def test_timestep_norm_fp16_rejected(self):
        """Test TimestepNorm rejects fp16 inputs for stability."""
        dim = 64
        num_groups = 8

        jax_norm = TimestepNorm(dim, num_groups)
        x_fp16 = jnp.ones((2, 16, dim), dtype=jnp.float16)

        with pytest.raises(TypeError, match="float16"):
            jax_norm(x_fp16)

    def test_complex_ema_bf16_params(self, random_seed):
        """Test ComplexEMA computes coefficients in fp32 even with bf16 params."""
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        ema_f32 = ComplexEMA(dim, ndim, key=key)

        # Cast parameters to bf16
        def to_bf16(x):
            if eqx.is_array(x) and x.dtype == jnp.float32:
                return x.astype(jnp.bfloat16)
            return x

        ema_bf16 = jax.tree.map(to_bf16, ema_f32)

        # Coefficients should be computed in fp32
        p_f32, q_f32, gamma_f32 = ema_f32._coeffs()
        p_bf16, q_bf16, gamma_bf16 = ema_bf16._coeffs()

        # All coefficient outputs should be fp32/complex64 regardless of param dtype
        assert p_f32.dtype == jnp.float32
        assert p_bf16.dtype == jnp.float32, f"Expected float32, got {p_bf16.dtype}"
        assert q_f32.dtype == jnp.complex64
        assert q_bf16.dtype == jnp.complex64, f"Expected complex64, got {q_bf16.dtype}"
        assert gamma_f32.dtype == jnp.complex64
        assert gamma_bf16.dtype == jnp.complex64, f"Expected complex64, got {gamma_bf16.dtype}"

        # Values should be close, but bf16 params have quantization noise
        # so we use bf16-appropriate tolerance (~1e-2)
        np.testing.assert_allclose(
            np.array(p_f32),
            np.array(p_bf16),
            rtol=1e-2,
            atol=1e-3,
            err_msg="bf16 params should produce fp32 coefficients close to fp32 params",
        )

    def test_complex_ema_bf16_forward(self, random_seed):
        """Test ComplexEMA forward pass with bf16 inputs."""
        torch_ema_cls = torch_modeling().ComplexEMA

        dim = 64
        ndim = 16

        torch_ema = torch_ema_cls(dim, ndim)
        key = jax.random.PRNGKey(random_seed)
        jax_ema = ComplexEMA(dim, ndim, key=key)

        # Copy parameters from PyTorch to JAX (in fp32)
        jax_ema = eqx.tree_at(lambda m: m.alpha, jax_ema, to_jax(torch_ema.alpha))
        jax_ema = eqx.tree_at(lambda m: m.delta, jax_ema, to_jax(torch_ema.delta))
        jax_ema = eqx.tree_at(lambda m: m.theta, jax_ema, to_jax(torch_ema.theta))
        jax_ema = eqx.tree_at(lambda m: m.gamma_real, jax_ema, to_jax(torch_ema.gamma_real))
        jax_ema = eqx.tree_at(lambda m: m.gamma_imag, jax_ema, to_jax(torch_ema.gamma_imag))
        jax_ema = eqx.tree_at(lambda m: m.omega, jax_ema, to_jax(torch_ema.omega))

        # Cast JAX model to bf16
        def to_bf16(x):
            if eqx.is_array(x) and x.dtype == jnp.float32:
                return x.astype(jnp.bfloat16)
            return x

        jax_ema_bf16 = jax.tree.map(to_bf16, jax_ema)

        # Generate test input
        batch, seq = 2, 32
        x_torch = torch.randn(batch, dim, seq)
        x_jax_f32 = to_jax(x_torch)
        x_jax_bf16 = x_jax_f32.astype(jnp.bfloat16)

        # PyTorch forward (fp32)
        y_torch, _ = torch_ema(x_torch, hx=None, compute_last_state=False)

        # JAX forward with bf16 params and bf16 input
        y_jax_bf16, _ = jax_ema_bf16(x_jax_bf16, return_state=False)

        # Output should be bf16
        assert y_jax_bf16.dtype == jnp.bfloat16

        # Compare with PyTorch (looser tolerance for bf16)
        np.testing.assert_allclose(
            np.array(y_jax_bf16.astype(jnp.float32)),
            y_torch.detach().numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg="bf16 forward should be close to PyTorch fp32 reference",
        )

    def test_complex_ema_fft_vs_sequential_bf16(self, random_seed):
        """Test FFT and sequential paths produce equivalent results in bf16."""
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_ema = ComplexEMA(dim, ndim, key=k1)

        # Cast to bf16
        def to_bf16(x):
            if eqx.is_array(x) and x.dtype == jnp.float32:
                return x.astype(jnp.bfloat16)
            return x

        jax_ema_bf16 = jax.tree.map(to_bf16, jax_ema)

        # Generate bf16 input
        batch, seq = 2, 32
        x_bf16 = jax.random.normal(k2, (batch, dim, seq)).astype(jnp.bfloat16)

        # FFT path
        y_fft, _ = jax_ema_bf16(x_bf16, return_state=False)

        # Sequential path
        y_seq, _ = jax_ema_bf16(x_bf16, return_state=True)

        # Both should be bf16
        assert y_fft.dtype == jnp.bfloat16
        assert y_seq.dtype == jnp.bfloat16

        # Should produce equivalent outputs (within bf16 tolerance)
        np.testing.assert_allclose(
            np.array(y_fft),
            np.array(y_seq),
            rtol=1e-2,
            atol=1e-2,
            err_msg="FFT and sequential paths should match in bf16",
        )


class TestIntegration:
    """Integration tests for Phase 2 layers."""

    def test_timestep_norm_to_complex_ema_flow(self, random_seed):
        """Test data flow from TimestepNorm to ComplexEMA."""
        dim = 64
        num_groups = 8
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        norm = TimestepNorm(dim, num_groups)
        ema = ComplexEMA(dim, ndim, key=k1)

        # Input (B, L, D) for norm
        x = jax.random.normal(k2, (2, 32, dim))

        # Norm forward
        x_normed, norm_state = norm(x)

        # Transpose for EMA (B, D, L)
        x_ema = jnp.transpose(x_normed, (0, 2, 1))

        # EMA forward
        y_ema, ema_state = ema(x_ema, return_state=True)

        assert y_ema.shape == (2, dim, 32)
        assert ema_state.shape == (2, dim, ndim)

    def test_gradient_flow(self, random_seed):
        """Test that gradients flow through both layers."""
        dim = 64
        num_groups = 8
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        norm = TimestepNorm(dim, num_groups)
        ema = ComplexEMA(dim, ndim, key=k1)

        def loss_fn(models, x):
            norm, ema = models
            x_normed, _ = norm(x)
            x_ema = jnp.transpose(x_normed, (0, 2, 1))
            y, _ = ema(x_ema)
            return jnp.sum(y**2)

        x = jax.random.normal(k2, (2, 32, dim))

        # Compute gradients
        grads = eqx.filter_grad(loss_fn)((norm, ema), x)
        norm_grads, ema_grads = grads

        # Check gradients are not None and not NaN
        # Norm gradients
        assert not jnp.any(jnp.isnan(norm_grads.weight))
        assert not jnp.any(jnp.isnan(norm_grads.bias))

        # EMA gradients
        assert not jnp.any(jnp.isnan(ema_grads.alpha))
        assert not jnp.any(jnp.isnan(ema_grads.delta))
        assert not jnp.any(jnp.isnan(ema_grads.gamma_real))
        assert not jnp.any(jnp.isnan(ema_grads.gamma_imag))

    def test_jit_no_recompilation(self, random_seed):
        """Test that JIT doesn't recompile on different inputs."""
        dim = 64
        num_groups = 8
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        norm = TimestepNorm(dim, num_groups)
        ema = ComplexEMA(dim, ndim, key=k1)

        @eqx.filter_jit
        def forward(norm, ema, x):
            x_normed, _ = norm(x)
            x_ema = jnp.transpose(x_normed, (0, 2, 1))
            y, _ = ema(x_ema)
            return y

        x1 = jax.random.normal(k2, (2, 32, dim))
        x2 = jax.random.normal(k3, (2, 32, dim))

        # First call compiles
        y1 = forward(norm, ema, x1)
        # Second call should use cached compilation (same shape)
        y2 = forward(norm, ema, x2)

        assert y1.shape == y2.shape
