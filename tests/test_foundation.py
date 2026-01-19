"""Phase 1 Foundation tests - config, types, RMSNorm, RotaryEmbedding parity."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax import MegalodonConfig
from megalodon_jax.layers import RMSNorm, RotaryEmbedding
from megalodon_jax.types import AttentionCache, LayerCache, NormState
from tests.utils import require_torch_modeling, to_jax


class TestMegalodonConfig:
    """Tests for MegalodonConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values.

        :return None: None.
        """
        cfg = MegalodonConfig()
        assert cfg.vocab_size == 32_000
        assert cfg.model_dim == 1024
        assert cfg.num_layers == 12
        assert cfg.num_heads == 1
        assert cfg.z_dim == 256
        assert cfg.value_dim == 2048
        assert cfg.ffn_hidden_dim == 2560
        assert cfg.cema_ndim == 16
        assert cfg.chunk_size == 2048
        assert cfg.param_dtype == jnp.float32
        assert cfg.compute_dtype == jnp.float32
        assert cfg.accum_dtype == jnp.float32
        assert cfg.softmax_dtype == jnp.float32
        assert cfg.gemm_backend == "default"

    def test_head_dim_property(self) -> None:
        """Test head_dim computed property.

        :return None: None.
        """
        cfg = MegalodonConfig()
        assert cfg.head_dim == 256  # z_dim / num_heads

    def test_value_head_dim_property(self) -> None:
        """Test value_head_dim computed property.

        :return None: None.
        """
        cfg = MegalodonConfig()
        assert cfg.value_head_dim == 2048  # value_dim / num_heads

    def test_7b_preset(self) -> None:
        """Test 7B configuration preset.

        :return None: None.
        """
        cfg = MegalodonConfig.from_7b()
        assert cfg.model_dim == 4096
        assert cfg.num_layers == 32
        assert cfg.num_heads == 4
        assert cfg.z_dim == 1024
        assert cfg.value_dim == 8192
        assert cfg.ffn_hidden_dim == 11264
        assert cfg.chunk_size == 4096
        assert cfg.norm_num_groups == 64
        assert cfg.rope_base == 100_000.0
        assert cfg.swiglu is True

    def test_z_dim_divisibility_validation(self) -> None:
        """Test z_dim must be divisible by num_heads.

        :return None: None.
        """
        with pytest.raises(ValueError, match="z_dim.*divisible by num_heads"):
            MegalodonConfig(z_dim=255, num_heads=2)

    def test_value_dim_divisibility_validation(self) -> None:
        """Test value_dim must be divisible by num_heads.

        :return None: None.
        """
        with pytest.raises(ValueError, match="value_dim.*divisible by num_heads"):
            MegalodonConfig(value_dim=2047, num_heads=2)

    def test_model_dim_divisibility_validation(self) -> None:
        """Test model_dim must be divisible by norm_num_groups.

        :return None: None.
        """
        with pytest.raises(ValueError, match="model_dim.*divisible by.*norm_num_groups"):
            MegalodonConfig(model_dim=1000, norm_num_groups=32)

    def test_max_cache_len_validation(self) -> None:
        """Test max_cache_len must be positive and >= chunk_size when provided.

        :return None: None.
        """
        with pytest.raises(ValueError, match="max_cache_len.*positive"):
            MegalodonConfig(chunk_size=8, max_cache_len=0)
        with pytest.raises(ValueError, match="max_cache_len.*positive"):
            MegalodonConfig(chunk_size=8, max_cache_len=-1)
        with pytest.raises(ValueError, match="max_cache_len.*chunk_size"):
            MegalodonConfig(chunk_size=8, max_cache_len=4)

    def test_dtype_validation(self) -> None:
        """Test dtype constraints reject float16.

        :return None: None.
        """
        with pytest.raises(ValueError, match="float16"):
            MegalodonConfig(param_dtype=jnp.float16)
        with pytest.raises(ValueError, match="float16"):
            MegalodonConfig(compute_dtype=jnp.float16)
        with pytest.raises(ValueError, match="float16"):
            MegalodonConfig(accum_dtype=jnp.float16)
        with pytest.raises(ValueError, match="float16"):
            MegalodonConfig(softmax_dtype=jnp.float16)

    def test_gemm_backend_validation(self) -> None:
        """Test gemm_backend validation.

        :return None: None.
        """
        with pytest.raises(ValueError, match="gemm_backend"):
            MegalodonConfig(gemm_backend="unsupported")  # type: ignore[arg-type]

    def test_accum_dtype_precision_validation(self) -> None:
        """Test accum_dtype must be at least as wide as compute_dtype.

        :return None: None.
        """
        with pytest.raises(ValueError, match="accum_dtype should be >= compute_dtype"):
            MegalodonConfig(compute_dtype=jnp.float32, accum_dtype=jnp.bfloat16)

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable.

        :return None: None.
        """
        cfg = MegalodonConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.model_dim = 2048

    def test_config_is_hashable(self) -> None:
        """Test that config can be used as dict key.

        :return None: None.
        """
        cfg = MegalodonConfig()
        d = {cfg: "test"}
        assert d[cfg] == "test"


class TestRMSNormParity:
    """Parity tests for RMSNorm against PyTorch reference."""

    @pytest.mark.torch_ref
    def test_forward_parity(self, random_seed: int) -> None:
        """Test RMSNorm forward pass matches PyTorch.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        torch = pytest.importorskip("torch")
        torch_norm_cls = require_torch_modeling().RMSNorm

        dim = 64
        eps = 1e-6

        # Create both modules
        torch_norm = torch_norm_cls(dim, eps=eps)
        jax_norm = RMSNorm(dim, eps=eps)

        # Copy weights from PyTorch to JAX
        jax_norm = eqx.tree_at(lambda m: m.gamma, jax_norm, to_jax(torch_norm.gamma))

        # Generate test input
        x_torch = torch.randn(2, 16, dim)
        x_jax = to_jax(x_torch)

        # Forward pass
        y_torch = torch_norm(x_torch)
        y_jax = jax_norm(x_jax)

        # Compare
        np.testing.assert_allclose(np.array(y_jax), y_torch.detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_gamma_initialization(self) -> None:
        """Test that gamma is initialized to zeros.

        :return None: None.
        """
        norm = RMSNorm(64)
        np.testing.assert_array_equal(np.array(norm.gamma), np.zeros(64))

    def test_effective_scale_is_one(self) -> None:
        """Test that effective scale is 1.0 with zero gamma.

        :return None: None.
        """
        norm = RMSNorm(64)
        x = jnp.ones((2, 16, 64))
        y = norm(x)
        # With gamma=0, effective scale is (gamma+1)=1, so output should be normalized to 1
        # RMS of ones is 1.0, so output should be 1.0 * 1.0 = 1.0
        np.testing.assert_allclose(np.array(y), 1.0, rtol=1e-5)

    def test_different_shapes(self) -> None:
        """Test RMSNorm works with various input shapes.

        :return None: None.
        """
        norm = RMSNorm(128)
        for shape in [(1, 10, 128), (4, 32, 128), (2, 1, 128)]:
            x = jnp.ones(shape)
            y = norm(x)
            assert y.shape == shape


class TestRotaryEmbeddingParity:
    """Parity tests for RotaryEmbedding against PyTorch reference."""

    @pytest.mark.torch_ref
    def test_forward_parity(self, random_seed: int) -> None:
        """Test RotaryEmbedding forward pass matches PyTorch.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        torch = pytest.importorskip("torch")
        torch_rope_cls = require_torch_modeling().RotaryEmbedding

        dim = 64
        base = 10000.0

        # Create both modules
        torch_rope = torch_rope_cls(dim, base=base)
        jax_rope = RotaryEmbedding(dim, base=base)

        # Verify inv_freq matches
        np.testing.assert_allclose(
            np.array(jax_rope.inv_freq),
            torch_rope.inv_freq.numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

        # Generate test input
        batch, seq, heads = 2, 16, 4
        q_torch = torch.randn(batch, seq, heads, dim)
        k_torch = torch.randn(batch, seq, heads, dim)
        q_jax = to_jax(q_torch)
        k_jax = to_jax(k_torch)

        # Forward pass at position 0
        start_index = 0
        q_rot_torch, k_rot_torch = torch_rope(q_torch, k_torch, start_index=start_index)
        q_rot_jax, k_rot_jax = jax_rope(q_jax, k_jax, jnp.array(start_index, dtype=jnp.int32))

        # Compare
        np.testing.assert_allclose(
            np.array(q_rot_jax), q_rot_torch.detach().numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(k_rot_jax), k_rot_torch.detach().numpy(), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.torch_ref
    def test_different_start_positions(self, random_seed: int) -> None:
        """Test RotaryEmbedding at various start positions.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        torch = pytest.importorskip("torch")
        torch_rope_cls = require_torch_modeling().RotaryEmbedding

        dim = 64
        torch_rope = torch_rope_cls(dim)
        jax_rope = RotaryEmbedding(dim)

        batch, seq, heads = 2, 8, 4
        q_torch = torch.randn(batch, seq, heads, dim)
        k_torch = torch.randn(batch, seq, heads, dim)
        q_jax = to_jax(q_torch)
        k_jax = to_jax(k_torch)

        for start_index in [0, 10, 100, 1000]:
            q_rot_torch, k_rot_torch = torch_rope(q_torch, k_torch, start_index=start_index)
            q_rot_jax, k_rot_jax = jax_rope(q_jax, k_jax, jnp.array(start_index, dtype=jnp.int32))

            # Tolerance scales with position due to exp() implementation differences
            # between JAX (XLA) and PyTorch. These are within float32 precision bounds.
            # Error analysis shows: pos 0-10: ~1e-6, pos 100: ~1e-5, pos 1000: ~1e-4
            if start_index >= 1000:
                rtol, atol = 1e-4, 1e-4
            elif start_index >= 100:
                rtol, atol = 5e-5, 5e-5
            else:
                rtol, atol = 1e-5, 1e-5

            np.testing.assert_allclose(
                np.array(q_rot_jax),
                q_rot_torch.detach().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"Q mismatch at start_index={start_index}",
            )
            np.testing.assert_allclose(
                np.array(k_rot_jax),
                k_rot_torch.detach().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"K mismatch at start_index={start_index}",
            )

    def test_even_dim_required(self) -> None:
        """Test that odd dimension raises ValueError.

        :return None: None.
        """
        with pytest.raises(ValueError, match="even head dimension"):
            RotaryEmbedding(63)

    @pytest.mark.torch_ref
    def test_different_base_values(self, random_seed: int) -> None:
        """Test RotaryEmbedding with different base values.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        torch = pytest.importorskip("torch")
        torch_rope_cls = require_torch_modeling().RotaryEmbedding

        dim = 64
        batch, seq, heads = 2, 8, 4
        q_torch = torch.randn(batch, seq, heads, dim)
        k_torch = torch.randn(batch, seq, heads, dim)
        q_jax = to_jax(q_torch)
        k_jax = to_jax(k_torch)

        for base in [10000.0, 100000.0]:
            torch_rope = torch_rope_cls(dim, base=base)
            jax_rope = RotaryEmbedding(dim, base=base)

            q_rot_torch, k_rot_torch = torch_rope(q_torch, k_torch, start_index=0)
            q_rot_jax, k_rot_jax = jax_rope(q_jax, k_jax, jnp.array(0, dtype=jnp.int32))

            np.testing.assert_allclose(
                np.array(q_rot_jax),
                q_rot_torch.detach().numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Mismatch at base={base}",
            )


class TestCacheTypes:
    """Tests for cache/state type definitions."""

    def test_attention_cache_structure(self) -> None:
        """Test AttentionCache fields and structure.

        :return None: None.
        """
        k = jnp.zeros((2, 16, 4, 64))
        v = jnp.zeros((2, 16, 4, 128))
        count = jnp.array(16, dtype=jnp.int32)

        cache = AttentionCache(k=k, v=v, count=count)
        # Check field values
        assert cache.k.shape == (2, 16, 4, 64)
        assert cache.v.shape == (2, 16, 4, 128)
        assert cache.count == 16
        # Buffer capacity available via shape (not properties - see types.py docstring)
        assert cache.k.shape[1] == 16

    def test_layer_cache_default_position(self) -> None:
        """Test LayerCache has JAX scalar position by default.

        :return None: None.
        """
        cache = LayerCache()
        assert cache.position.dtype == jnp.int32
        assert cache.position.shape == ()
        assert cache.position == 0

    def test_jax_scalar_counters_jit_compatible(self) -> None:
        """Test that JAX scalar counters work in JIT without recompilation.

        :return None: None.
        """

        @jax.jit
        def increment_position(cache: LayerCache) -> LayerCache:
            """Increment the layer cache position counter.

            :param LayerCache cache: Cache with position counter to advance.
            :return LayerCache: Updated cache with incremented position.
            """
            return LayerCache(
                attn=cache.attn,
                norm=cache.norm,
                ema=cache.ema,
                position=cache.position + 1,
            )

        cache = LayerCache()
        for i in range(5):
            cache = increment_position(cache)
            assert cache.position == i + 1

    def test_norm_state_shapes(self) -> None:
        """Test NormState shape handling.

        :return None: None.
        """
        count = jnp.zeros((4,), dtype=jnp.int32)
        mean = jnp.zeros((4, 32))
        var = jnp.ones((4, 32))

        state = NormState(count=count, mean=mean, var=var)
        assert state.count.shape == (4,)
        assert state.mean.shape == (4, 32)
        assert state.var.shape == (4, 32)
