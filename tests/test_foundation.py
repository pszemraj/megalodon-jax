"""Phase 1 Foundation tests - config, types, RMSNorm, RotaryEmbedding parity."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax import MegalodonConfig, MegalodonForCausalLM
from megalodon_jax.layers import RMSNorm, RotaryEmbedding
from megalodon_jax.layers.norms import BatchedLayerNorm
from megalodon_jax.types import LayerCache


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
        assert cfg.attention_softmax_dtype == jnp.float32
        assert cfg.loss_softmax_dtype == jnp.float32
        assert cfg.init_mode == "he"
        assert cfg.share_emb is False

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

    def test_exact_named_presets_and_counts(self) -> None:
        """Released and paper presets retain distinct identities and exact counts."""
        presets = {
            "mega200m": (
                MegalodonConfig.from_upstream_mega200m(vocab_size=32_000),
                220_627_968,
            ),
            "mega1_3b": (
                MegalodonConfig.from_upstream_mega1_3b(vocab_size=32_000),
                1_342_832_640,
            ),
            "mega1_3b_pg19": (
                MegalodonConfig.from_upstream_mega1_3b_pg19(vocab_size=32_000),
                1_327_628_288,
            ),
            "mega7_1b": (
                MegalodonConfig.from_upstream_mega7_1b(vocab_size=32_000),
                7_117_381_632,
            ),
            "mega7_3b": (
                MegalodonConfig.from_upstream_mega7_3b(vocab_size=32_000),
                7_385_817_088,
            ),
            "paper_7b": (MegalodonConfig.from_paper_7b(), 7_385_817_088),
        }
        for name, (config, expected) in presets.items():
            assert config.parameter_count_breakdown()["total"] == expected, name

        assert presets["mega7_1b"][0].swiglu is False
        assert presets["mega7_1b"][0].ffn_hidden_dim == 11_264
        assert presets["mega7_3b"][0].swiglu is True
        assert presets["mega7_3b"][0].ffn_hidden_dim == 8_192
        assert presets["paper_7b"][0].chunk_size == 4_096
        assert presets["paper_7b"][0].effective_rope_base == 100_000.0
        assert presets["mega1_3b_pg19"][0].share_emb is True

    @pytest.mark.parametrize("share_emb", [False, True])
    @pytest.mark.parametrize("swiglu", [False, True])
    def test_parameter_count_matches_allocated_trainable_leaves(
        self,
        share_emb: bool,
        swiglu: bool,
    ) -> None:
        """The independent count formula matches every allocated inexact array leaf."""
        config = MegalodonConfig(
            vocab_size=17,
            model_dim=8,
            num_layers=2,
            num_heads=2,
            z_dim=8,
            value_dim=8,
            ffn_hidden_dim=12,
            cema_ndim=2,
            chunk_size=8,
            norm_num_groups=2,
            share_emb=share_emb,
            swiglu=swiglu,
            rescale_nffn=True,
        )
        model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(0))
        trainable = eqx.filter(model, eqx.is_inexact_array)
        allocated = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(trainable))
        assert allocated == config.parameter_count_breakdown()["total"]

    def test_ambiguous_7b_factory_is_rejected(self) -> None:
        """The historical hybrid factory must never select semantics silently."""
        with pytest.raises(ValueError, match="invalid hybrid"):
            MegalodonConfig.from_7b()

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

    def test_positive_head_and_chunk_validation(self) -> None:
        """Invalid structural values fail at config construction with clear errors."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            MegalodonConfig(num_heads=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            MegalodonConfig(chunk_size=0)

    def test_model_dim_divisibility_validation(self) -> None:
        """Test model_dim must be divisible by norm_num_groups.

        :return None: None.
        """
        with pytest.raises(ValueError, match="model_dim.*divisible by.*norm_num_groups"):
            MegalodonConfig(model_dim=1000, norm_num_groups=32)

    def test_attention_window_validation(self) -> None:
        """Sliding-window width is optional and positive when present.

        :return None: None.
        """
        assert MegalodonConfig(chunk_size=8).cache_capacity == 8
        assert MegalodonConfig(chunk_size=8, attention_window=4).cache_capacity == 4
        with pytest.raises(ValueError, match="attention_window.*positive"):
            MegalodonConfig(chunk_size=8, attention_window=0)
        with pytest.raises(ValueError, match="attention_window.*positive"):
            MegalodonConfig(chunk_size=8, attention_window=-1)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("model_dim", 0, "model_dim must be positive"),
            ("z_dim", 0, "z_dim must be positive"),
            ("value_dim", 0, "value_dim must be positive"),
            ("ffn_hidden_dim", 0, "ffn_hidden_dim must be positive"),
            ("cema_ndim", 0, "cema_ndim must be positive"),
            ("norm_num_groups", 0, "norm_num_groups must be positive"),
            ("num_layers", -1, "num_layers must be non-negative"),
            ("z_dim", 3, "head_dim must be even"),
            ("norm_eps", float("nan"), "norm_eps must be finite and positive"),
            ("norm_eps", float("inf"), "norm_eps must be finite and positive"),
            ("rope_base", 0.0, "rope_base must be finite and positive"),
            ("rope_base", float("nan"), "rope_base must be finite and positive"),
            ("rope_base", float("inf"), "rope_base must be finite and positive"),
            ("bos_token_id", 32_000, "bos_token_id must be"),
            ("eos_token_id", -1, "eos_token_id must be"),
        ],
    )
    def test_structural_and_finite_validation(
        self, field: str, value: int | float, message: str
    ) -> None:
        """Invalid structures and non-finite numerics fail at the config boundary."""
        with pytest.raises(ValueError, match=message):
            MegalodonConfig(**{field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("chunk_size", 4.5),
            ("attention_window", float("nan")),
            ("model_dim", 1024.0),
            ("num_layers", 1.5),
            ("num_layers", True),
            ("vocab_size", 32_000.0),
            ("output_size", 3.5),
            ("norm_num_groups", 32.0),
            ("bos_token_id", 1.0),
        ],
    )
    def test_structural_fields_require_integer_types(self, field: str, value: object) -> None:
        """Shape, count, and token-ID fields reject coercible non-integers."""
        with pytest.raises(ValueError, match=rf"{field} must be an integer"):
            MegalodonConfig(**{field: value})

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
            MegalodonConfig(attention_softmax_dtype=jnp.float16)
        with pytest.raises(ValueError, match="float16"):
            MegalodonConfig(loss_softmax_dtype=jnp.float16)
        with pytest.raises(ValueError, match="param_dtype must be float32"):
            MegalodonConfig(param_dtype=jnp.bfloat16)

    def test_accum_dtype_precision_validation(self) -> None:
        """Test accum_dtype must be at least as wide as compute_dtype.

        :return None: None.
        """
        with pytest.raises(ValueError, match="accum_dtype must be float32"):
            MegalodonConfig(compute_dtype=jnp.float32, accum_dtype=jnp.bfloat16)

    def test_dropout_and_fresh_init_validation(self) -> None:
        """Degenerate dropout and loader-only initialization are rejected."""
        for field in ("dropout", "attention_dropout", "hidden_dropout"):
            with pytest.raises(ValueError, match=r"\[0, 1\)"):
                MegalodonConfig(**{field: 1.0})
        with pytest.raises(ValueError, match="fresh-model init_mode"):
            MegalodonConfig(init_mode="none")  # type: ignore[arg-type]

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


class TestRMSNorm:
    """Mathematical and parameterization tests for RMSNorm."""

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


class TestLayerNormStorage:
    """Tests for released plus-one LayerNorm parameter storage."""

    def test_zero_stored_weight_has_identity_effective_scale(self) -> None:
        """Serialized zero gamma must produce ordinary LayerNorm output."""
        norm = BatchedLayerNorm(4, eps=1e-5)
        np.testing.assert_array_equal(np.asarray(norm.weight), np.zeros(4))

        x = jnp.asarray([[[1.0, 2.0, 4.0, 8.0]]], dtype=jnp.float32)
        output = norm(x)
        x_np = np.asarray(x)
        mean = x_np.mean(axis=-1, keepdims=True)
        var = np.square(x_np - mean).mean(axis=-1, keepdims=True)
        expected = (x_np - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(np.asarray(output), expected, atol=2e-6, rtol=2e-6)


class TestRotaryEmbeddingParity:
    """Parity tests for RotaryEmbedding against PyTorch reference."""

    def test_adjacent_pair_oracle(self) -> None:
        """RoPE rotates adjacent coordinates as released upstream does."""
        rope = RotaryEmbedding(8, base=10_000.0)
        q = jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 1, 1, 8)
        k = q + 10.0
        q_actual, k_actual = rope(q, k, jnp.asarray(7, dtype=jnp.int32))

        pairs = np.asarray(q).reshape(1, 1, 1, 4, 2)
        inv = np.power(10_000.0, -np.arange(4, dtype=np.float32) / 4)
        angles = 7.0 * inv
        expected_pairs = np.stack(
            [
                pairs[..., 0] * np.cos(angles) - pairs[..., 1] * np.sin(angles),
                pairs[..., 1] * np.cos(angles) + pairs[..., 0] * np.sin(angles),
            ],
            axis=-1,
        )
        np.testing.assert_allclose(np.asarray(q_actual), expected_pairs.reshape(q.shape), atol=2e-6)
        assert k_actual.shape == k.shape

    def test_frequencies_are_derived_not_array_leaves(self) -> None:
        """Fixed rotary data must not enter generic Equinox optimizers."""
        rope = RotaryEmbedding(8, base=10_000.0)
        assert jax.tree_util.tree_leaves(rope) == []

    def test_explicit_positions_match_start_offset(self) -> None:
        """Explicit positions and a scalar cache offset use the same phases."""
        rope = RotaryEmbedding(8)
        q = jnp.arange(48, dtype=jnp.float32).reshape(2, 3, 1, 8)
        k = q + 1.0
        offset_q, offset_k = rope(q, k, jnp.asarray(5, dtype=jnp.int32))
        positions = jnp.broadcast_to(jnp.arange(5, 8, dtype=jnp.int32), (2, 3))
        explicit_q, explicit_k = rope(
            q,
            k,
            jnp.asarray(0, dtype=jnp.int32),
            position_ids=positions,
        )
        np.testing.assert_array_equal(np.asarray(offset_q), np.asarray(explicit_q))
        np.testing.assert_array_equal(np.asarray(offset_k), np.asarray(explicit_k))

    def test_even_dim_required(self) -> None:
        """Test that odd dimension raises ValueError.

        :return None: None.
        """
        with pytest.raises(ValueError, match="even head dimension"):
            RotaryEmbedding(63)


class TestCacheTypes:
    """Tests for cache/state type definitions."""

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
