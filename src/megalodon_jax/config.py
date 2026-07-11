# Copyright 2025 Peter Szemraj.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Megalodon configuration."""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

InitMode = Literal["gaussian", "xavier", "he", "bert"]
AttentionDropoutMode = Literal["post_softmax", "dropkey"]


@dataclass(frozen=True)
class MegalodonConfig:
    """Configuration for Megalodon model.

    This is a frozen dataclass (immutable and hashable) that defines all
    hyperparameters for the Megalodon architecture.
    The dtype policy separates parameter storage (param_dtype) from compute
    (compute_dtype), with explicit GEMM accumulation (accum_dtype) and
    softmax/loss dtypes for AMP-style behavior in JAX.
    """

    vocab_size: int = 32_000
    model_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 1
    z_dim: int = 256  # shared Q/K projection width
    value_dim: int = 2048
    ffn_hidden_dim: int = 2560
    cema_ndim: int = 16  # complex EMA orders per channel
    chunk_size: int = 2048
    attention_window: int | None = None  # None = released chunk-local; int = sliding extension
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    rope_base: float | None = None  # defaults to 10000.0 if None
    swiglu: bool = False
    rescale_nffn: bool = False
    scale_emb: bool = False
    share_emb: bool = False  # Explicit embedding/output tying; never inferred from shape
    norm_affine: bool = True  # RMSNorm and FFN LayerNorm only; TimestepNorm is always affine
    dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_dropout_mode: AttentionDropoutMode = "post_softmax"
    hidden_dropout: float = 0.0
    pad_token_id: int | None = None  # Metadata only; token IDs are never zero-masked implicitly
    bos_token_id: int = 1
    eos_token_id: int = 2
    init_mode: InitMode = "he"
    use_checkpoint: bool = False  # Enable gradient checkpointing (disables cache during training)
    # Segmented CEMA path for packed sequences: parallel associative scan
    # (fast, materializes (L, B, D, N) complex64 tensors) vs sequential scan
    # (10-60x slower on GPU, O(1) extra memory). Only affects segment_ids runs.
    use_associative_segment_scan: bool = True
    output_size: int = -1  # LM head width; -1 resolves to vocab_size
    param_dtype: jnp.dtype = jnp.float32  # Parameter storage dtype
    compute_dtype: jnp.dtype = jnp.float32  # Compute dtype for matmuls/activations
    accum_dtype: jnp.dtype = jnp.float32  # Accumulation dtype for GEMM/reductions
    attention_softmax_dtype: jnp.dtype = jnp.float32
    loss_softmax_dtype: jnp.dtype = jnp.float32

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.output_size < -1 or self.output_size == 0:
            raise ValueError(f"output_size must be -1 or positive, got {self.output_size}")
        if self.share_emb and self.effective_output_size != self.vocab_size:
            raise ValueError("share_emb requires output_size to resolve to vocab_size")
        if self.z_dim % self.num_heads != 0:
            raise ValueError(
                f"z_dim ({self.z_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.value_dim % self.num_heads != 0:
            raise ValueError(
                f"value_dim ({self.value_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.model_dim % self.norm_num_groups != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by "
                f"norm_num_groups ({self.norm_num_groups})"
            )
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
        if not 0.0 <= self.hidden_dropout < 1.0:
            raise ValueError(f"hidden_dropout must be in [0, 1), got {self.hidden_dropout}")
        if self.attention_dropout_mode not in ("post_softmax", "dropkey"):
            raise ValueError(
                "attention_dropout_mode must be 'post_softmax' or 'dropkey', got "
                f"{self.attention_dropout_mode!r}"
            )
        if self.init_mode not in ("gaussian", "xavier", "he", "bert"):
            raise ValueError(f"unsupported fresh-model init_mode: {self.init_mode!r}")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError(
                f"pad_token_id must be in [0, {self.vocab_size}) or None, got {self.pad_token_id}"
            )
        if self.attention_window is not None and self.attention_window <= 0:
            raise ValueError("attention_window must be positive when provided")
        for name, dtype in (
            ("param_dtype", self.param_dtype),
            ("compute_dtype", self.compute_dtype),
            ("accum_dtype", self.accum_dtype),
            ("attention_softmax_dtype", self.attention_softmax_dtype),
            ("loss_softmax_dtype", self.loss_softmax_dtype),
        ):
            if not jnp.issubdtype(dtype, jnp.floating):
                raise ValueError(f"{name} must be a floating dtype, got {dtype}")
            if dtype == jnp.float16:
                raise ValueError("float16 is unsupported; use float32 or bfloat16 instead.")
        if self.param_dtype != jnp.float32:
            raise ValueError("param_dtype must be float32")
        if self.compute_dtype not in (jnp.float32, jnp.bfloat16):
            raise ValueError("compute_dtype must be float32 or bfloat16")
        if self.accum_dtype != jnp.float32:
            raise ValueError("accum_dtype must be float32")
        for name, dtype in (
            ("attention_softmax_dtype", self.attention_softmax_dtype),
            ("loss_softmax_dtype", self.loss_softmax_dtype),
        ):
            if dtype not in (jnp.float32, jnp.bfloat16):
                raise ValueError(f"{name} must be float32 or bfloat16")

    @property
    def head_dim(self) -> int:
        """Per-head dimension for Q/K projection.

        :return int: z_dim divided by num_heads.
        """
        return self.z_dim // self.num_heads

    @property
    def value_head_dim(self) -> int:
        """Per-head dimension for value projection.

        :return int: value_dim divided by num_heads.
        """
        return self.value_dim // self.num_heads

    @property
    def effective_rope_base(self) -> float:
        """RoPE frequency base, defaulting to 10000.0 if not specified.

        :return float: rope_base if set, otherwise 10000.0.
        """
        return self.rope_base if self.rope_base is not None else 10000.0

    @property
    def cache_capacity(self) -> int:
        """Return the fixed KV ring capacity for the selected attention mode."""
        return self.chunk_size if self.attention_window is None else self.attention_window

    @property
    def effective_output_size(self) -> int:
        """Resolve the language-model output width.

        :return int: Vocabulary size when output_size is -1, otherwise output_size.
        """
        return self.vocab_size if self.output_size == -1 else self.output_size

    def parameter_count_breakdown(self) -> dict[str, int]:
        """Return the exact trainable parameter count for this configuration."""
        d = self.model_dim
        z = self.z_dim
        v = self.value_dim
        f = self.ffn_hidden_dim
        n = self.cema_ndim

        timestep_norm = 2 * d
        cema = 4 * d * n + 2 * d
        rmsnorm = d if self.norm_affine else 0
        attention_projections = (d * z + z) + (d * v + v) + (d * v + v) + (d * d + d) + (v * d)
        qk_affine = 4 * z
        ffn_norm = 2 * d if self.norm_affine else 0
        ffn = 2 * d * f + (d * f if self.swiglu else 0)
        ffn_rescale = d if self.rescale_nffn else 0
        per_layer = (
            timestep_norm
            + cema
            + rmsnorm
            + attention_projections
            + qk_affine
            + ffn_norm
            + ffn
            + ffn_rescale
        )
        embedding = self.vocab_size * d
        output_head = 0 if self.share_emb else self.effective_output_size * d
        layers = self.num_layers * per_layer
        final_norm = 2 * d
        return {
            "embedding": embedding,
            "timestep_norm_per_layer": timestep_norm,
            "cema_per_layer": cema,
            "rmsnorm_per_layer": rmsnorm,
            "attention_projections_per_layer": attention_projections,
            "qk_affine_per_layer": qk_affine,
            "ffn_norm_per_layer": ffn_norm,
            "ffn_per_layer": ffn,
            "ffn_rescale_per_layer": ffn_rescale,
            "per_layer": per_layer,
            "layers": layers,
            "final_norm": final_norm,
            "output_head": output_head,
            "total": embedding + layers + final_norm + output_head,
        }

    @classmethod
    def from_upstream_mega200m(cls, *, vocab_size: int, output_size: int = -1) -> "MegalodonConfig":
        """Create the released mega200M configuration."""
        return cls(
            vocab_size=vocab_size,
            output_size=output_size,
            num_layers=12,
            model_dim=1024,
            num_heads=1,
            z_dim=256,
            value_dim=2048,
            ffn_hidden_dim=2560,
            chunk_size=2048,
            norm_num_groups=32,
        )

    @classmethod
    def from_upstream_mega1_3b(cls, *, vocab_size: int, output_size: int = -1) -> "MegalodonConfig":
        """Create the released mega1.3B configuration."""
        return cls(
            vocab_size=vocab_size,
            output_size=output_size,
            num_layers=24,
            model_dim=2048,
            num_heads=2,
            z_dim=512,
            value_dim=4096,
            ffn_hidden_dim=4864,
            chunk_size=2048,
            norm_num_groups=64,
        )

    @classmethod
    def from_upstream_mega1_3b_pg19(
        cls, *, vocab_size: int, output_size: int = -1
    ) -> "MegalodonConfig":
        """Create the released tied PG-19 mega1.3B configuration."""
        return cls(
            vocab_size=vocab_size,
            output_size=output_size,
            num_layers=24,
            model_dim=2048,
            num_heads=2,
            z_dim=512,
            value_dim=4096,
            ffn_hidden_dim=3584,
            chunk_size=2048,
            norm_num_groups=64,
            swiglu=True,
            scale_emb=True,
            share_emb=True,
        )

    @classmethod
    def from_upstream_mega7_1b(cls, *, vocab_size: int, output_size: int = -1) -> "MegalodonConfig":
        """Create the released non-SwiGLU mega7.1B configuration."""
        return cls(
            vocab_size=vocab_size,
            output_size=output_size,
            num_layers=32,
            model_dim=4096,
            num_heads=4,
            z_dim=1024,
            value_dim=8192,
            ffn_hidden_dim=11264,
            chunk_size=2048,
            norm_num_groups=64,
        )

    @classmethod
    def from_upstream_mega7_3b(cls, *, vocab_size: int, output_size: int = -1) -> "MegalodonConfig":
        """Create the released SwiGLU mega7.3B configuration."""
        return cls(
            vocab_size=vocab_size,
            output_size=output_size,
            num_layers=32,
            model_dim=4096,
            num_heads=4,
            z_dim=1024,
            value_dim=8192,
            ffn_hidden_dim=8192,
            chunk_size=2048,
            norm_num_groups=64,
            swiglu=True,
        )

    @classmethod
    def from_paper_7b(cls, *, vocab_size: int = 32_000, output_size: int = -1) -> "MegalodonConfig":
        """Create the distinct 7B training configuration reported in the paper."""
        return cls(
            vocab_size=vocab_size,
            output_size=output_size,
            num_layers=32,
            model_dim=4096,
            num_heads=4,
            z_dim=1024,
            value_dim=8192,
            ffn_hidden_dim=8192,
            chunk_size=4096,
            norm_num_groups=64,
            rope_base=100_000.0,
            swiglu=True,
        )

    @classmethod
    def from_7b(cls) -> "MegalodonConfig":
        """Reject the historically ambiguous and incorrect 7B factory."""
        raise ValueError(
            "from_7b() was an invalid hybrid preset; choose from_paper_7b(), "
            "from_upstream_mega7_1b(), or from_upstream_mega7_3b() explicitly"
        )
