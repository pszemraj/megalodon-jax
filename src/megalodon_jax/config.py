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

InitMode = Literal["gaussian", "xavier", "he", "bert", "none"]
GemmBackend = Literal["default"]


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
    max_cache_len: int | None = None  # defaults to chunk_size if None; must be >= chunk_size
    cache_unbounded: bool = False
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    rope_base: float | None = None  # defaults to 10000.0 if None
    swiglu: bool = False
    rescale_nffn: bool = False
    scale_emb: bool = False
    norm_affine: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_positions: int = 1_000_000
    init_mode: InitMode = "gaussian"
    use_checkpoint: bool = False  # Enable gradient checkpointing (disables cache during training)
    output_size: int = -1  # LM head size; -1 ties to vocab_size
    param_dtype: jnp.dtype = jnp.float32  # Parameter storage dtype
    compute_dtype: jnp.dtype = jnp.float32  # Compute dtype for matmuls/activations
    accum_dtype: jnp.dtype = jnp.float32  # Accumulation dtype for GEMM/reductions
    softmax_dtype: jnp.dtype = jnp.float32  # Softmax/log-softmax dtype
    gemm_backend: GemmBackend = "default"

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
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
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1], got {self.attention_dropout}")
        if not 0.0 <= self.hidden_dropout <= 1.0:
            raise ValueError(f"hidden_dropout must be in [0, 1], got {self.hidden_dropout}")
        if self.max_cache_len is not None and self.max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive when provided.")
        if self.max_cache_len is not None and self.max_cache_len < self.chunk_size:
            raise ValueError("max_cache_len must be >= chunk_size to preserve causal attention.")
        for name, dtype in (
            ("param_dtype", self.param_dtype),
            ("compute_dtype", self.compute_dtype),
            ("accum_dtype", self.accum_dtype),
            ("softmax_dtype", self.softmax_dtype),
        ):
            if not jnp.issubdtype(dtype, jnp.floating):
                raise ValueError(f"{name} must be a floating dtype, got {dtype}")
            if dtype == jnp.float16:
                raise ValueError("float16 is unsupported; use float32 or bfloat16 instead.")
        if self.gemm_backend != "default":
            raise ValueError(
                f"gemm_backend must be 'default' until other backends are implemented, got "
                f"{self.gemm_backend}"
            )

    @property
    def head_dim(self) -> int:
        """Per-head dimension for Q/K projection."""
        return self.z_dim // self.num_heads

    @property
    def value_head_dim(self) -> int:
        """Per-head dimension for value projection."""
        return self.value_dim // self.num_heads

    @property
    def effective_rope_base(self) -> float:
        """RoPE frequency base, defaulting to 10000.0 if not specified."""
        return self.rope_base if self.rope_base is not None else 10000.0

    @property
    def effective_max_cache_len(self) -> int:
        """Max cache length, defaulting to chunk_size if not specified."""
        return self.max_cache_len if self.max_cache_len is not None else self.chunk_size

    @classmethod
    def from_7b(cls) -> "MegalodonConfig":
        """Create configuration for 7B parameter model."""
        return cls(
            model_dim=4096,
            num_layers=32,
            num_heads=4,
            z_dim=1024,
            value_dim=8192,
            ffn_hidden_dim=11264,
            chunk_size=4096,
            norm_num_groups=64,
            rope_base=100_000.0,
            swiglu=True,
        )
