"""Megalodon configuration."""

from dataclasses import dataclass
from typing import Literal

InitMode = Literal["gaussian", "xavier", "he", "bert", "none"]


@dataclass(frozen=True)
class MegalodonConfig:
    """Configuration for Megalodon model.

    This is a frozen dataclass (immutable and hashable) that defines all
    hyperparameters for the Megalodon architecture.
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
