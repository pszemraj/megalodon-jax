"""Top-level Megalodon model classes.

This module contains the complete model assembly:
- MegalodonBlock: Single decoder block (attention + FFN)
- MegalodonModel: Full decoder stack with embeddings and final norm
- MegalodonForCausalLM: Model wrapper with tied LM head
"""

from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.layers import MegalodonAttention, NormalizedFFN, TimestepNorm
from megalodon_jax.types import LayerCache, NormState

T = TypeVar("T")


def _register_pytree(cls: type[T]) -> type[T]:
    """Register a dataclass as a JAX pytree node.

    This decorator enables JAX transformations (jit, vmap, scan) to work with
    the decorated dataclass by defining how to flatten and unflatten it.

    Args:
        cls: A dataclass type to register.

    Returns:
        The same class, now registered as a pytree node.
    """

    def flatten(obj: T) -> tuple[tuple[Any, ...], None]:
        """Flatten dataclass to (children, aux_data)."""
        children = tuple(getattr(obj, f.name) for f in fields(obj))
        return children, None

    def unflatten(aux_data: None, children: tuple[Any, ...]) -> T:
        """Reconstruct dataclass from children."""
        return cls(*children)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


@_register_pytree
@dataclass
class ModelCache:
    """Full model cache: layer caches + final norm state.

    This cache structure holds all streaming state for the model:
    - One LayerCache per decoder layer (attention, norm, EMA state)
    - One NormState for the final TimestepNorm

    Note: layer_caches must be a tuple (not list) for JAX pytree compatibility.
    """

    layer_caches: tuple[LayerCache | None, ...]
    final_norm: NormState | None = None


# -----------------------------------------------------------------------------
# MegalodonBlock
# -----------------------------------------------------------------------------


class MegalodonBlock(eqx.Module):
    """A single decoder block: Attention + FFN with two-hop residual.

    Architecture:
        residual_base = x
        x = MegalodonAttention(x, cache)
        x = NormalizedFFN(x, residual_base=residual_base)  # Two-hop residual

    The two-hop residual is critical: FFN adds the residual from the original
    block input (residual_base), not from the post-attention activations.
    This prevents variance accumulation across blocks.
    """

    attn: MegalodonAttention
    ffn: NormalizedFFN
    layer_id: int = eqx.field(static=True)

    def __init__(
        self,
        config: MegalodonConfig,
        layer_id: int,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize block with attention and FFN sub-modules.

        Args:
            config: Model configuration.
            layer_id: Index of this block (0-indexed) for rescaling.
            key: PRNG key for initialization.
        """
        k1, k2 = jax.random.split(key)
        self.layer_id = layer_id

        self.attn = MegalodonAttention(
            model_dim=config.model_dim,
            z_dim=config.z_dim,
            value_dim=config.value_dim,
            num_heads=config.num_heads,
            cema_ndim=config.cema_ndim,
            chunk_size=config.chunk_size,
            norm_num_groups=config.norm_num_groups,
            norm_eps=config.norm_eps,
            norm_affine=config.norm_affine,
            rope_base=config.effective_rope_base,
            max_cache_len=config.effective_max_cache_len,
            cache_unbounded=config.cache_unbounded,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            key=k1,
        )

        self.ffn = NormalizedFFN(
            model_dim=config.model_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            norm_eps=config.norm_eps,
            norm_affine=config.norm_affine,
            swiglu=config.swiglu,
            rescale=config.rescale_nffn,
            layer_id=layer_id,
            hidden_dropout=config.hidden_dropout,
            dropout=config.dropout,
            key=k2,
        )

    def __call__(
        self,
        x: Float[Array, "batch seq dim"],
        cache: LayerCache | None = None,
        mask: Bool[Array, "batch seq"] | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "batch seq dim"], LayerCache | None]:
        """Apply attention + FFN with two-hop residual.

        Args:
            x: Input tensor of shape (batch, seq, dim).
            cache: Optional LayerCache from previous forward.
            mask: Optional attention mask (True = valid token).
            return_cache: Whether to return updated cache.
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Tuple of (output, updated_cache).
        """
        # CRITICAL: Two-hop residual - save BEFORE attention
        residual_base = x

        # Split key for attention and FFN
        if key is not None:
            k_attn, k_ffn = jax.random.split(key)
        else:
            k_attn = k_ffn = None

        # Attention block
        x, cache = self.attn(
            x,
            cache=cache,
            mask=mask,
            return_cache=return_cache,
            deterministic=deterministic,
            key=k_attn,
        )

        # FFN with two-hop residual
        x = self.ffn(
            x,
            residual_base=residual_base,
            deterministic=deterministic,
            key=k_ffn,
        )

        return x, cache


# -----------------------------------------------------------------------------
# MegalodonModel
# -----------------------------------------------------------------------------


class MegalodonModel(eqx.Module):
    """Bare Megalodon decoder: embeddings + layer stack + final norm.

    This is the core decoder without the LM head. It processes input token IDs
    through embedding, a stack of MegalodonBlocks, and a final TimestepNorm.

    The final TimestepNorm carries state just like per-layer norms, enabling
    proper streaming normalization across chunks.
    """

    embed: eqx.nn.Embedding
    layers: tuple[MegalodonBlock, ...]
    norm: TimestepNorm
    scale: float = eqx.field(static=True)
    config: MegalodonConfig = eqx.field(static=True)

    def __init__(
        self,
        config: MegalodonConfig,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize embeddings, layer stack, and final TimestepNorm.

        Args:
            config: Model configuration.
            key: PRNG key for initialization.
        """
        keys = jax.random.split(key, config.num_layers + 1)

        self.config = config
        self.scale = float(jnp.sqrt(config.model_dim)) if config.scale_emb else 1.0

        self.embed = eqx.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_size=config.model_dim,
            key=keys[0],
        )

        self.layers = tuple(
            MegalodonBlock(config, i, key=keys[i + 1]) for i in range(config.num_layers)
        )

        self.norm = TimestepNorm(
            num_features=config.model_dim,
            num_groups=config.norm_num_groups,
            eps=config.norm_eps,
            affine=config.norm_affine,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        attention_mask: Bool[Array, "batch seq"] | None = None,
        cache: ModelCache | None = None,
        return_cache: bool = True,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "batch seq dim"], ModelCache | None]:
        """Forward pass through embeddings, layers, and final norm.

        Args:
            input_ids: Token IDs of shape (batch, seq).
            attention_mask: Optional mask (True = valid token).
            cache: Optional ModelCache from previous forward.
            return_cache: Whether to return updated cache.
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Tuple of (hidden_states, updated_cache).
        """
        # Embedding with optional scaling
        x = self.embed(input_ids) * self.scale

        # Parse cache
        if cache is not None:
            layer_caches = list(cache.layer_caches)
            final_norm_state = cache.final_norm
        else:
            layer_caches = [None] * len(self.layers)
            final_norm_state = None

        # Split keys for layers
        if key is not None:
            keys = list(jax.random.split(key, len(self.layers)))
        else:
            keys = [None] * len(self.layers)

        # Process layers
        new_caches: list[LayerCache | None] = []
        for layer, layer_cache, layer_key in zip(self.layers, layer_caches, keys):
            x, new_cache = layer(
                x,
                cache=layer_cache,
                mask=attention_mask,
                return_cache=return_cache,
                deterministic=deterministic,
                key=layer_key,
            )
            new_caches.append(new_cache)

        # Final TimestepNorm
        x, final_norm_state = self.norm(x, state=final_norm_state, mask=attention_mask)

        # Build output cache
        if return_cache:
            out_cache = ModelCache(
                layer_caches=tuple(new_caches),
                final_norm=final_norm_state,
            )
        else:
            out_cache = None

        return x, out_cache


# -----------------------------------------------------------------------------
# MegalodonForCausalLM
# -----------------------------------------------------------------------------


class MegalodonForCausalLM(eqx.Module):
    """Megalodon decoder with tied LM head for causal language modeling.

    The LM head shares weights with the input embeddings (weight tying).
    This is implemented by directly computing logits as:
        logits = hidden @ embed.weight.T

    No separate lm_head parameter is needed.
    """

    model: MegalodonModel
    config: MegalodonConfig = eqx.field(static=True)

    def __init__(
        self,
        config: MegalodonConfig,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize model with tied embeddings.

        Args:
            config: Model configuration.
            key: PRNG key for initialization.
        """
        self.config = config
        self.model = MegalodonModel(config, key=key)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        attention_mask: Bool[Array, "batch seq"] | None = None,
        cache: ModelCache | None = None,
        return_cache: bool = True,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "batch seq vocab"], ModelCache | None]:
        """Forward pass returning logits.

        Args:
            input_ids: Token IDs of shape (batch, seq).
            attention_mask: Optional mask (True = valid token).
            cache: Optional ModelCache from previous forward.
            return_cache: Whether to return updated cache.
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Tuple of (logits, updated_cache).
        """
        hidden, cache = self.model(
            input_ids,
            attention_mask=attention_mask,
            cache=cache,
            return_cache=return_cache,
            deterministic=deterministic,
            key=key,
        )

        # Weight-tied projection (no separate lm_head)
        logits = hidden @ self.model.embed.weight.T

        return logits, cache

    def compute_loss(
        self,
        input_ids: Int[Array, "batch seq"],
        labels: Int[Array, "batch seq"],
        attention_mask: Bool[Array, "batch seq"] | None = None,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Compute causal LM loss with label shifting.

        Computes cross-entropy loss for next-token prediction.
        Labels are automatically shifted: position i predicts position i+1.

        Args:
            input_ids: Input token IDs of shape (batch, seq).
            labels: Target token IDs of shape (batch, seq).
            attention_mask: Optional mask (True = valid token).
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Scalar cross-entropy loss.
        """
        logits, _ = self(
            input_ids,
            attention_mask=attention_mask,
            cache=None,
            return_cache=False,
            deterministic=deterministic,
            key=key,
        )

        # Shift for causal LM: predict next token
        shift_logits = logits[:, :-1, :]  # (B, L-1, V)
        shift_labels = labels[:, 1:]  # (B, L-1)

        # Cross-entropy loss
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        B, L, V = log_probs.shape

        # Gather log prob of correct token
        batch_idx = jnp.arange(B)[:, None]
        seq_idx = jnp.arange(L)[None, :]
        target_log_probs = log_probs[batch_idx, seq_idx, shift_labels]

        # Mean over all positions
        return -target_log_probs.mean()
