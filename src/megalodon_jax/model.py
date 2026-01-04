"""Top-level Megalodon model classes.

This module contains the complete model assembly:
- MegalodonBlock: Single decoder block (attention + FFN)
- MegalodonModel: Full decoder stack with embeddings and final norm
- MegalodonForCausalLM: Model wrapper with tied LM head
"""

from dataclasses import dataclass, fields
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.layers import MegalodonAttention, NormalizedFFN, TimestepNorm
from megalodon_jax.types import LayerCache, NormState
from megalodon_jax.utils import get_initializer, reinit_linear_weights

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


@eqx.filter_checkpoint
def _checkpointed_layer(
    layer: "MegalodonBlock",
    x: Float[Array, "batch seq dim"],
    mask: Bool[Array, "batch seq"] | None,
    key: PRNGKeyArray | None,
) -> Float[Array, "batch seq dim"]:
    """Execute layer without caching for gradient checkpointing.

    This function is defined at module level (not inside the loop) to avoid
    recreating the decorated function on each iteration.
    """
    out, _ = layer(x, cache=None, mask=mask, return_cache=False, deterministic=False, key=key)
    return out


def _stop_if_array(x: Any) -> Any:
    """Apply stop_gradient only to arrays, passing through non-arrays unchanged.

    This is needed because cache pytrees may contain None leaves (e.g., when
    checkpointing is active), and jax.lax.stop_gradient only accepts arrays.
    """
    return jax.lax.stop_gradient(x) if eqx.is_array(x) else x


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
    use_checkpoint: bool = eqx.field(static=True)
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
        self.config = config
        self.scale = float(jnp.sqrt(config.model_dim)) if config.scale_emb else 1.0
        self.use_checkpoint = config.use_checkpoint

        # Split keys explicitly for each purpose (clearer allocation)
        # - k_embed: initial embedding weights
        # - k_embed_reinit: embedding reinitialization (if init_mode != "none")
        # - k_layer_reinit: layer reinitialization (if init_mode != "none")
        # - layer_keys: one key per MegalodonBlock
        k_embed, k_embed_reinit, k_layer_reinit, k_layers = jax.random.split(key, 4)
        layer_keys = jax.random.split(k_layers, config.num_layers)

        # Initialize embedding
        embed = eqx.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_size=config.model_dim,
            key=k_embed,
        )

        # Apply init_mode to embedding weights
        if config.init_mode != "none":
            init_fn = get_initializer(config.init_mode, dim=config.model_dim)
            new_embed_weight = init_fn(k_embed_reinit, embed.weight.shape, embed.weight.dtype)
            embed = eqx.tree_at(lambda e: e.weight, embed, new_embed_weight)
        self.embed = embed

        # Initialize layers
        layers = tuple(
            MegalodonBlock(config, i, key=layer_keys[i]) for i in range(config.num_layers)
        )

        # Apply init_mode to all Linear layers in blocks
        if config.init_mode != "none":
            layer_reinit_keys = jax.random.split(k_layer_reinit, len(layers))
            # Match PyTorch: gaussian Linear init uses std=1.0 (dim=None).
            linear_dim = None if config.init_mode == "gaussian" else config.model_dim
            layers = tuple(
                reinit_linear_weights(layer, config.init_mode, k, dim=linear_dim)
                for layer, k in zip(layers, layer_reinit_keys)
            )
        self.layers = layers

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
        return_cache: bool = False,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "batch seq dim"], ModelCache | None]:
        """Forward pass through embeddings, layers, and final norm.

        Args:
            input_ids: Token IDs of shape (batch, seq).
            attention_mask: Optional mask (True = valid token).
            cache: Optional ModelCache from previous forward.
            return_cache: Whether to return updated cache. Defaults to False to
                avoid the streaming path during training. For streaming inference,
                set return_cache=True.
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Tuple of (hidden_states, updated_cache).
        """
        B, L = input_ids.shape

        # Handle empty inputs gracefully (B=0 or L=0)
        # Use embed dtype to match model's native dtype (e.g., bf16 if autocast applied)
        if B == 0 or L == 0:
            empty_hidden = jnp.zeros((B, L, self.config.model_dim), dtype=self.embed.weight.dtype)
            if return_cache:
                # Preserve any existing streaming state when input is empty.
                empty_cache = (
                    cache
                    if cache is not None
                    else ModelCache(tuple([None] * len(self.layers)), None)
                )
            else:
                empty_cache = None
            return empty_hidden, empty_cache

        # Disable streaming cache updates during training (matches PyTorch behavior).
        layer_return_cache = return_cache and deterministic

        # Validate PRNG key for dropout - prevent silent no-op when training
        if not deterministic and key is None:
            if (
                self.config.dropout > 0.0
                or self.config.attention_dropout > 0.0
                or self.config.hidden_dropout > 0.0
            ):
                raise ValueError(
                    "PRNG key required when deterministic=False and dropout is enabled. "
                    "Pass a key via `key=jax.random.PRNGKey(...)` or set deterministic=True."
                )

        # Validate token bounds - prevents silent incorrect embeddings from OOB indices
        # Note: Uses eqx.error_if for JIT-safe traced-value errors
        vocab_size = self.config.vocab_size
        has_invalid_tokens = jnp.any((input_ids < 0) | (input_ids >= vocab_size))
        input_ids = eqx.error_if(
            input_ids,
            has_invalid_tokens,
            f"input_ids contain out-of-bounds values. Valid range: [0, {vocab_size})",
        )

        # Embedding with optional scaling
        # eqx.nn.Embedding takes scalar indices, so vmap over batch and sequence
        # Cast scale to embedding dtype to preserve bf16/mixed-precision execution
        x = jax.vmap(jax.vmap(self.embed))(input_ids)
        if self.scale != 1.0:
            x = x * jnp.asarray(self.scale, dtype=x.dtype)

        # Zero-mask pad tokens (matches PyTorch padding_idx behavior)
        # This ensures pad tokens have zero embeddings and receive no gradient updates
        # Use dtype-matched zero to avoid upcasting bf16 to float32
        pad_mask = input_ids == self.config.pad_token_id
        x = jnp.where(pad_mask[:, :, None], jnp.zeros((), dtype=x.dtype), x)

        # Validate cache + padding constraint
        # Caching with padding is unsupported because:
        # - Attention cache stores K/V entries but not per-position validity flags
        # - Once cached, subsequent tokens would attend to padded positions
        # - ComplexEMA and TimestepNorm have mask support for prefill, but
        #   cache semantics assume autoregressive decode (no mid-sequence padding)
        # Guard both cache input AND output - streaming path is used in either case
        uses_streaming = layer_return_cache or cache is not None
        if uses_streaming and attention_mask is not None:
            # Check if any position is masked (False = padding)
            # Use eqx.error_if for traced-value-safe conditional errors
            has_padding = ~jnp.all(attention_mask)
            x = eqx.error_if(
                x,  # Value to pass through (and attach error to)
                has_padding,
                "Cannot use cache with padding in attention_mask. "
                "Caching is only supported for autoregressive generation without padding. "
                "Use cache=None and return_cache=False for padded prefill.",
            )

        # Parse cache with validation
        if cache is not None:
            if len(cache.layer_caches) != len(self.layers):
                raise ValueError(
                    f"Cache has {len(cache.layer_caches)} layer entries, "
                    f"expected {len(self.layers)}"
                )
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
        # When checkpointing is enabled during training (not deterministic),
        # we disable caching per the Phase 4 spec (matches PyTorch behavior)
        use_ckpt = self.use_checkpoint and not deterministic
        new_caches: list[LayerCache | None] = []
        for layer, layer_cache, layer_key in zip(self.layers, layer_caches, keys):
            if use_ckpt:
                # Checkpointed path: disable cache during training
                x = _checkpointed_layer(layer, x, attention_mask, layer_key)
                new_caches.append(None)
            else:
                # Standard path with optional caching
                x, new_cache = layer(
                    x,
                    cache=layer_cache,
                    mask=attention_mask,
                    return_cache=layer_return_cache,
                    deterministic=deterministic,
                    key=layer_key,
                )
                new_caches.append(new_cache)

        # Final TimestepNorm
        x, final_norm_state = self.norm(x, state=final_norm_state, mask=attention_mask)

        # Build output cache with stop_gradient to prevent accidental backprop
        # through cache history when cache is fed back under jax.grad
        if return_cache:
            out_cache = ModelCache(
                layer_caches=tuple(new_caches),
                final_norm=final_norm_state,
            )
            out_cache = jax.tree.map(_stop_if_array, out_cache)
        else:
            out_cache = None

        return x, out_cache


# -----------------------------------------------------------------------------
# MegalodonForCausalLM
# -----------------------------------------------------------------------------


class MegalodonForCausalLM(eqx.Module):
    """Megalodon decoder with LM head for causal language modeling.

    Supports both tied and untied LM heads:
    - When output_size == vocab_size or output_size == -1: weights are tied
    - When output_size != vocab_size: separate lm_head is created

    For tied weights, logits are computed as: hidden @ embed.weight.T
    For untied weights, logits are computed via the lm_head Linear layer.
    """

    model: MegalodonModel
    lm_head: eqx.nn.Linear | None  # None when tied
    tied: bool = eqx.field(static=True)
    config: MegalodonConfig = eqx.field(static=True)

    def __init__(
        self,
        config: MegalodonConfig,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize model with optional LM head.

        Args:
            config: Model configuration.
            key: PRNG key for initialization.
        """
        self.config = config

        # Determine output size and whether to tie weights
        # output_size=-1 means "use vocab_size" (tied weights)
        lm_out = config.vocab_size if config.output_size == -1 else config.output_size
        self.tied = lm_out == config.vocab_size

        if self.tied:
            # Tied weights: only need key for model
            self.model = MegalodonModel(config, key=key)
            self.lm_head = None
        else:
            # Untied weights: split keys for model, lm_head, and reinit
            k_model, k_head, k_head_reinit = jax.random.split(key, 3)
            self.model = MegalodonModel(config, key=k_model)

            lm_head = eqx.nn.Linear(config.model_dim, lm_out, use_bias=False, key=k_head)
            # Apply init_mode to untied lm_head
            # For gaussian init, use dim=None to get std=1.0 (matches PyTorch/other Linear layers)
            if config.init_mode != "none":
                linear_dim = None if config.init_mode == "gaussian" else lm_out
                init_fn = get_initializer(config.init_mode, dim=linear_dim)
                new_weight = init_fn(k_head_reinit, lm_head.weight.shape, lm_head.weight.dtype)
                lm_head = eqx.tree_at(lambda h: h.weight, lm_head, new_weight)
            self.lm_head = lm_head

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        attention_mask: Bool[Array, "batch seq"] | None = None,
        cache: ModelCache | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "batch seq vocab"], ModelCache | None]:
        """Forward pass returning logits.

        Args:
            input_ids: Token IDs of shape (batch, seq).
            attention_mask: Optional mask (True = valid token).
            cache: Optional ModelCache from previous forward.
            return_cache: Whether to return updated cache. Defaults to False; set
                True for streaming inference.
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

        if self.tied:
            # Weight-tied projection
            logits = hidden @ self.model.embed.weight.T
        else:
            # Separate LM head
            # eqx.nn.Linear takes single vectors, so vmap over batch and sequence
            logits = jax.vmap(jax.vmap(self.lm_head))(hidden)

        return logits, cache

    def compute_loss(
        self,
        input_ids: Int[Array, "batch seq"],
        labels: Int[Array, "batch seq"],
        attention_mask: Bool[Array, "batch seq"] | None = None,
        ignore_index: int = -100,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Compute causal LM loss with label shifting.

        Computes cross-entropy loss for next-token prediction.
        Labels are automatically shifted: position i predicts position i+1.

        Positions with labels equal to ignore_index are excluded from loss
        computation, following HuggingFace Transformers convention. This allows
        padding tokens to be marked with -100 in the labels array.

        Args:
            input_ids: Input token IDs of shape (batch, seq).
            labels: Target token IDs of shape (batch, seq). Positions with value
                equal to ignore_index are excluded from loss computation.
            attention_mask: Optional mask (True = valid token). Masked positions
                are excluded from the loss to prevent gradient noise from padding.
            ignore_index: Label value to ignore in loss computation. Default -100
                matches HuggingFace Transformers convention.
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Scalar cross-entropy loss.
        """
        # Validate PRNG key for dropout - prevent silent no-op when training
        if not deterministic and key is None:
            if (
                self.config.dropout > 0.0
                or self.config.attention_dropout > 0.0
                or self.config.hidden_dropout > 0.0
            ):
                raise ValueError(
                    "PRNG key required when deterministic=False and dropout is enabled. "
                    "Pass a key via `key=jax.random.PRNGKey(...)` or set deterministic=True."
                )

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

        # Guard against empty sequences (seq=0 or seq=1 after shift)
        # jnp.mean() on empty array returns NaN, which would poison training
        if shift_labels.shape[1] == 0:
            return jnp.zeros((), dtype=shift_logits.dtype)  # Match model dtype

        # Build valid mask: positions that contribute to loss
        # Excludes both ignore_index labels and attention-masked positions
        valid_mask = shift_labels != ignore_index
        if attention_mask is not None:
            shift_attn_mask = attention_mask[:, 1:]  # (B, L-1)
            valid_mask = valid_mask & shift_attn_mask

        # Validate label bounds only on positions that will be used
        # Prevents silent wrong gradients from JAX index wrapping
        vocab_size = shift_logits.shape[-1]
        # Check: any valid position has out-of-bounds label?
        has_invalid_labels = jnp.any(
            valid_mask & ((shift_labels < 0) | (shift_labels >= vocab_size))
        )
        shift_labels = eqx.error_if(
            shift_labels,
            has_invalid_labels,
            f"labels contain out-of-bounds values. Valid range: [0, {vocab_size})",
        )

        # Replace ignored labels with 0 for safe indexing (will be masked out)
        safe_labels = jnp.where(valid_mask, shift_labels, 0)

        # Cross-entropy loss
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        B, L, V = log_probs.shape

        # Gather log prob of correct token
        batch_idx = jnp.arange(B)[:, None]
        seq_idx = jnp.arange(L)[None, :]
        target_log_probs = log_probs[batch_idx, seq_idx, safe_labels]

        # Apply valid mask and compute mean over valid positions only
        target_log_probs = jnp.where(
            valid_mask, target_log_probs, jnp.zeros((), dtype=target_log_probs.dtype)
        )

        # Compute loss in float32 for numerical stability (standard practice for
        # mixed-precision training - avoids dtype promotion issues from int/float division)
        target_f32 = target_log_probs.astype(jnp.float32)
        num_valid = valid_mask.sum().astype(jnp.float32)
        # Avoid division by zero (return 0 loss if no valid positions)
        num_valid = jnp.maximum(num_valid, 1.0)
        return -target_f32.sum() / num_valid
