"""Reusable deterministic training gates for modeling correctness tests."""

from __future__ import annotations

import math
from typing import Any


def deterministic_tiny_overfit(seed: int = 405) -> dict[str, float | bool]:
    """Run the shared deterministic FP32 tiny-batch overfit gate."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from megalodon_jax import MegalodonConfig, MegalodonForCausalLM

    config = MegalodonConfig(
        vocab_size=8,
        model_dim=8,
        num_layers=1,
        num_heads=1,
        z_dim=4,
        value_dim=8,
        ffn_hidden_dim=12,
        cema_ndim=2,
        chunk_size=8,
        norm_num_groups=2,
    )
    model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(seed))
    tokens = jnp.asarray([[1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32)
    learning_rate = 2e-2

    @eqx.filter_jit
    def step(candidate: MegalodonForCausalLM) -> tuple[MegalodonForCausalLM, Any]:
        def loss_fn(current: MegalodonForCausalLM) -> Any:
            return current.compute_loss(tokens, tokens)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(candidate)
        updates = jax.tree.map(lambda gradient: -learning_rate * gradient, grads)
        return eqx.apply_updates(candidate, updates), loss

    initial = float(model.compute_loss(tokens, tokens))
    for _ in range(80):
        model, _ = step(model)
    final = float(model.compute_loss(tokens, tokens))
    ratio = final / initial
    return {
        "passed": math.isfinite(final) and ratio < 0.35,
        "initial_loss": initial,
        "final_loss": final,
        "ratio": ratio,
    }
