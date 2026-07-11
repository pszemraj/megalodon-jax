"""Independent released-source parity and training-resumption gates."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax import MegalodonConfig, MegalodonForCausalLM
from megalodon_jax.convert import export_upstream_state_dict
from tools.verify_modeling_correctness import (
    differentiable_state,
    source_forward,
    trainable_upstream_keys,
)


def parity_config(*, swiglu: bool = True) -> MegalodonConfig:
    """Return the smallest config that exercises all released tensor families."""
    return MegalodonConfig(
        vocab_size=17,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        z_dim=8,
        value_dim=8,
        ffn_hidden_dim=12,
        cema_ndim=2,
        chunk_size=8,
        norm_num_groups=2,
        swiglu=swiglu,
        rescale_nffn=True,
        scale_emb=True,
    )


@pytest.mark.torch_ref
@pytest.mark.parametrize("swiglu", [False, True])
def test_tiny_forward_matches_released_source_equations(swiglu: bool) -> None:
    """Compare complete logits against the local-source-derived Torch oracle."""
    torch = pytest.importorskip("torch")
    config = parity_config(swiglu=swiglu)
    model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(101))
    tokens = jnp.asarray([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=jnp.int32)

    jax_logits, _ = model(tokens)
    state = export_upstream_state_dict(model)
    torch_tokens = torch.from_numpy(np.asarray(tokens).copy())
    torch_logits = source_forward(torch_tokens, state, config)

    np.testing.assert_allclose(
        np.asarray(jax_logits),
        torch_logits.detach().cpu().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.torch_ref
def test_all_parameter_gradients_match_released_source_equations() -> None:
    """Compare every trainable upstream-schema gradient, not selected probes."""
    torch = pytest.importorskip("torch")
    config = parity_config()
    model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(202))
    tokens = jnp.asarray([[1, 3, 5, 7, 9]], dtype=jnp.int32)
    coefficients = jax.random.normal(
        jax.random.PRNGKey(303),
        (1, tokens.shape[1], config.effective_output_size),
    )

    def objective(candidate: MegalodonForCausalLM) -> jax.Array:
        logits, _ = candidate(tokens)
        return jnp.sum(logits * coefficients)

    jax_value, jax_grads = eqx.filter_value_and_grad(objective)(model)
    jax_grad_state = export_upstream_state_dict(jax_grads)

    torch_state = differentiable_state(export_upstream_state_dict(model))
    torch_tokens = torch.from_numpy(np.asarray(tokens).copy())
    torch_coefficients = torch.from_numpy(np.asarray(coefficients).copy())
    torch_value = (source_forward(torch_tokens, torch_state, config) * torch_coefficients).sum()
    torch_value.backward()

    np.testing.assert_allclose(float(jax_value), float(torch_value.detach()), rtol=2e-5, atol=2e-5)
    for name in trainable_upstream_keys(torch_state):
        actual = torch_state[name].grad
        assert actual is not None, f"Torch reference did not use {name}"
        np.testing.assert_allclose(
            np.asarray(jax_grad_state[name]),
            actual.detach().cpu().numpy(),
            rtol=8e-4,
            atol=4e-5,
            err_msg=f"gradient mismatch for {name}",
        )


@pytest.mark.torch_ref
def test_three_step_sgd_matches_released_source_equations() -> None:
    """Require loss, updates, and post-update logits to agree for three steps."""
    torch = pytest.importorskip("torch")
    config = parity_config()
    model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(404))
    tokens = jnp.asarray([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32)
    torch_tokens = torch.from_numpy(np.asarray(tokens).copy())
    torch_state = differentiable_state(export_upstream_state_dict(model))
    learning_rate = 2e-3

    def jax_loss(candidate: MegalodonForCausalLM) -> jax.Array:
        return candidate.compute_loss(tokens, tokens)

    jax_loss_and_grad = eqx.filter_value_and_grad(jax_loss)
    for _ in range(3):
        loss, grads = jax_loss_and_grad(model)
        updates = jax.tree.map(lambda gradient: -learning_rate * gradient, grads)
        model = eqx.apply_updates(model, updates)

        torch_logits = source_forward(torch_tokens, torch_state, config)
        torch_loss = torch.nn.functional.cross_entropy(
            torch_logits[:, :-1].reshape(-1, config.effective_output_size),
            torch_tokens[:, 1:].reshape(-1).long(),
        )
        torch_loss.backward()
        np.testing.assert_allclose(float(loss), float(torch_loss.detach()), rtol=3e-5, atol=3e-5)

        next_state: dict[str, torch.Tensor] = {}
        trainable = set(trainable_upstream_keys(torch_state))
        for name, value in torch_state.items():
            if name in trainable:
                assert value.grad is not None
                updated = value - learning_rate * value.grad
            else:
                updated = value
            next_state[name] = updated.detach().clone().requires_grad_(name in trainable)
        torch_state = next_state

    jax_logits, _ = model(tokens)
    torch_logits = source_forward(torch_tokens, torch_state, config)
    np.testing.assert_allclose(
        np.asarray(jax_logits),
        torch_logits.detach().cpu().numpy(),
        rtol=8e-4,
        atol=5e-5,
    )


def test_deterministic_tiny_batch_overfit() -> None:
    """Ensure a fresh FP32 model can substantially overfit a fixed tiny batch."""
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
    model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(505))
    tokens = jnp.asarray([[1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32)
    learning_rate = 2e-2

    @eqx.filter_jit
    def step(candidate: MegalodonForCausalLM) -> tuple[MegalodonForCausalLM, jax.Array]:
        def loss_fn(current: MegalodonForCausalLM) -> jax.Array:
            return current.compute_loss(tokens, tokens)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(candidate)
        updates = jax.tree.map(lambda gradient: -learning_rate * gradient, grads)
        return eqx.apply_updates(candidate, updates), loss

    initial = float(model.compute_loss(tokens, tokens))
    for _ in range(80):
        model, _ = step(model)
    final = float(model.compute_loss(tokens, tokens))

    assert np.isfinite(final)
    assert final < 0.35 * initial, f"tiny-batch loss only improved from {initial} to {final}"
