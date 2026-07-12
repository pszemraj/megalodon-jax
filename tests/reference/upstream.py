"""Pure-Torch reference for the released Megalodon model equations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

TorchStateDict = Mapping[str, Any]


def trainable_upstream_keys(state: TorchStateDict) -> tuple[str, ...]:
    """Return released-source parameters, excluding zero-prior and RoPE buffers."""
    buffers = (".prior_count", ".prior_mean", ".prior_logv")
    return tuple(
        name
        for name, value in state.items()
        if value.is_floating_point() and name != "rope.freqs" and not name.endswith(buffers)
    )


def differentiable_state(state: TorchStateDict) -> dict[str, Any]:
    """Clone a Torch checkpoint state with autograd enabled only for source parameters."""
    trainable = set(trainable_upstream_keys(state))
    return {
        name: value.detach().clone().requires_grad_(name in trainable)
        for name, value in state.items()
    }


def _linear(x: Any, state: TorchStateDict, prefix: str) -> Any:
    """Apply a released ``(out_features, in_features)`` projection."""
    import torch.nn.functional as functional

    return functional.linear(x, state[f"{prefix}.weight"], state.get(f"{prefix}.bias"))


def _timestep_norm(
    x: Any,
    state: TorchStateDict,
    prefix: str,
    groups: int,
    eps: float,
) -> Any:
    """Evaluate group TimestepNorm with population block-Welford updates."""
    import torch

    batch, length, dim = x.shape
    group_size = dim // groups
    grouped = x.float().reshape(batch, length, groups, group_size)
    mean = torch.zeros((batch, groups), dtype=torch.float32, device=x.device)
    variance = torch.ones_like(mean)
    outputs = []

    for index in range(length):
        block = grouped[:, index]
        block_mean = block.mean(dim=-1)
        block_var = torch.square(block - block_mean.unsqueeze(-1)).mean(dim=-1)
        count = float(index)
        next_count = count + 1.0
        delta = block_mean - mean
        mean = mean + delta / next_count
        variance = (
            count * variance + block_var + torch.square(delta) * count / next_count
        ) / next_count
        normalized = (block - mean.unsqueeze(-1)) * torch.rsqrt(variance.unsqueeze(-1) + eps)
        outputs.append(normalized.reshape(batch, dim))

    y = torch.stack(outputs, dim=1).to(x.dtype)
    scale = (state[f"{prefix}.weight"] + 1.0).to(y.dtype)
    bias = state[f"{prefix}.bias"].to(y.dtype)
    return y * scale + bias


def _rms_norm(x: Any, weight: Any | None, eps: float) -> Any:
    """Evaluate the released RMSNorm fallback in fp32."""
    import torch

    normalized = x.float() * torch.rsqrt(torch.square(x.float()).mean(dim=-1, keepdim=True) + eps)
    normalized = normalized.to(x.dtype)
    return normalized if weight is None else normalized * (weight + 1.0).to(x.dtype)


def _layer_norm(x: Any, weight: Any | None, bias: Any | None, eps: float) -> Any:
    """Evaluate released plus-one LayerNorm storage semantics."""
    import torch.nn.functional as functional

    effective_weight = None if weight is None else weight + 1.0
    return functional.layer_norm(x, (x.shape[-1],), effective_weight, bias, eps)


def _cema(x: Any, state: TorchStateDict, prefix: str, ndim: int) -> Any:
    """Evaluate the exact source-compatible CEMA recurrence and residual."""
    import torch

    alpha = torch.sigmoid(state[f"{prefix}.alpha"].float()).squeeze(-1)
    delta = torch.sigmoid(state[f"{prefix}.delta"].float()).squeeze(-1)
    theta = torch.sigmoid(state[f"{prefix}.theta"].float()) * (2.0 * math.pi / ndim)
    wavelets = torch.arange(1, ndim + 1, dtype=torch.float32, device=x.device).reshape(1, ndim)
    phase = theta.squeeze(-1) * wavelets
    q = torch.polar(1.0 - alpha * delta, phase)
    gamma_parts = state[f"{prefix}.gamma"].float()
    gamma = torch.view_as_complex(gamma_parts.contiguous()) / math.sqrt(ndim)

    batch, dim, length = x.shape
    hidden = torch.zeros((batch, dim, ndim), dtype=torch.complex64, device=x.device)
    outputs = []
    for index in range(length):
        hidden = q.unsqueeze(0) * hidden + alpha.unsqueeze(0) * x[:, :, index, None].float()
        outputs.append((hidden * gamma.unsqueeze(0)).real.sum(dim=-1))
    recurrent = torch.stack(outputs, dim=-1)
    omega = state[f"{prefix}.omega"].float().squeeze(-1)
    return (recurrent + x.float() * omega[None, :, None]).to(x.dtype)


def _rope(q: Any, k: Any, frequencies: Any) -> tuple[Any, Any]:
    """Apply the released adjacent-pair ``view_as_complex`` RoPE convention."""
    import torch

    length = q.shape[1]
    positions = torch.arange(length, dtype=torch.float32, device=q.device)
    angles = torch.outer(positions, frequencies.float())
    rotation = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)

    def rotate(x: Any) -> Any:
        complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(complex_x * rotation).flatten(-2).to(x.dtype)

    return rotate(q), rotate(k)


def _attention(
    q: Any,
    k: Any,
    v: Any,
    frequencies: Any,
    num_heads: int,
) -> Any:
    """Evaluate source attention with no Transformer ``1/sqrt(d)`` scale."""
    import torch

    batch, length, z_dim = q.shape
    value_dim = v.shape[-1]
    q = q.reshape(batch, length, num_heads, z_dim // num_heads)
    k = k.reshape_as(q)
    v = v.reshape(batch, length, num_heads, value_dim // num_heads)
    q, k = _rope(q, k, frequencies)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1))
    causal = torch.triu(
        torch.full((length, length), float("-inf"), dtype=scores.dtype, device=scores.device),
        diagonal=1,
    )
    probabilities = torch.softmax(scores + causal, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probabilities, v).transpose(1, 2).reshape(batch, length, value_dim)


def source_forward(tokens: Any, state: TorchStateDict, config: Any) -> Any:
    """Run a tiny differentiable forward pass from the exact released equations.

    Dropout must be disabled and the sequence must fit within one attention chunk.
    These constraints avoid fused kernels while retaining every modeling equation
    and trainable tensor used by the released architecture.
    """
    import torch.nn.functional as functional

    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape (batch, length), got {tuple(tokens.shape)}")
    if tokens.shape[1] > config.chunk_size:
        raise ValueError("source oracle supports at most one attention chunk")
    if any((config.dropout, config.attention_dropout, config.hidden_dropout)):
        raise ValueError("source oracle requires dropout probabilities to be zero")
    if config.attention_window is not None:
        raise ValueError("source oracle covers released chunk-local attention only")

    x = functional.embedding(tokens, state["embed.weight"])
    if config.scale_emb:
        x = x * math.sqrt(config.model_dim)

    for layer_index in range(config.num_layers):
        residual_base = x
        attention_prefix = f"layers.{layer_index}.mega"
        normalized = _timestep_norm(
            x,
            state,
            f"{attention_prefix}.timenorm",
            config.norm_num_groups,
            config.norm_eps,
        )
        cema = _cema(
            normalized.transpose(1, 2),
            state,
            f"{attention_prefix}.cema",
            config.cema_ndim,
        ).transpose(1, 2)
        rms_weight = state.get(f"{attention_prefix}.rmsnorm.weight")
        memory = _rms_norm(cema, rms_weight, config.norm_eps)

        z = _linear(memory, state, f"{attention_prefix}.wz")
        z = z.reshape(*z.shape[:-1], config.num_heads, config.head_dim)
        z = _rms_norm(z, None, config.norm_eps).reshape(*z.shape[:-2], config.z_dim)
        scale = (state[f"{attention_prefix}.gamma"] + 1.0) / math.sqrt(config.head_dim)
        z = z.unsqueeze(2) * scale + state[f"{attention_prefix}.beta"]
        q, k = z.unbind(dim=2)

        v = functional.silu(_linear(normalized, state, f"{attention_prefix}.wv"))
        reset = functional.silu(_linear(memory, state, f"{attention_prefix}.wr"))
        attended = _attention(q, k, v, state["rope.freqs"], config.num_heads)
        attention_output = _linear(memory, state, f"{attention_prefix}.wh1") + _linear(
            attended * reset, state, f"{attention_prefix}.wh2"
        )
        after_attention = residual_base + attention_output

        ffn_prefix = f"layers.{layer_index}.nffn"
        ffn_input = _layer_norm(
            after_attention,
            state.get(f"{ffn_prefix}.norm.weight"),
            state.get(f"{ffn_prefix}.norm.bias"),
            config.norm_eps,
        )
        hidden = functional.silu(_linear(ffn_input, state, f"{ffn_prefix}.fc1"))
        if config.swiglu:
            hidden = hidden * _linear(ffn_input, state, f"{ffn_prefix}.fc3")
        ffn_output = _linear(hidden, state, f"{ffn_prefix}.fc2")
        alpha = state.get(f"{ffn_prefix}.alpha")
        if alpha is not None:
            ffn_output = ffn_output * alpha
        x = residual_base + ffn_output

    x = _timestep_norm(
        x,
        state,
        "output.final_norm",
        config.norm_num_groups,
        config.norm_eps,
    )
    output_weight = state["embed.weight"] if config.share_emb else state["output.output.weight"]
    return functional.linear(x, output_weight).float()
