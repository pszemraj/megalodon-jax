"""Benchmark CEMA FFT, associative packed, and sequential packed paths.

This is an isolated accelerator benchmark, not part of the test suite. It
measures forward and full parameter/input forward-backward execution, compiler
temporary memory, and the conceptual size of the associative ``(A, b)``
operands. Use ``--dim 1024 --ndim 16 --lengths 2048 --batches 8`` for one
representative downstream gate; see ``docs/dev.md`` for the complete matrix.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from megalodon_jax.layers.complex_ema import ComplexEMA


def _positive_csv(value: str) -> tuple[int, ...]:
    """Parse a comma-separated sequence of positive integers."""
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def _ready(value: Any) -> Any:
    """Synchronize every array leaf in a result tree."""
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def _compile(fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    """Compile a callable and return its executable and wall time."""
    started = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    return compiled, time.perf_counter() - started


def _temp_megabytes(compiled: Any) -> float:
    """Return compiler-reported temporary bytes in decimal megabytes."""
    analysis = compiled.memory_analysis()
    if analysis is None:
        return float("nan")
    return float(analysis.temp_size_in_bytes) / 1e6


def _all_finite(value: Any) -> bool:
    """Return whether every inexact result leaf is finite."""
    return all(
        bool(jax.device_get(jnp.all(jnp.isfinite(leaf))))
        for leaf in jax.tree.leaves(value)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    )


def bench(fn: Callable[[], Any], n_warmup: int, n_iters: int) -> float:
    """Time a synchronized callable in seconds per call after warmup."""
    for _ in range(n_warmup):
        _ready(fn())
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ready(fn())
    return (time.perf_counter() - t0) / n_iters


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--ndim", type=int, default=16)
    parser.add_argument("--lengths", type=_positive_csv, default=(512, 1024, 2048, 4096))
    parser.add_argument("--batches", type=_positive_csv, default=(2, 8))
    parser.add_argument("--segment-length", type=int, default=257)
    parser.add_argument("--padding", type=int, default=32)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser


def main() -> None:
    """Run the selected isolated CEMA benchmark matrix."""
    args = _parser().parse_args()
    if args.dim <= 0 or args.ndim <= 0 or args.segment_length <= 0:
        raise ValueError("dim, ndim, and segment-length must be positive")
    if args.padding < 0:
        raise ValueError("padding must be non-negative")
    if args.warmups < 0 or args.iterations <= 0:
        raise ValueError("warmups must be non-negative and iterations must be positive")

    print(f"backend: {jax.default_backend()}, devices: {jax.devices()}")
    dtype = jnp.dtype(args.dtype)
    key = jax.random.PRNGKey(0)
    ema = ComplexEMA(args.dim, args.ndim, key=key)

    header = (
        f"{'L':>6} {'B':>3} {'path':>6} | {'fwd compile':>11} {'fwd':>9} | "
        f"{'f+b compile':>11} {'f+b':>9} {'f+b temp':>10} {'finite':>6} | {'A+b':>9}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in args.lengths:
        for batch in args.batches:
            x = jax.random.normal(
                jax.random.fold_in(key, seq_len * batch),
                (batch, args.dim, seq_len),
                dtype=dtype,
            )
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            row_segments = (positions // args.segment_length) % 7 + 1
            valid_length = max(seq_len - args.padding, 0)
            row_segments = jnp.where(positions < valid_length, row_segments, 0)
            segments = jnp.broadcast_to(row_segments, (batch, seq_len))
            mask = segments > 0

            def output(module: ComplexEMA, values: jax.Array, path: str) -> jax.Array:
                if path == "fft":
                    return module(values, mask=mask)[0]
                return module(
                    values,
                    mask=mask,
                    segment_ids=segments,
                    use_associative_segment_scan=path == "assoc",
                )[0]

            for path in ("fft", "assoc", "seq"):

                def forward(module: ComplexEMA, values: jax.Array, path: str = path) -> jax.Array:
                    return output(module, values, path)

                def loss(module: ComplexEMA, values: jax.Array, path: str = path) -> jax.Array:
                    result = output(module, values, path).astype(jnp.float32)
                    return jnp.mean(result**2)

                training = jax.value_and_grad(loss, argnums=(0, 1))
                compiled_fwd, compile_fwd = _compile(forward, ema, x)
                compiled_train, compile_train = _compile(training, ema, x)
                finite = _all_finite(_ready(compiled_train(ema, x)))
                if not finite:
                    raise RuntimeError(f"non-finite forward/backward result for {path}")
                fwd_seconds = bench(lambda: compiled_fwd(ema, x), args.warmups, args.iterations)
                train_seconds = bench(lambda: compiled_train(ema, x), args.warmups, args.iterations)
                operands_mb = (
                    2 * seq_len * batch * args.dim * args.ndim * 8 / 1e6
                    if path == "assoc"
                    else None
                )
                operands = "-" if operands_mb is None else f"{operands_mb:.0f}MB"
                print(
                    f"{seq_len:>6} {batch:>3} {path:>6} | "
                    f"{compile_fwd:>9.2f}s {fwd_seconds * 1e3:>7.2f}ms | "
                    f"{compile_train:>9.2f}s {train_seconds * 1e3:>7.2f}ms "
                    f"{_temp_megabytes(compiled_train):>8.0f}MB {str(finite):>6} | {operands:>9}"
                )


if __name__ == "__main__":
    main()
