"""Benchmark: CEMA FFT vs segmented associative scan vs sequential reset scan.

Not part of the test suite. Compares the three ComplexEMA compute paths on
forward and forward+backward, plus the memory footprint of the associative
path's (A, b) tensors.

Run: conda run --name mega-jax python benchmarks/benchmark_cema_reset.py
"""

import time

import jax
import jax.numpy as jnp

from megalodon_jax.layers.complex_ema import ComplexEMA


def bench(fn, n_warmup: int = 3, n_iters: int = 10) -> float:
    """Time fn() in seconds per call after warmup."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    out = None
    for _ in range(n_iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n_iters


def main() -> None:
    print(f"backend: {jax.default_backend()}, devices: {jax.devices()}")
    dim, ndim = 256, 16
    key = jax.random.PRNGKey(0)
    ema = ComplexEMA(dim, ndim, key=key)

    header = (
        f"{'L':>6} {'B':>3} | {'fft fwd':>9} {'assoc fwd':>9} {'seq fwd':>9} | "
        f"{'fft f+b':>9} {'assoc f+b':>9} {'seq f+b':>9} | {'A+b mem':>8}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in (512, 1024, 2048, 4096):
        for batch in (2, 8):
            x = jax.random.normal(jax.random.PRNGKey(1), (batch, dim, seq_len))
            seg = jnp.ones((batch, seq_len), dtype=jnp.int32)  # trivial single segment
            mask = jnp.ones((batch, seq_len), dtype=jnp.bool_)

            fwd_fft = jax.jit(lambda x: ema(x, mask=mask)[0])
            fwd_assoc = jax.jit(lambda x: ema(x, mask=mask, segment_ids=seg)[0])
            fwd_seq = jax.jit(
                lambda x: ema(x, mask=mask, segment_ids=seg, use_associative_segment_scan=False)[0]
            )

            g_fft = jax.jit(jax.grad(lambda x: jnp.sum(fwd_fft(x) ** 2)))
            g_assoc = jax.jit(jax.grad(lambda x: jnp.sum(fwd_assoc(x) ** 2)))
            g_seq = jax.jit(jax.grad(lambda x: jnp.sum(fwd_seq(x) ** 2)))

            t = [
                bench(lambda f=f: f(x))
                for f in (fwd_fft, fwd_assoc, fwd_seq, g_fft, g_assoc, g_seq)
            ]
            # A and b tensors, (L, B, D, N) complex64 = 8 bytes each
            mem_mb = 2 * seq_len * batch * dim * ndim * 8 / 1e6
            print(
                f"{seq_len:>6} {batch:>3} | "
                f"{t[0] * 1e3:>7.2f}ms {t[1] * 1e3:>7.2f}ms {t[2] * 1e3:>7.2f}ms | "
                f"{t[3] * 1e3:>7.2f}ms {t[4] * 1e3:>7.2f}ms {t[5] * 1e3:>7.2f}ms | "
                f"{mem_mb:>6.0f}MB"
            )


if __name__ == "__main__":
    main()
