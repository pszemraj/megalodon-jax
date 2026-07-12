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
"""Complex Exponential Moving Average (EMA) for Megalodon JAX.

ComplexEMA is the core long-range memory mechanism in Megalodon. It uses
complex-valued coefficients to enable richer temporal dynamics than real EMA.

The recurrence is:
    h[t] = q * h[t-1] + p * x[t]
    y[t] = Re(sum(h[t] * gamma))

Three computation paths are provided:
- FFT: O(L log L) convolution for training (no state needed)
- Sequential: O(L) scan for streaming inference (state needed)
- Segmented: parallel associative scan for packed sequences (segment_ids),
  resetting the state at segment boundaries; a sequential low-memory
  fallback is also available.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float, Int, PRNGKeyArray

from megalodon_jax.layers.segments import segment_boundaries, valid_segment_mask

# Chunk size for kernel computation to bound memory usage
FFT_KERNEL_CHUNK = 4096


class ComplexEMA(eqx.Module):
    """Complex exponential moving average with FFT/sequential dispatch.

    Attributes:
        embed_dim: Hidden dimension D of the input tensor.
        ndim: Number of EMA orders tracked per hidden unit.
        scale: Output scaling factor (1/sqrt(ndim)).
        alpha: Logit-space input coefficient parameter.
        delta: Logit-space decay coefficient parameter.
        theta: Logit-space base angle parameter.
        gamma_real: Real part of output projection.
        gamma_imag: Imaginary part of output projection.
        omega: Residual skip weight.
    """

    embed_dim: int = eqx.field(static=True)
    ndim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    # Learnable parameters
    alpha: Float[Array, "dim ndim 1"]
    delta: Float[Array, "dim ndim 1"]
    theta: Float[Array, "dim 1 1"]
    gamma_real: Float[Array, "dim ndim"]
    gamma_imag: Float[Array, "dim ndim"]
    omega: Float[Array, "dim"]

    def __init__(self, embed_dim: int, ndim: int, *, key: PRNGKeyArray):
        """Initialize ComplexEMA with learnable parameters.

        :param int embed_dim: Hidden dimension of the input tensor.
        :param int ndim: Number of EMA orders per hidden unit.
        :param PRNGKeyArray key: PRNG key for initialization.
        :return None: None.
        """
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.scale = math.sqrt(1.0 / float(ndim))

        # Split key for different initializations
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # alpha & delta (logit space): normal(0, 0.2)
        self.alpha = jax.random.normal(k1, (embed_dim, ndim, 1), dtype=jnp.float32) * 0.2
        self.delta = jax.random.normal(k2, (embed_dim, ndim, 1), dtype=jnp.float32) * 0.2

        # theta (logit space): inverse-sigmoid of permuted frequency schedule
        freqs = math.log(embed_dim) / float(embed_dim)
        freqs = jnp.exp(jnp.arange(1, embed_dim + 1, dtype=jnp.float32) * -freqs)
        # Permute frequencies
        perm = jax.random.permutation(k3, embed_dim)
        freqs = freqs[perm]
        # Clamp to valid range for logit
        freqs = jnp.clip(freqs, 1e-6, 1.0 - 1e-6)
        # Inverse sigmoid (logit)
        freqs = jnp.log(freqs / (1.0 - freqs))
        self.theta = freqs.reshape(embed_dim, 1, 1)

        # gamma: real part normal(0, 1), imaginary part zeros
        self.gamma_real = jax.random.normal(k4, (embed_dim, ndim), dtype=jnp.float32)
        self.gamma_imag = jnp.zeros((embed_dim, ndim), dtype=jnp.float32)

        # omega: truncated normal(0, 0.25, -1, 1)
        self.omega = (
            jax.random.truncated_normal(k5, -4.0, 4.0, (embed_dim,), dtype=jnp.float32) * 0.25
        )

    def _coeffs(
        self,
    ) -> tuple[Float[Array, "dim ndim"], Complex[Array, "dim ndim"], Complex[Array, "dim ndim"]]:
        """Compute EMA coefficients (p, q, gamma).

        All computations are forced to fp32/complex64 for numerical stability,
        regardless of the parameter dtype. This matches the PyTorch reference
        behavior and prevents precision loss when the model is cast to bf16.

        :return tuple[Float[Array, "dim ndim"], Complex[Array, "dim ndim"], Complex[Array, "dim ndim"]]: Tuple of (p, q, gamma) with float32/complex64 dtypes.
        """
        N = self.ndim

        # Force fp32 for all coefficient computations to match PyTorch reference
        alpha_f32 = self.alpha.astype(jnp.float32)
        delta_f32 = self.delta.astype(jnp.float32)
        theta_f32 = self.theta.astype(jnp.float32)
        gamma_real_f32 = self.gamma_real.astype(jnp.float32)
        gamma_imag_f32 = self.gamma_imag.astype(jnp.float32)

        # theta -> phase angles
        theta = jax.nn.sigmoid(theta_f32) * (2.0 * jnp.pi / float(N))  # (D, 1, 1)
        wavelets = jnp.arange(1, N + 1, dtype=jnp.float32).reshape(1, N, 1)  # (1, N, 1)
        phi = wavelets * theta  # (D, N, 1)
        phi = phi.squeeze(-1)  # (D, N)

        # alpha, delta -> p, |q|
        alpha = jax.nn.sigmoid(alpha_f32)  # (D, N, 1)
        delta = jax.nn.sigmoid(delta_f32)  # (D, N, 1)

        p = alpha.squeeze(-1)  # (D, N) - real input coefficient
        magnitude = (1.0 - alpha * delta).squeeze(-1)  # (D, N)

        # q = magnitude * exp(i * phi) - complex with |q| < 1
        q = magnitude * jnp.exp(1j * phi)  # (D, N) complex64

        # gamma from real/imag parts
        scale = jnp.asarray(self.scale, dtype=gamma_real_f32.dtype)
        gamma = (gamma_real_f32 + 1j * gamma_imag_f32) * scale  # (D, N) complex64

        return p, q, gamma

    @staticmethod
    def _real_of_product(a: Complex[Array, "..."], b: Complex[Array, "..."]) -> Float[Array, "..."]:
        """Compute real part of complex product efficiently.

        Re(a * b) = a.real * b.real - a.imag * b.imag

        :param Complex[Array, "..."] a: Complex tensor input.
        :param Complex[Array, "..."] b: Complex tensor input.
        :return Float[Array, "..."]: Real part of the product.
        """
        return a.real * b.real - a.imag * b.imag

    def _forward_segmented(
        self,
        x: Float[Array, "batch dim seq"],
        segment_ids: Int[Array, "batch seq"],
    ) -> tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"]]:
        """Segment-aware EMA via parallel associative scan.

        Each step is an affine map h[t] = A[t] * h[t-1] + b[t] with A[t] = q,
        except at segment starts where A[t] = 0 so no state crosses document
        boundaries in packed sequences. Composition of affine maps is
        associative, so the full trajectory is computed in log depth.

        Materializes (L, B, D, N) complex64 tensors for A and b; for
        memory-constrained cases use the sequential fallback
        (``_forward_sequential`` with ``segment_ids``).

        :param Float[Array, "batch dim seq"] x: Input tensor (masked positions pre-zeroed).
        :param Int[Array, "batch seq"] segment_ids: Per-token segment IDs (0 = padding).
        :return tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"]]: Output
            tensor and final EMA state, anchored at each row's last non-padding
            token (zeros for all-padding rows).
        """
        p, q, gamma = self._coeffs()  # (D, N) each

        # (L, B, 1, 1) reset flags broadcasting against coefficient (D, N) axes
        reset = jnp.moveaxis(segment_boundaries(segment_ids), -1, 0)[:, :, None, None]

        # Per-step affine coefficients, (L, B, D, N) complex64
        x_t = jnp.moveaxis(x.astype(jnp.float32), -1, 0)[:, :, :, None]  # (L, B, D, 1)
        A = jnp.where(reset, jnp.zeros((), dtype=jnp.complex64), q[None, None, :, :])
        b = p[None, None, :, :] * x_t.astype(jnp.complex64)

        def combine(
            left: tuple[Complex[Array, "..."], Complex[Array, "..."]],
            right: tuple[Complex[Array, "..."], Complex[Array, "..."]],
        ) -> tuple[Complex[Array, "..."], Complex[Array, "..."]]:
            """Compose two affine maps (earlier left, later right).

            :param tuple left: Coefficients (A, b) of the earlier map.
            :param tuple right: Coefficients (A, b) of the later map.
            :return tuple: Coefficients of the composed map.
            """
            A_l, b_l = left
            A_r, b_r = right
            return A_r * A_l, A_r * b_l + b_r

        # Inclusive prefix composition; with h[-1] = 0 (guaranteed by the
        # h_init guard in __call__) the state trajectory is exactly b_cum.
        _, h_seq = jax.lax.associative_scan(combine, (A, b), axis=0)  # (L, B, D, N)

        y_seq = self._real_of_product(h_seq, gamma[None, None, :, :]).sum(axis=-1)  # (L, B, D)
        y = jnp.moveaxis(y_seq, 0, -1)  # (B, D, L)

        # Anchor the final state at each row's last real token: trailing
        # padding (id 0) starts its own run and resets the scan, so h_seq[-1]
        # would report the padding run's zero state instead of the last
        # document's.
        B, L = segment_ids.shape
        positions = jnp.arange(L, dtype=jnp.int32)
        # Shared packed-state contract: anchor continuation at each row's last
        # real token. Keep the segmented paths below and TimestepNorm aligned.
        last_valid = jnp.max(
            jnp.where(valid_segment_mask(segment_ids), positions[None, :], -1), axis=1
        )  # (B,)
        h_final = h_seq[jnp.maximum(last_valid, 0), jnp.arange(B)]  # (B, D, N)
        h_final = jnp.where(
            (last_valid >= 0)[:, None, None], h_final, jnp.zeros((), dtype=h_final.dtype)
        )

        # Return fp32 - caller handles dtype conversion
        return y, h_final

    def _forward_fft(self, x: Float[Array, "batch dim seq"]) -> Float[Array, "batch dim seq"]:
        """FFT-based convolution for training.

        Uses O(L log L) FFT convolution when no streaming state is needed.

        :param Float[Array, "batch dim seq"] x: Input tensor.
        :return Float[Array, "batch dim seq"]: Output tensor.
        """
        B, D, L = x.shape

        if L == 0:
            return jnp.zeros((B, D, L), dtype=x.dtype)

        # Get coefficients
        p, q, gamma = self._coeffs()  # p: (D, N), q: (D, N) complex, gamma: (D, N) complex

        # gamma * p for kernel computation
        gp = gamma * p  # (D, N) complex

        # Compute q^j via magnitude/phase for stability
        radius = jnp.abs(q).clip(max=1.0)  # (D, N)
        phi = jnp.angle(q)  # (D, N)

        # Build kernel in chunks to bound memory
        def compute_kernel_chunk(start: int, end: int) -> Complex[Array, "dim chunk"]:
            """Compute a slice of the EMA kernel from position start to end.

            Computes q^j for j in [start, end) where q = magnitude * exp(i*phi).
            Uses magnitude/phase representation for numerical stability.

            :param int start: Starting position index (inclusive).
            :param int end: Ending position index (exclusive).
            :return Complex[Array, "dim chunk"]: Complex kernel slice.
            """
            j = jnp.arange(start, end, dtype=jnp.float32)  # (chunk,)
            # q^j = radius^j * exp(i * phi * j)
            mag_chunk = radius[:, :, None] ** j[None, None, :]  # (D, N, chunk)
            phase_chunk = jnp.exp(1j * phi[:, :, None] * j[None, None, :])  # (D, N, chunk)
            q_pows = mag_chunk * phase_chunk  # (D, N, chunk) complex
            # Sum over N dimension
            return (gp[:, :, None] * q_pows).sum(axis=1)  # (D, chunk) complex

        # Compute full kernel
        if L <= FFT_KERNEL_CHUNK:
            kernel = compute_kernel_chunk(0, L)
        else:
            kernel = jnp.concatenate(
                [
                    compute_kernel_chunk(start, min(start + FFT_KERNEL_CHUNK, L))
                    for start in range(0, L, FFT_KERNEL_CHUNK)
                ],
                axis=-1,
            )  # (D, L) complex

        # FFT convolution using rfft for efficiency (x is real)
        fft_len = 1 << int(2 * L - 1).bit_length()
        x_f32 = x.astype(jnp.float32)

        # Only kernel.real contributes to real output
        kernel_real = kernel.real  # (D, L)

        X = jnp.fft.rfft(x_f32, n=fft_len, axis=-1)  # (B, D, fft_len//2+1)
        K = jnp.fft.rfft(kernel_real, n=fft_len, axis=-1)  # (D, fft_len//2+1)
        Y = X * K[None, :, :]
        y = jnp.fft.irfft(Y, n=fft_len, axis=-1)[..., :L]

        # Return fp32 - caller handles dtype conversion
        return y

    def _forward_sequential(
        self,
        x: Float[Array, "batch dim seq"],
        h_init: Complex[Array, "batch dim ndim"] | None,
        segment_ids: Int[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"]]:
        """Sequential recurrence using lax.scan.

        Computes the EMA recurrence h[t] = q * h[t-1] + p * x[t] using
        jax.lax.scan for efficiency. When segment_ids is given, the state is
        zeroed at segment boundaries (low-memory fallback to the parallel
        ``_forward_segmented`` path).

        :param Float[Array, "batch dim seq"] x: Input tensor.
        :param Complex[Array, "batch dim ndim"] | None h_init: Initial EMA state.
        :param Int[Array, "batch seq"] | None segment_ids: Optional per-token segment IDs.
        :return tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"]]: Output
            tensor and final EMA state; with segment_ids the state is anchored
            at each row's last non-padding token (zeros for all-padding rows).
        """
        B, D, L = x.shape
        N = self.ndim

        # Get coefficients
        p, q, gamma = self._coeffs()  # (D, N) each

        if h_init is None:
            h_init = jnp.zeros((B, D, N), dtype=jnp.complex64)

        # Broadcast coefficients for batched operation
        p_b = p[None, :, :]  # (1, D, N)
        q_b = q[None, :, :]  # (1, D, N)
        gamma_b = gamma[None, :, :]  # (1, D, N)

        def step(
            h: Complex[Array, "batch dim ndim"], x_t: Float[Array, "batch dim"]
        ) -> tuple[Complex[Array, "batch dim ndim"], Float[Array, "batch dim"]]:
            """Single EMA step.

            :param Complex[Array, "batch dim ndim"] h: Previous EMA state.
            :param Float[Array, "batch dim"] x_t: Input for the timestep.
            :return tuple[Complex[Array, "batch dim ndim"], Float[Array, "batch dim"]]: Updated state and output.
            """
            x_t_c = x_t[:, :, None].astype(jnp.complex64)  # (B, D, 1)
            h_new = q_b * h + p_b * x_t_c  # (B, D, N)
            y_t = self._real_of_product(h_new, gamma_b).sum(axis=-1)  # (B, D)
            return h_new, y_t

        # x is (B, D, L), scan over L
        x_transposed = jnp.moveaxis(x.astype(jnp.float32), -1, 0)  # (L, B, D)
        if segment_ids is None:
            h_final, y_seq = jax.lax.scan(step, h_init, x_transposed)
        else:
            reset_seq = jnp.moveaxis(segment_boundaries(segment_ids), -1, 0)  # (L, B)
            # Preserve the same last-real-token continuation contract as the FFT path.
            valid_seq = jnp.moveaxis(valid_segment_mask(segment_ids), -1, 0)  # (L, B)

            def step_with_reset(
                carry: tuple[Complex[Array, "batch dim ndim"], Complex[Array, "batch dim ndim"]],
                inputs: tuple[
                    Float[Array, "batch dim"], Bool[Array, "batch"], Bool[Array, "batch"]
                ],
            ) -> tuple[
                tuple[Complex[Array, "batch dim ndim"], Complex[Array, "batch dim ndim"]],
                Float[Array, "batch dim"],
            ]:
                """Single EMA step with state reset at segment starts.

                Alongside the running state, the carry tracks the state at the
                last real token so trailing padding (whose reset zeroes the
                running state) does not blank the returned final state.

                :param tuple carry: (running state, state at last real token).
                :param tuple inputs: Input, reset flag, and validity flag for the timestep.
                :return tuple: Updated carry and output.
                """
                h, h_anchor = carry
                x_t, reset_t, valid_t = inputs
                h = jnp.where(reset_t[:, None, None], jnp.zeros((), dtype=jnp.complex64), h)
                h_new, y_t = step(h, x_t)
                h_anchor = jnp.where(valid_t[:, None, None], h_new, h_anchor)
                return (h_new, h_anchor), y_t

            (_, h_final), y_seq = jax.lax.scan(
                step_with_reset, (h_init, h_init), (x_transposed, reset_seq, valid_seq)
            )
        y = jnp.moveaxis(y_seq, 0, -1)  # (B, D, L)

        # Return fp32 - caller handles dtype conversion
        return y, h_final

    def __call__(
        self,
        x: Float[Array, "batch dim seq"],
        h_init: Complex[Array, "batch dim ndim"] | None = None,
        return_state: bool = False,
        mask: Bool[Array, "batch seq"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        use_associative_segment_scan: bool = True,
    ) -> tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"] | None]:
        """Apply EMA and optionally return final state.

        Automatically selects FFT path (faster) when no state is needed,
        or sequential path when streaming. When segment_ids is given (packed
        sequences), the state resets at segment boundaries via a parallel
        associative scan; the FFT path cannot express resets and is bypassed.

        :param Float[Array, "batch dim seq"] x: Input tensor.
        :param Complex[Array, "batch dim ndim"] | None h_init: Initial EMA state.
        :param bool return_state: Whether to return the final complex state.
        :param Bool[Array, "batch seq"] | None mask: Optional validity mask.
        :param Int[Array, "batch seq"] | None segment_ids: Optional per-token segment IDs
            (0 = padding) for packed-sequence state resets. Training-only:
            incompatible with h_init.
        :param bool use_associative_segment_scan: Use the parallel associative scan for
            the segmented path; False selects the sequential low-memory fallback.
        :return tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"] | None]: Output
            and optional final state; with segment_ids the state is anchored at
            each row's last non-padding token so trailing padding does not
            zero it.
        """
        if segment_ids is not None and h_init is not None:
            raise ValueError(
                "segment_ids is not supported together with an incoming EMA state "
                "(h_init). Packed-sequence resets are a training-only, non-streaming "
                "feature."
            )

        # Store input dtype for output cast
        input_dtype = x.dtype

        # Positions outside any real segment (padding, id 0) must not contribute
        if segment_ids is not None:
            # Preserve the same last-real-token continuation contract as the FFT path.
            seg_valid = valid_segment_mask(segment_ids)
            mask = seg_valid if mask is None else (mask & seg_valid)

        # Zero masked positions to prevent new information from padding entering state.
        # Note: Masked positions still produce output based on decayed prior state
        # (h[t] = q*h[t-1] when x[t]=0), but no new information is added.
        # Matches PyTorch reference: zero masked positions to avoid padding contamination.
        if mask is not None:
            # mask: (batch, seq) -> (batch, 1, seq) for broadcasting with x: (batch, dim, seq)
            # Use dtype-matched zero to preserve bf16
            x = jnp.where(mask[:, None, :], x, jnp.zeros((), dtype=x.dtype))

        # Cast omega to fp32 for residual computation (matches PyTorch reference)
        omega_f32 = self.omega.astype(jnp.float32)
        x_f32 = x.astype(jnp.float32)
        residual = x_f32 * omega_f32[None, :, None]

        if segment_ids is not None:
            if use_associative_segment_scan:
                y, h_final = self._forward_segmented(x, segment_ids)
            else:
                y, h_final = self._forward_sequential(x, None, segment_ids=segment_ids)
            return (y + residual).astype(input_dtype), h_final if return_state else None

        use_fft = h_init is None and not return_state
        if use_fft:
            y = self._forward_fft(x)  # already returns fp32 internally
            return (y + residual).astype(input_dtype), None

        y, h_final = self._forward_sequential(x, h_init)  # already returns fp32 internally
        return (y + residual).astype(input_dtype), h_final if return_state else None
