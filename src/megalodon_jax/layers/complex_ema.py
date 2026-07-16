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
- FFT: O(L log L) convolution for training and long cached chunks
- Sequential: O(L) scan for short streaming-inference chunks
- Segmented: parallel associative scan for packed sequences (segment_ids),
  resetting the state at segment boundaries; a sequential low-memory
  fallback is also available.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float, Int, PRNGKeyArray

from megalodon_jax.layers.segments import (
    SegmentMetadata,
    derive_segment_metadata,
)

# Chunk size for kernel computation to bound memory usage
FFT_KERNEL_CHUNK = 4096

# Below the released fused FFT kernel's lower bound, a recurrent scan is faster
# and avoids constructing a convolution kernel for tokenwise decoding.
FFT_RECURRENT_MIN_LENGTH = 32


class ComplexEMA(eqx.Module):
    """Complex exponential moving average with FFT/sequential dispatch.

    Attributes:
        ndim: Number of EMA orders tracked per hidden unit.
        scale: Output scaling factor (1/sqrt(ndim)).
        alpha: Logit-space input coefficient parameter.
        delta: Logit-space decay coefficient parameter.
        theta: Logit-space base angle parameter.
        gamma_real: Real part of output projection.
        gamma_imag: Imaginary part of output projection.
        omega: Residual skip weight.
    """

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
        segment_metadata: SegmentMetadata | None = None,
    ) -> tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"]]:
        """Segment-aware EMA via parallel associative scan.

        Each step is an affine map h[t] = A[t] * h[t-1] + b[t] with A[t] = q,
        except at segment starts where A[t] = 0 so no state crosses document
        boundaries in packed sequences. Composition of affine maps is
        associative, so the full trajectory is computed in log depth.

        Materializes (L, B, D, N) complex64 tensors for A and b. This is the
        production packed-training path; the sequential implementation is a
        fallback and correctness cross-check whose autodiff memory must be
        measured separately.

        :param Float[Array, "batch dim seq"] x: Input tensor (masked positions pre-zeroed).
        :param Int[Array, "batch seq"] segment_ids: Per-token segment IDs (0 = padding).
        :param SegmentMetadata | None segment_metadata: Optional shared derived metadata.
        :return tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"]]: Output
            tensor and final EMA state, anchored at each row's last non-padding
            token (zeros for all-padding rows).
        """
        p, q, gamma = self._coeffs()  # (D, N) each
        if segment_metadata is None:
            segment_metadata = derive_segment_metadata(segment_ids)

        # (L, B, 1, 1) reset flags broadcasting against coefficient (D, N) axes
        reset = jnp.moveaxis(segment_metadata.boundaries, -1, 0)[:, :, None, None]

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
            jnp.where(segment_metadata.valid, positions[None, :], -1), axis=1
        )  # (B,)
        h_final = h_seq[jnp.maximum(last_valid, 0), jnp.arange(B)]  # (B, D, N)
        h_final = jnp.where(
            (last_valid >= 0)[:, None, None], h_final, jnp.zeros((), dtype=h_final.dtype)
        )

        # Return fp32 - caller handles dtype conversion
        return y, h_final

    @staticmethod
    def _power_block(
        q: Complex[Array, "dim ndim"], start: Array | int, size: int
    ) -> Complex[Array, "dim ndim block"]:
        """Return stable complex powers ``q**j`` for one fixed-size block.

        :param Complex[Array, "dim ndim"] q: Complex decay coefficients.
        :param Array | int start: Starting exponent.
        :param int size: Static number of exponents.
        :return Complex[Array, "dim ndim block"]: Powers for the requested interval.
        """
        exponents = jnp.asarray(start, dtype=jnp.float32) + jnp.arange(size, dtype=jnp.float32)
        radius = jnp.abs(q).clip(max=1.0)
        phase = jnp.angle(q)
        magnitude = radius[:, :, None] ** exponents[None, None, :]
        rotation = jnp.exp(1j * phase[:, :, None] * exponents[None, None, :])
        return magnitude * rotation

    @classmethod
    def _power_chunk(
        cls, q: Complex[Array, "dim ndim"], start: int, end: int
    ) -> Complex[Array, "dim ndim chunk"]:
        """Return stable complex powers for a static interval.

        :param Complex[Array, "dim ndim"] q: Complex decay coefficients.
        :param int start: Starting exponent, inclusive.
        :param int end: Ending exponent, exclusive.
        :return Complex[Array, "dim ndim chunk"]: Powers for the requested interval.
        """
        return cls._power_block(q, start, end - start)

    def _forward_fft_with_coeffs(
        self,
        x: Float[Array, "batch dim seq"],
        p: Float[Array, "dim ndim"],
        q: Complex[Array, "dim ndim"],
        gamma: Complex[Array, "dim ndim"],
    ) -> Float[Array, "batch dim seq"]:
        """Apply FFT convolution using already-computed EMA coefficients.

        :param Float[Array, "batch dim seq"] x: Input tensor.
        :param Float[Array, "dim ndim"] p: Real input coefficients.
        :param Complex[Array, "dim ndim"] q: Complex decay coefficients.
        :param Complex[Array, "dim ndim"] gamma: Complex output coefficients.
        :return Float[Array, "batch dim seq"]: FP32 convolution output.
        """
        batch, dim, length = x.shape
        if length == 0:
            return jnp.zeros((batch, dim, length), dtype=jnp.float32)

        gamma_p = gamma * p

        def kernel_chunk(start: int, end: int) -> Complex[Array, "dim chunk"]:
            """Construct the convolution kernel over a static interval.

            :param int start: Starting exponent, inclusive.
            :param int end: Ending exponent, exclusive.
            :return Complex[Array, "dim chunk"]: Kernel values for the interval.
            """
            powers = self._power_chunk(q, start, end)
            return (gamma_p[:, :, None] * powers).sum(axis=1)

        if length <= FFT_KERNEL_CHUNK:
            kernel = kernel_chunk(0, length)
        else:
            num_blocks = math.ceil(length / FFT_KERNEL_CHUNK)
            block_size = math.ceil(length / num_blocks)

            def kernel_block(block_index: Array) -> Complex[Array, "dim block"]:
                """Construct one fixed-size convolution-kernel block.

                :param Array block_index: Zero-based block index.
                :return Complex[Array, "dim block"]: Kernel values for the block.
                """
                powers = self._power_block(q, block_index * block_size, block_size)
                return (gamma_p[:, :, None] * powers).sum(axis=1)

            block_ids = jnp.arange(num_blocks, dtype=jnp.int32)
            blocks = jax.lax.map(jax.checkpoint(kernel_block), block_ids)
            kernel = blocks.transpose(1, 0, 2).reshape(dim, num_blocks * block_size)[:, :length]

        fft_len = 1 << int(2 * length - 1).bit_length()
        inputs_fft = jnp.fft.rfft(x.astype(jnp.float32), n=fft_len, axis=-1)
        kernel_fft = jnp.fft.rfft(kernel.real, n=fft_len, axis=-1)
        return jnp.fft.irfft(inputs_fft * kernel_fft[None, :, :], n=fft_len, axis=-1)[..., :length]

    def _forward_fft(self, x: Float[Array, "batch dim seq"]) -> Float[Array, "batch dim seq"]:
        """Apply source-compatible FFT convolution.

        :param Float[Array, "batch dim seq"] x: Input tensor.
        :return Float[Array, "batch dim seq"]: FP32 convolution output.
        """
        if x.shape[-1] == 0:
            return jnp.zeros(x.shape, dtype=jnp.float32)
        return self._forward_fft_with_coeffs(x, *self._coeffs())

    def _initial_state_bias(
        self,
        h_init: Complex[Array, "batch dim ndim"],
        q: Complex[Array, "dim ndim"],
        gamma: Complex[Array, "dim ndim"],
        length: int,
    ) -> Float[Array, "batch dim seq"]:
        """Project the incoming recurrent state across a sequence.

        At timestep ``j`` the contribution is
        ``Re(sum(gamma * h_init * q ** (j + 1), axis=-1))``. Explicit
        real/imaginary reductions avoid an XLA complex batched-dot temporary.

        :param Complex[Array, "batch dim ndim"] h_init: Incoming recurrent state.
        :param Complex[Array, "dim ndim"] q: Complex decay coefficients.
        :param Complex[Array, "dim ndim"] gamma: Complex output coefficients.
        :param int length: Number of output timesteps.
        :return Float[Array, "batch dim seq"]: FP32 initial-state contribution.
        """
        projected = h_init * gamma[None, :, :] * q[None, :, :]

        def bias_chunk(start: Array | int, size: int) -> Float[Array, "batch dim chunk"]:
            """Project the incoming state over one interval.

            :param Array | int start: Starting exponent.
            :param int size: Static number of timesteps.
            :return Float[Array, "batch dim chunk"]: Initial-state contribution.
            """
            powers = self._power_block(q, start, size)
            return (
                projected.real[:, :, :, None] * powers.real[None, :, :, :]
                - projected.imag[:, :, :, None] * powers.imag[None, :, :, :]
            ).sum(axis=2)

        if length <= FFT_KERNEL_CHUNK:
            return bias_chunk(0, length)

        num_blocks = math.ceil(length / FFT_KERNEL_CHUNK)
        block_size = math.ceil(length / num_blocks)

        def bias_block(block_index: Array) -> Float[Array, "batch dim block"]:
            """Project the incoming state over one fixed-size block.

            :param Array block_index: Zero-based block index.
            :return Float[Array, "batch dim block"]: Initial-state contribution for the block.
            """
            return bias_chunk(block_index * block_size, block_size)

        block_ids = jnp.arange(num_blocks, dtype=jnp.int32)
        blocks = jax.lax.map(jax.checkpoint(bias_block), block_ids)
        return blocks.transpose(1, 2, 0, 3).reshape(
            h_init.shape[0], h_init.shape[1], num_blocks * block_size
        )[..., :length]

    def _final_state_parallel(
        self,
        x: Float[Array, "batch dim seq"],
        h_init: Complex[Array, "batch dim ndim"] | None,
        p: Float[Array, "dim ndim"],
        q: Complex[Array, "dim ndim"],
    ) -> Complex[Array, "batch dim ndim"]:
        """Compute the final recurrent state with parallel reductions.

        Evaluates ``q**L * h_init + p * sum(x[L-1-j] * q**j, j)``. This is the
        closed form used by the released parallel ``ema_hidden`` kernel.

        :param Float[Array, "batch dim seq"] x: Input sequence.
        :param Complex[Array, "batch dim ndim"] | None h_init: Incoming state.
        :param Float[Array, "dim ndim"] p: Real input coefficients.
        :param Complex[Array, "dim ndim"] q: Complex decay coefficients.
        :return Complex[Array, "batch dim ndim"]: Final complex64 state.
        """
        batch, dim, length = x.shape
        if length == 0:
            if h_init is None:
                return jnp.zeros((batch, dim, self.ndim), dtype=jnp.complex64)
            return h_init

        reversed_x = x.astype(jnp.float32)[..., ::-1]

        def state_contribution(
            values: Float[Array, "batch dim block"],
            powers: Complex[Array, "dim ndim block"],
        ) -> Complex[Array, "batch dim ndim"]:
            """Reduce one time block into its final-state contribution.

            :param Float[Array, "batch dim block"] values: Reversed input values.
            :param Complex[Array, "dim ndim block"] powers: Decay powers for the block.
            :return Complex[Array, "batch dim ndim"]: Complex state contribution.
            """
            real = (values[:, :, None, :] * powers.real[None, :, :, :]).sum(axis=-1)
            imag = (values[:, :, None, :] * powers.imag[None, :, :, :]).sum(axis=-1)
            return real + 1j * imag

        if length <= FFT_KERNEL_CHUNK:
            driven = state_contribution(reversed_x, self._power_chunk(q, 0, length))
        else:
            num_blocks = math.ceil(length / FFT_KERNEL_CHUNK)
            block_size = math.ceil(length / num_blocks)
            padded_length = num_blocks * block_size
            padded_x = jnp.pad(reversed_x, ((0, 0), (0, 0), (0, padded_length - length)))
            block_ids = jnp.arange(num_blocks, dtype=jnp.int32)

            def reduce_block(
                carry: Complex[Array, "batch dim ndim"], block_index: Array
            ) -> tuple[Complex[Array, "batch dim ndim"], None]:
                """Accumulate one block into the driven final state.

                :param Complex[Array, "batch dim ndim"] carry: Accumulated state contribution.
                :param Array block_index: Zero-based block index.
                :return tuple[Complex[Array, "batch dim ndim"], None]: Updated state and no output.
                """
                start = block_index * block_size
                values = jax.lax.dynamic_slice_in_dim(padded_x, start, block_size, axis=-1)
                powers = self._power_block(q, start, block_size)
                return carry + state_contribution(values, powers), None

            initial = jnp.zeros((batch, dim, self.ndim), dtype=jnp.complex64)
            driven, _ = jax.lax.scan(jax.checkpoint(reduce_block), initial, block_ids)
        h_final = p[None, :, :] * driven
        if h_init is not None:
            q_to_length = self._power_chunk(q, length, length + 1)[..., 0]
            h_final = h_final + q_to_length[None, :, :] * h_init
        return h_final

    def _forward_sequential(
        self,
        x: Float[Array, "batch dim seq"],
        h_init: Complex[Array, "batch dim ndim"] | None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        segment_metadata: SegmentMetadata | None = None,
    ) -> tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"]]:
        """Sequential recurrence using lax.scan.

        Computes the EMA recurrence h[t] = q * h[t-1] + p * x[t] using
        jax.lax.scan for efficiency. When segment_ids is given, the state is
        zeroed at segment boundaries (low-memory fallback to the parallel
        ``_forward_segmented`` path).

        :param Float[Array, "batch dim seq"] x: Input tensor.
        :param Complex[Array, "batch dim ndim"] | None h_init: Initial EMA state.
        :param Int[Array, "batch seq"] | None segment_ids: Optional per-token segment IDs.
        :param SegmentMetadata | None segment_metadata: Optional shared derived metadata.
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
            if segment_metadata is None:
                segment_metadata = derive_segment_metadata(segment_ids)
            reset_seq = jnp.moveaxis(segment_metadata.boundaries, -1, 0)  # (L, B)
            # Preserve the same last-real-token continuation contract as the FFT path.
            valid_seq = jnp.moveaxis(segment_metadata.valid, -1, 0)  # (L, B)

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

    def _final_state_sequential_with_coeffs(
        self,
        x: Float[Array, "batch dim seq"],
        h_init: Complex[Array, "batch dim ndim"] | None = None,
        *,
        p: Float[Array, "dim ndim"],
        q: Complex[Array, "dim ndim"],
    ) -> Complex[Array, "batch dim ndim"]:
        """Advance only the compact state using precomputed coefficients.

        :param Float[Array, "batch dim seq"] x: Input sequence.
        :param Complex[Array, "batch dim ndim"] | None h_init: Optional incoming EMA state.
        :param Float[Array, "dim ndim"] p: Real input coefficients.
        :param Complex[Array, "dim ndim"] q: Complex decay coefficients.
        :return Complex[Array, "batch dim ndim"]: Final recurrent state.
        """
        batch, dim, _ = x.shape
        if h_init is None:
            h_init = jnp.zeros((batch, dim, self.ndim), dtype=jnp.complex64)
        p_b = p[None, :, :]
        q_b = q[None, :, :]

        def step(h: Array, x_t: Array) -> tuple[Array, None]:
            """Advance the recurrent state by one timestep.

            :param Array h: Previous complex EMA state.
            :param Array x_t: Input vectors for one timestep.
            :return tuple[Array, None]: Updated state and an empty scan output.
            """
            x_t_c = x_t[:, :, None].astype(jnp.complex64)
            return q_b * h + p_b * x_t_c, None

        h_final, _ = jax.lax.scan(
            step,
            h_init,
            jnp.moveaxis(x.astype(jnp.float32), -1, 0),
        )
        return h_final

    def _final_state_sequential(
        self,
        x: Float[Array, "batch dim seq"],
        h_init: Complex[Array, "batch dim ndim"] | None = None,
    ) -> Complex[Array, "batch dim ndim"]:
        """Advance only the compact recurrent state without emitting outputs.

        :param Float[Array, "batch dim seq"] x: Input sequence.
        :param Complex[Array, "batch dim ndim"] | None h_init: Optional incoming EMA state.
        :return Complex[Array, "batch dim ndim"]: Final recurrent state.
        """
        p, q, _ = self._coeffs()
        return self._final_state_sequential_with_coeffs(x, h_init, p=p, q=q)

    def __call__(
        self,
        x: Float[Array, "batch dim seq"],
        h_init: Complex[Array, "batch dim ndim"] | None = None,
        return_state: bool = False,
        mask: Bool[Array, "batch seq"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        use_associative_segment_scan: bool = True,
        *,
        _segment_metadata: SegmentMetadata | None = None,
    ) -> tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"] | None]:
        """Apply EMA and optionally return final state.

        Uses FFT convolution for training and long non-segmented chunks, with
        a parallel closed-form state reduction when continuation state is
        requested. Short streaming chunks use a sequential scan. When
        segment_ids is given (packed sequences), the state resets at segment
        boundaries via a parallel associative scan; the FFT path cannot
        express resets and is bypassed.

        :param Float[Array, "batch dim seq"] x: Input tensor.
        :param Complex[Array, "batch dim ndim"] | None h_init: Initial EMA state.
        :param bool return_state: Whether to return the final complex state.
        :param Bool[Array, "batch seq"] | None mask: Optional validity mask.
        :param Int[Array, "batch seq"] | None segment_ids: Optional per-token segment IDs
            (0 = padding) for packed-sequence state resets. Training-only:
            incompatible with h_init.
        :param bool use_associative_segment_scan: Use the parallel associative scan for
            the segmented path; False selects the sequential fallback. The latter has a
            compact forward carry but is not guaranteed to reduce compiled backward memory.
        :param SegmentMetadata | None _segment_metadata: Internal per-model-call metadata.
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
            if _segment_metadata is None:
                _segment_metadata = derive_segment_metadata(segment_ids)
            # Preserve the same last-real-token continuation contract as the FFT path.
            seg_valid = _segment_metadata.valid
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
                y, h_final = self._forward_segmented(x, segment_ids, _segment_metadata)
            else:
                y, h_final = self._forward_sequential(
                    x,
                    None,
                    segment_ids=segment_ids,
                    segment_metadata=_segment_metadata,
                )
            return (y + residual).astype(input_dtype), h_final if return_state else None

        length = x.shape[-1]
        if length == 0:
            if return_state:
                if h_init is None:
                    h_final = jnp.zeros((x.shape[0], x.shape[1], self.ndim), dtype=jnp.complex64)
                else:
                    h_final = h_init
            else:
                h_final = None
            return residual.astype(input_dtype), h_final

        # Tokenwise decoding and very short cached chunks are faster as one
        # recurrent scan. The length is shape-static, so this does not add a
        # traced data-dependent branch.
        if h_init is not None and length < FFT_RECURRENT_MIN_LENGTH:
            y, h_final = self._forward_sequential(x, h_init)
            return (y + residual).astype(input_dtype), h_final if return_state else None

        # Released long-chunk decomposition: FFT convolution for the driven
        # response, plus a projected initial-state bias when history exists.
        # Compute coefficients once for the output and optional final state.
        p, q, gamma = self._coeffs()
        y = self._forward_fft_with_coeffs(x, p, q, gamma)
        state_q = q
        if h_init is not None:
            bias_q = jax.lax.optimization_barrier(q)
            y = y + self._initial_state_bias(h_init, bias_q, gamma, length)
            state_q = jax.lax.optimization_barrier(bias_q)
        elif return_state and length >= FFT_RECURRENT_MIN_LENGTH:
            state_q = jax.lax.optimization_barrier(q)

        h_final = None
        if return_state:
            if length < FFT_RECURRENT_MIN_LENGTH:
                h_final = self._final_state_sequential_with_coeffs(x, p=p, q=q)
            else:
                # The FFT, optional bias, and state reduction use the same
                # powers. Recomputing them lets XLA release each chunk instead
                # of retaining a (D, N, chunk) complex buffer across the FFT.
                h_final = self._final_state_parallel(x, h_init, p, state_q)
        return (y + residual).astype(input_dtype), h_final
