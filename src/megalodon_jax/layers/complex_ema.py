"""Complex Exponential Moving Average (EMA) for Megalodon JAX.

ComplexEMA is the core long-range memory mechanism in Megalodon. It uses
complex-valued coefficients to enable richer temporal dynamics than real EMA.

The recurrence is:
    h[t] = q * h[t-1] + p * x[t]
    y[t] = Re(sum(h[t] * gamma))

Two computation paths are provided:
- FFT: O(L log L) convolution for training (no state needed)
- Sequential: O(L) scan for streaming inference (state needed)
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float, PRNGKeyArray

# Chunk size for kernel computation to bound memory usage
FFT_KERNEL_CHUNK = 4096

# Warn if sequence length exceeds this threshold
FFT_LENGTH_WARN_THRESHOLD = 16_384


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

        Args:
            embed_dim: Hidden dimension D of the input tensor.
            ndim: Number of EMA orders tracked per hidden unit.
            key: PRNG key for random initialization.
        """
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.scale = math.sqrt(1.0 / float(ndim))

        # Split key for different initializations
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # alpha & delta (logit space): normal(0, 0.2)
        self.alpha = jax.random.normal(k1, (embed_dim, ndim, 1)) * 0.2
        self.delta = jax.random.normal(k2, (embed_dim, ndim, 1)) * 0.2

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
        self.gamma_real = jax.random.normal(k4, (embed_dim, ndim))
        self.gamma_imag = jnp.zeros((embed_dim, ndim))

        # omega: truncated normal(0, 0.25, -1, 1)
        self.omega = jax.random.truncated_normal(k5, -4.0, 4.0, (embed_dim,)) * 0.25

    def _coeffs(
        self,
    ) -> tuple[Float[Array, "dim ndim"], Complex[Array, "dim ndim"], Complex[Array, "dim ndim"]]:
        """Compute EMA coefficients (p, q, gamma).

        All computations are forced to fp32/complex64 for numerical stability,
        regardless of the parameter dtype. This matches the PyTorch reference
        behavior and prevents precision loss when the model is cast to bf16.

        Returns:
            Tuple of (p, q, gamma) where:
            - p: Real input coefficient (D, N), dtype float32
            - q: Complex recurrence coefficient with |q| < 1 (D, N), dtype complex64
            - gamma: Complex output projection (D, N), dtype complex64
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
        gamma = (gamma_real_f32 + 1j * gamma_imag_f32) * self.scale  # (D, N) complex64

        return p, q, gamma

    @staticmethod
    def _real_of_product(a: Complex[Array, "..."], b: Complex[Array, "..."]) -> Float[Array, "..."]:
        """Compute real part of complex product efficiently.

        Re(a * b) = a.real * b.real - a.imag * b.imag
        """
        return a.real * b.real - a.imag * b.imag

    def _forward_fft(self, x: Float[Array, "batch dim seq"]) -> Float[Array, "batch dim seq"]:
        """FFT-based convolution for training.

        Uses O(L log L) FFT convolution when no streaming state is needed.

        Args:
            x: Input tensor of shape (batch, dim, seq).

        Returns:
            Output tensor of shape (batch, dim, seq).
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

            Args:
                start: Starting position index (inclusive).
                end: Ending position index (exclusive).

            Returns:
                Complex kernel slice of shape (dim, end-start).
            """
            j = jnp.arange(start, end, dtype=jnp.float32)  # (chunk,)
            # q^j = radius^j * exp(i * phi * j)
            mag_chunk = radius[:, :, None] ** j[None, None, :]  # (D, N, chunk)
            phase_chunk = jnp.exp(1j * phi[:, :, None] * j[None, None, :])  # (D, N, chunk)
            q_pows = mag_chunk * phase_chunk  # (D, N, chunk) complex
            # Sum over N dimension
            return (gp[:, :, None] * q_pows).sum(axis=1)  # (D, chunk) complex

        # Compute full kernel
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
    ) -> tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"]]:
        """Sequential recurrence using lax.scan.

        Computes the EMA recurrence h[t] = q * h[t-1] + p * x[t] using
        jax.lax.scan for efficiency.

        Args:
            x: Input tensor of shape (batch, dim, seq).
            h_init: Optional initial complex state of shape (batch, dim, ndim).

        Returns:
            Tuple of (output, final_state) where output has shape (batch, dim, seq)
            and final_state has shape (batch, dim, ndim).
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
            """Single EMA step."""
            x_t_c = x_t[:, :, None].astype(jnp.complex64)  # (B, D, 1)
            h_new = q_b * h + p_b * x_t_c  # (B, D, N)
            y_t = self._real_of_product(h_new, gamma_b).sum(axis=-1)  # (B, D)
            return h_new, y_t

        # x is (B, D, L), scan over L
        x_transposed = jnp.moveaxis(x.astype(jnp.float32), -1, 0)  # (L, B, D)
        h_final, y_seq = jax.lax.scan(step, h_init, x_transposed)
        y = jnp.moveaxis(y_seq, 0, -1)  # (B, D, L)

        # Return fp32 - caller handles dtype conversion
        return y, h_final

    def __call__(
        self,
        x: Float[Array, "batch dim seq"],
        h_init: Complex[Array, "batch dim ndim"] | None = None,
        return_state: bool = False,
        mask: Bool[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch dim seq"], Complex[Array, "batch dim ndim"] | None]:
        """Apply EMA and optionally return final state.

        Automatically selects FFT path (faster) when no state is needed,
        or sequential path when streaming.

        Args:
            x: Input tensor of shape (batch, dim, seq).
            h_init: Optional initial EMA state for streaming inference.
            return_state: Whether to return the final complex state.
            mask: Optional boolean mask of shape (batch, seq) where True marks
                valid tokens. Masked positions (False) are zeroed before EMA
                processing to prevent contamination of the hidden state.

        Returns:
            Tuple of (output, state) where state is None unless return_state=True
            or h_init was provided. Masked positions in output will be zero.
        """
        # Store input dtype for output cast
        input_dtype = x.dtype

        # Zero masked positions to prevent EMA state contamination from padding
        # This is applied at input level since EMA is a linear operation:
        # EMA(0) = 0, so masked positions don't contribute to state or output
        if mask is not None:
            # mask: (batch, seq) -> (batch, 1, seq) for broadcasting with x: (batch, dim, seq)
            x = jnp.where(mask[:, None, :], x, 0.0)

        # Cast omega to fp32 for residual computation (matches PyTorch reference)
        omega_f32 = self.omega.astype(jnp.float32)
        x_f32 = x.astype(jnp.float32)
        residual = x_f32 * omega_f32[None, :, None]

        use_fft = h_init is None and not return_state
        if use_fft:
            y = self._forward_fft(x)  # already returns fp32 internally
            return (y + residual).astype(input_dtype), None

        y, h_final = self._forward_sequential(x, h_init)  # already returns fp32 internally
        return (y + residual).astype(input_dtype), h_final if return_state else None
