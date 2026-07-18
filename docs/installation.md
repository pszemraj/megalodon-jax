# Installation and setup

Install a released version from PyPI unless you are deliberately testing unreleased code or contributing to the project.

## Requirements

- Python 3.11 or newer
- A JAX-supported CPU, or an NVIDIA GPU on Linux using the pip-managed CUDA 13 runtime
- Linux x86_64 or aarch64 for the `cuda13` extra

The supported dependency ranges are defined in [`pyproject.toml`](../pyproject.toml).

## Install from PyPI

### CPU

```bash
pip install megalodon-jax
```

### NVIDIA GPU

```bash
pip install "megalodon-jax[cuda13]"
```

The `cuda13` extra installs JAX's CUDA plugin, PJRT, CUDA, and cuDNN wheels. It does not require a separately installed CUDA toolkit, but the NVIDIA driver must support the bundled runtime.

## Verify the installation

```bash
python - <<'PY'
from importlib.metadata import version

import jax

print("megalodon-jax", version("megalodon-jax"))
print("JAX backend", jax.default_backend())
print("JAX devices", jax.devices())
PY
```

## Optional checkpoint conversion

Install the `convert` extra when converting original-upstream PyTorch checkpoints:

```bash
pip install "megalodon-jax[convert]"
```

Combine it with the NVIDIA runtime when both are needed:

```bash
pip install "megalodon-jax[cuda13,convert]"
```

## Install unreleased `main`

For CPU:

```bash
pip install "megalodon-jax @ git+https://github.com/pszemraj/megalodon-jax.git"
```

For NVIDIA GPUs:

```bash
pip install "megalodon-jax[cuda13] @ git+https://github.com/pszemraj/megalodon-jax.git"
```

These commands track the current `main` branch rather than a released PyPI version.

## Development install

For CPU development:

```bash
git clone https://github.com/pszemraj/megalodon-jax.git
cd megalodon-jax
pip install -e ".[dev]"
```

For NVIDIA development:

```bash
git clone https://github.com/pszemraj/megalodon-jax.git
cd megalodon-jax
pip install -e ".[cuda13,dev]"
```

The `dev` extra includes tests, linting, PyTorch conversion, and parity tooling. Continue with the [development guide](dev.md) for repository commands and test gates.

## CUDA troubleshooting

- CUDA 12 and locally managed CUDA installations are not supported package profiles.
- Keep `LD_LIBRARY_PATH` unset when using the pip-managed CUDA libraries; pointing it at another CUDA toolkit can make XLA load incompatible libraries.
- Check `jax.default_backend()` and `jax.devices()` before debugging model code. PyTorch may install different CUDA vendor wheels independently, so its package list does not identify the active JAX runtime.
