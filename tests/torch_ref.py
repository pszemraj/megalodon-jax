"""Helpers for importing the external megalodon-hf package in tests."""

from __future__ import annotations

import importlib
import site
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_MEGALODON = _PROJECT_ROOT / "src" / "megalodon"


def _prefer_site_packages() -> None:
    """Ensure site-packages is ahead of the repo on sys.path."""
    site_paths: list[str] = []
    for path in site.getsitepackages():
        if path and path not in site_paths:
            site_paths.append(path)
    user_site = site.getusersitepackages()
    if user_site and user_site not in site_paths:
        site_paths.append(user_site)
    for path in sys.path:
        if ("site-packages" in path or "dist-packages" in path) and path not in site_paths:
            site_paths.append(path)
    remainder = [path for path in sys.path if path not in site_paths]
    sys.path[:] = site_paths + remainder


def _ensure_external(module: object, name: str) -> object:
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        return module
    resolved = Path(module_path).resolve()
    if _LOCAL_MEGALODON in resolved.parents:
        raise RuntimeError(
            f"{name} resolved to local src/megalodon. Install megalodon-hf and ensure "
            "site-packages precedes the repo root on sys.path."
        )
    return module


def _import_external(name: str) -> object:
    module = sys.modules.get(name)
    if module is not None:
        return _ensure_external(module, name)
    _prefer_site_packages()
    module = importlib.import_module(name)
    return _ensure_external(module, name)


def megalodon() -> object:
    """Import the external megalodon package from site-packages."""
    return _import_external("megalodon")


def modeling() -> object:
    """Import megalodon.modeling_megalodon from the external package."""
    _import_external("megalodon")
    return _import_external("megalodon.modeling_megalodon")
