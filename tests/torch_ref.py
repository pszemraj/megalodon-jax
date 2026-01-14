"""Helpers for importing the external megalodon-hf package in tests."""

from __future__ import annotations

import importlib
from types import ModuleType


def _import_external(name: str) -> ModuleType:
    """Import a module from the external megalodon-hf package.

    :param str name: Module import path.
    :return ModuleType: Imported module instance.
    """
    return importlib.import_module(name)


def megalodon() -> object:
    """Import the external megalodon package.

    :return object: The imported megalodon package module.
    """
    return _import_external("megalodon")


def modeling() -> object:
    """Import megalodon.modeling_megalodon from the external package.

    :return object: The imported modeling module from megalodon-hf.
    """
    _import_external("megalodon")
    return _import_external("megalodon.modeling_megalodon")
