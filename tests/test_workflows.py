"""Regression tests for repository automation workflows."""

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.fast


def test_pypi_checkout_is_bound_to_release_event_commit() -> None:
    """Published artifacts come from the commit recorded by the release event."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/publish.yml").read_text()
    expected = """\
        with:
          ref: ${{ github.sha }}
          fetch-depth: 0
          persist-credentials: false
"""
    assert expected in workflow
    assert "ref: ${{ github.event.release.tag_name }}" not in workflow


def test_pypi_concurrency_is_scoped_to_release_tag() -> None:
    """Publishing a newer version cannot evict an older pending release run."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/publish.yml").read_text()
    expected = """\
concurrency:
  group: pypi-publish-${{ github.event.release.tag_name }}
  cancel-in-progress: false
"""
    assert expected in workflow
