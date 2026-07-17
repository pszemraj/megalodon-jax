"""Regression tests for repository automation workflows."""

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.fast


def test_pypi_concurrency_is_scoped_to_release_tag() -> None:
    """Publishing a newer version cannot evict an older pending release run."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/publish.yml").read_text()
    expected = """\
concurrency:
  group: pypi-publish-${{ github.event.release.tag_name }}
  cancel-in-progress: false
"""
    assert expected in workflow
