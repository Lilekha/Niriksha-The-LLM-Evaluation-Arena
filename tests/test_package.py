import importlib
from importlib.metadata import version

import pytest

import niriksha


def test_version_matches_installed_metadata():
    # Fails if pyproject.toml and the package disagree, or the src layout is not installed.
    assert niriksha.__version__ == version("niriksha")


@pytest.mark.parametrize("name", ["core", "providers", "scorers"])
def test_subpackages_importable(name):
    importlib.import_module(f"niriksha.{name}")
