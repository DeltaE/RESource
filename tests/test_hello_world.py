import importlib
from importlib.metadata import entry_points

import pytest


def test_public_package_metadata():
    import RESource

    assert RESource.__version__ == "2025.7.0"


def test_legacy_namespace_remains_available():
    with pytest.warns(DeprecationWarning, match="use 'RESource'"):
        legacy_package = importlib.import_module("RES")

    assert legacy_package.__version__ == "2025.7.0"


def test_command_entry_points_are_installed():
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}

    assert scripts["resource"] == "RESource.cli:entrypoint"
    assert scripts["resource-multiyear"] == "RESource.multiyear:entrypoint"
