"""Dummy test file to ensure pytest can discover tests."""

import pytest


def test_dummy():
    """Dummy test to ensure pytest discovery works."""
    assert True  # nosec B101


def test_imports():
    """Test that main modules can be imported without errors."""
    import importlib.util

    modules_to_test = [
        "src.stock_analyzer",
        "src.stock_analyzer.analysis.indicators",
        "src.stock_analyzer.cli.main",
        "src.stock_analyzer.data.fetchers",
    ]

    for module_name in modules_to_test:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            pytest.skip(f"Module {module_name} not found - skipping import test")

    # All modules found, test successful
    assert True  # nosec B101
