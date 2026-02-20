"""Test to ensure heavy dependencies are not imported at module level."""

import sys

import pytest


def test_api_main_import_is_lightweight():
    """Assert that importing training.server does not import heavy dependencies."""
    # Ensure heavy modules are not already in sys.modules
    heavy_modules = ["mlflow", "scipy", "matplotlib"]
    for mod in heavy_modules:
        if mod in sys.modules:
            # If already present (e.g., from previous tests), we can't reliably
            # assert import side effects in this test.
            # but we can check if they are top-level or not.
            # For this test, skip to avoid false negatives.
            # in a shared test runner environment.
            pytest.skip(
                f"Module {mod} already in sys.modules, skipping lightweight check"
            )

    # Import training.server
    import training.server  # noqa: F401

    # Check again
    for mod in heavy_modules:
        assert mod not in sys.modules, (
            f"Module {mod} was imported eagerly by training.server"
        )


def test_forecast_import_is_lightweight():
    """Assert that importing forecast package does not import heavy dependencies."""
    heavy_modules = ["mlflow", "scipy", "matplotlib"]
    for mod in heavy_modules:
        if mod in sys.modules:
            pytest.skip(
                f"Module {mod} already in sys.modules, skipping lightweight check"
            )

    # Import forecast
    import forecast  # noqa: F401

    for mod in heavy_modules:
        assert mod not in sys.modules, f"Module {mod} was imported eagerly by forecast"
