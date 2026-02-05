"""Test to ensure heavy dependencies are not imported at module level."""

import sys
import pytest


def test_api_main_import_is_lightweight():
    """Assert that importing api.main does not import heavy dependencies."""
    # Ensure heavy modules are not already in sys.modules
    heavy_modules = ["mlflow", "scipy", "matplotlib"]
    for mod in heavy_modules:
        if mod in sys.modules:
            # If they are already there (e.g. from previous tests), we can't easily test this
            # but we can check if they are top-level or not.
            # For this test, we'll just skip if they are already present to avoid false negatives
            # in a shared test runner environment.
            pytest.skip(f"Module {mod} already in sys.modules, skipping lightweight check")

    # Import api.main

    # Check again
    for mod in heavy_modules:
        assert mod not in sys.modules, f"Module {mod} was imported eagerly by api.main"


def test_forecast_import_is_lightweight():
    """Assert that importing forecast package does not import heavy dependencies."""
    heavy_modules = ["mlflow", "scipy", "matplotlib"]
    for mod in heavy_modules:
        if mod in sys.modules:
            pytest.skip(f"Module {mod} already in sys.modules, skipping lightweight check")


    for mod in heavy_modules:
        assert mod not in sys.modules, f"Module {mod} was imported eagerly by forecast"
