"""Guardrail tests to ensure no regression into using Python rules engine.

Ensures that decisioning logic remains in the Go gateway.
"""

from pathlib import Path

import pytest


def test_no_legacy_signal_evaluation_route():
    """Assert that /evaluate/signal route no longer exists in main.py."""
    main_py = Path("src/api/main.py").read_text()
    assert "/evaluate/signal" not in main_py


def test_no_legacy_decisioning_methods():
    """Assert that SignalForecaster no longer has legacy decisioning methods."""
    services_py = Path("src/forecast/services.py").read_text()
    assert "def evaluate(" not in services_py
    assert "def _apply_rules(" not in services_py
    assert "def _identify_risk_components(" not in services_py


def test_sandbox_no_python_decisioning():
    """Assert sandbox doesn't use rules_management.rules.evaluate_rules."""
    main_py = Path("src/api/main.py").read_text()

    # Verify evaluate_rules is NOT imported in main.py from rules_management.rules
    assert "from rules_management.rules import evaluate_rules" not in main_py
    assert "import rules_management.rules.evaluate_rules" not in main_py


def test_backtest_no_python_decisioning():
    """Assert backtest doesn't use rules_management.rules.evaluate_rules."""
    backtest_py = Path("src/rules_management/backtest.py").read_text()

    # Assert no import of evaluate_rules from rules_management.rules
    assert "from rules_management.rules import evaluate_rules" not in backtest_py

    # Assert it uses gateway_client
    assert "get_gateway_client" in backtest_py


def test_no_remaining_decisioning_calls():
    """Search for any remaining calls to evaluate_rules in critical paths."""

    # Check all files in src/api except main.py (which might have it in docs/strings)
    api_dir = Path("src/api")
    for py_file in api_dir.glob("*.py"):
        # rules.py defines it, llm_rules.py might use it,
        # gateway_client defines a method with same name
        if py_file.name in ["rules.py", "llm_rules.py", "gateway_client.py"]:
            continue

        content = py_file.read_text()

        # Check if evaluate_rules is imported from rules_management.rules
        forbidden_import = (
            "from rules_management.rules import evaluate_rules" in content
            or "import rules_management.rules.evaluate_rules" in content
        )

        if forbidden_import:
            pytest.fail(f"Forbidden import of evaluate_rules found in {py_file}")

        # Also check for direct calls if rules_management.rules was imported as a whole
        if "rules_management.rules.evaluate_rules(" in content:
            pytest.fail(
                "Forbidden call to rules_management.rules.evaluate_rules found "
                f"in {py_file}"
            )
