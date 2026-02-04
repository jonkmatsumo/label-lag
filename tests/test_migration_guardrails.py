"""Guardrail tests to ensure no regression into using Python rules engine for decisioning."""

import os
import re
from pathlib import Path

import pytest


def test_sandbox_no_python_decisioning():
    """Assert that sandbox implementation doesn't use api.rules.evaluate_rules."""
    main_py = Path("src/api/main.py").read_text()
    
    # Verify evaluate_rules is NOT imported in main.py from api.rules
    # It might still be imported in other ways, but we check for common patterns
    assert "from api.rules import evaluate_rules" not in main_py
    assert "import api.rules.evaluate_rules" not in main_py


def test_backtest_no_python_decisioning():
    """Assert that backtest implementation doesn't use api.rules.evaluate_rules."""
    backtest_py = Path("src/api/backtest.py").read_text()
    
    # Assert no import of evaluate_rules from api.rules
    assert "from api.rules import evaluate_rules" not in backtest_py
    
    # Assert it uses gateway_client
    assert "get_gateway_client" in backtest_py


def test_no_remaining_decisioning_calls():
    """Search for any remaining calls to evaluate_rules in critical paths."""
    # We allow evaluate_rules in tests and in the legacy /evaluate/signal route for now
    
    # Check all files in src/api except main.py (which has the legacy route)
    api_dir = Path("src/api")
    for py_file in api_dir.glob("*.py"):
        # rules.py defines it, services.py uses it for legacy route, 
        # llm_rules.py might use it, gateway_client defines a method with same name
        if py_file.name in ["main.py", "rules.py", "services.py", "llm_rules.py", "gateway_client.py"]:
            continue
            
        content = py_file.read_text()
        
        # Check if evaluate_rules is imported from api.rules
        forbidden_import = "from api.rules import evaluate_rules" in content or "import api.rules.evaluate_rules" in content
        
        if forbidden_import:
             pytest.fail(f"Forbidden import of evaluate_rules found in {py_file}")
             
        # Also check for direct calls if api.rules was imported as a whole
        if "api.rules.evaluate_rules(" in content:
             pytest.fail(f"Forbidden call to api.rules.evaluate_rules found in {py_file}")
