"""Shared pytest fixtures for rule testing."""

import json
from pathlib import Path
from typing import Any

import pytest

from api.rules import Rule, RuleSet


@pytest.fixture
def sample_features() -> dict[str, Any]:
    """Sample feature dictionary for testing."""
    return {
        "velocity_24h": 10,
        "amount_to_avg_ratio_30d": 3.5,
        "balance_volatility_z_score": -1.5,
        "bank_connections_24h": 5,
        "merchant_risk_score": 75,
        "has_history": True,
        "transaction_amount": 1000.0,
    }


@pytest.fixture
def sample_ruleset() -> RuleSet:
    """Sample ruleset for testing."""
    rules = [
        Rule(
            id="velocity_high",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            severity="high",
            reason="High transaction velocity",
        ),
        Rule(
            id="amount_ratio_high",
            field="amount_to_avg_ratio_30d",
            op=">",
            value=3.0,
            action="clamp_min",
            score=75,
            severity="medium",
        ),
    ]
    return RuleSet(version="v1", rules=rules)


@pytest.fixture
def all_operators() -> list[str]:
    """List of all supported operators."""
    return [">", ">=", "<", "<=", "==", "in", "not_in"]


@pytest.fixture
def all_actions() -> list[str]:
    """List of all supported actions."""
    return ["override_score", "clamp_min", "clamp_max", "reject"]


@pytest.fixture
def golden_file_path(tmp_path: Path) -> Path:
    """Path to a temporary golden file directory."""
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    return golden_dir


def load_golden_file(golden_dir: Path, test_name: str) -> dict[str, Any] | None:
    """Load a golden file if it exists.

    Args:
        golden_dir: Directory containing golden files.
        test_name: Name of the test (used as filename).

    Returns:
        Golden file contents as dict, or None if file doesn't exist.
    """
    golden_file = golden_dir / f"{test_name}.json"
    if not golden_file.exists():
        return None
    with open(golden_file) as f:
        return json.load(f)


def save_golden_file(golden_dir: Path, test_name: str, data: dict[str, Any]) -> None:
    """Save a golden file.

    Args:
        golden_dir: Directory containing golden files.
        test_name: Name of the test (used as filename).
        data: Data to save.
    """
    golden_file = golden_dir / f"{test_name}.json"
    with open(golden_file, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
