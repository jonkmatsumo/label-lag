"""Shared pytest fixtures for testing."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


class FakeAnalyticsClient:
    """In-memory fake analytics client for testing."""

    def __init__(self):
        self.data = {}

    def clear_all_data(self):
        response = MagicMock()
        response.success = True
        response.tables_cleared = ["generated_records", "evaluation_metadata"]
        return response


@pytest.fixture
def fake_analytics_client():
    return FakeAnalyticsClient()


@pytest.fixture
def mock_crud_client(fake_analytics_client, monkeypatch):
    monkeypatch.setattr("training.crud_client._client", fake_analytics_client)
    return fake_analytics_client


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
