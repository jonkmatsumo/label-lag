"""Shared pytest fixtures for rule testing."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from api.rules import Rule, RuleSet


class FakeAnalyticsClient:
    """In-memory fake analytics client for testing.

    Provides deterministic responses without network calls.
    """

    def __init__(self):
        self.data = {
            "daily_stats": [],
            "transactions": [],
            "alerts": [],
            "backtest_results": [],
            "audit_logs": [],
        }

    def get_daily_stats(self, days=30):
        response = MagicMock()
        response.stats = self.data.get("daily_stats", [])
        return response

    def get_transaction_details(self, days=7, limit=1000):
        response = MagicMock()
        response.transactions = self.data.get("transactions", [])[:limit]
        return response

    def get_recent_alerts(self, limit=50):
        response = MagicMock()
        response.alerts = self.data.get("alerts", [])[:limit]
        return response

    def search_transactions(self, **kwargs):
        response = MagicMock()
        transactions = self.data.get("transactions", [])
        if kwargs.get("user_id"):
            transactions = [
                t
                for t in transactions
                if getattr(t, "user_id", "") == kwargs["user_id"]
            ]
        limit = kwargs.get("limit", 100)
        response.transactions = transactions[:limit]
        response.total = len(transactions)
        return response

    def get_features(self, user_id):
        for t in self.data.get("transactions", []):
            if getattr(t, "user_id", "") == user_id:
                return t
        return None

    def get_training_data(self, cutoff_date):
        response = MagicMock()
        response.train_records = self.data.get("train_records", [])
        response.test_records = self.data.get("test_records", [])
        return response

    def store_generated_data(self, records, metadata):
        response = MagicMock()
        response.success = True
        response.records_saved = len(records)
        return response

    def clear_all_data(self):
        response = MagicMock()
        response.success = True
        response.tables_cleared = [
            "generated_records",
            "evaluation_metadata",
            "feature_snapshots",
        ]
        self.data = {
            "daily_stats": [],
            "transactions": [],
            "alerts": [],
            "backtest_results": [],
            "audit_logs": [],
        }
        return response

    def materialize_features(self, batch_size=1000):
        response = MagicMock()
        response.success = True
        response.total_processed = len(self.data.get("transactions", []))
        return response

    def get_overview_metrics(self):
        response = MagicMock()
        transactions = self.data.get("transactions", [])
        response.total_records = len(transactions)
        response.fraud_records = sum(
            1 for t in transactions if getattr(t, "is_fraudulent", False)
        )
        response.fraud_rate = (
            response.fraud_records / response.total_records
            if response.total_records
            else 0.0
        )
        response.unique_users = len(
            set(getattr(t, "user_id", "") for t in transactions)
        )
        response.total_amount = sum(getattr(t, "amount", 0) for t in transactions)
        response.fraud_amount = sum(
            getattr(t, "amount", 0)
            for t in transactions
            if getattr(t, "is_fraudulent", False)
        )
        return response

    def get_dataset_fingerprint(self):
        response = MagicMock()
        response.generated_records = MagicMock()
        response.generated_records.count = len(self.data.get("transactions", []))
        response.feature_snapshots = MagicMock()
        response.feature_snapshots.count = len(self.data.get("transactions", []))
        return response

    def get_feature_sample(self, sample_size=100, stratify=False):
        response = MagicMock()
        response.samples = self.data.get("feature_samples", [])[:sample_size]
        return response

    def get_schema_summary(self, table_names=None):
        response = MagicMock()
        response.columns = []
        return response

    def get_backtest_features(self, start_date, end_date):
        response = MagicMock()
        response.features = self.data.get("backtest_features", [])
        return response

    def save_backtest_result(self, result):
        self.data.setdefault("backtest_results", []).append(result)
        response = MagicMock()
        response.success = True
        return response

    def list_backtest_results(self, rule_id=None, start_date=None, end_date=None):
        response = MagicMock()
        results = self.data.get("backtest_results", [])
        if rule_id:
            results = [r for r in results if getattr(r, "rule_id", "") == rule_id]
        response.results = results
        return response

    def get_backtest_result(self, job_id):
        response = MagicMock()
        for r in self.data.get("backtest_results", []):
            if getattr(r, "job_id", "") == job_id:
                response.result = r
                return response
        response.result = None
        return response

    def get_drift_window(self, hours):
        response = MagicMock()
        response.transactions = self.data.get(
            "drift_transactions", self.data.get("transactions", [])
        )
        return response


@pytest.fixture
def fake_analytics_client():
    """Fixture providing a fresh FakeAnalyticsClient instance."""
    return FakeAnalyticsClient()


@pytest.fixture
def mock_crud_client(fake_analytics_client, monkeypatch):
    """Fixture that patches get_crud_client to return the fake client.

    Use this fixture when testing endpoints that call get_crud_client().
    """
    monkeypatch.setattr("api.crud_client._client", fake_analytics_client)
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
