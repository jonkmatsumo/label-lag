"""Tests for forecast fallback observability metrics (Phase D)."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from forecast.metrics import (
    heuristic_fallback_total,
    model_fallback_total,
    zero_fallback_total,
)
from forecast.services import SignalForecaster
from training.schemas import SignalRequest


@pytest.fixture(autouse=True)
def _reset_counters():
    """Reset all counters before each test."""
    model_fallback_total.reset()
    heuristic_fallback_total.reset()
    zero_fallback_total.reset()
    yield


@pytest.fixture
def forecaster():
    return SignalForecaster()


@pytest.fixture
def mock_manager():
    with patch("forecast.model_manager.get_model_manager") as mock:
        manager = MagicMock()
        mock.return_value = manager
        yield manager


def _make_request(fallback_mode=None):
    return SignalRequest(
        user_id="user1",
        amount=Decimal("100.0"),
        client_transaction_id="tx1",
        fallback_mode=fallback_mode,
    )


def _mock_features():
    features = MagicMock()
    features.has_history = False
    features.velocity_24h = 0
    features.amount_to_avg_ratio_30d = 1.0
    features.balance_volatility_z_score = 0.0
    features.bank_connections_24h = 0
    features.merchant_risk_score = 0
    return features


class TestHeuristicFallbackCounter:
    def test_heuristic_fallback_increments(self, forecaster, mock_manager):
        mock_manager.model_loaded = False

        with patch.object(forecaster, "_fetch_features") as mock_fetch:
            mock_fetch.return_value = _mock_features()
            forecaster.predict(_make_request(fallback_mode="probability"))

        assert heuristic_fallback_total.value == 1

    def test_heuristic_fallback_increments_on_multiple_calls(
        self, forecaster, mock_manager
    ):
        mock_manager.model_loaded = False

        with patch.object(forecaster, "_fetch_features") as mock_fetch:
            mock_fetch.return_value = _mock_features()
            forecaster.predict(_make_request(fallback_mode="probability"))
            forecaster.predict(_make_request(fallback_mode="probability"))

        assert heuristic_fallback_total.value == 2


class TestZeroFallbackCounter:
    def test_zero_fallback_increments(self, forecaster, mock_manager):
        mock_manager.model_loaded = False

        with patch.object(forecaster, "_fetch_features") as mock_fetch:
            mock_fetch.return_value = _mock_features()
            forecaster.predict(_make_request(fallback_mode="zero"))

        assert zero_fallback_total.value == 1


class TestModelFallbackCounter:
    def test_model_fallback_increments_on_pickle_load(self):
        """When MLflow fails and pickle fallback succeeds, counter increments."""
        from forecast.model_manager import ModelManager

        manager = ModelManager.__new__(ModelManager)
        manager._initialized = False
        manager.__init__()

        with (
            patch.object(manager, "_load_from_mlflow", return_value=False),
            patch.object(manager, "_load_fallback_model", return_value=True),
        ):
            manager.load_production_model()

        assert model_fallback_total.value == 1


class TestNoFallbackOnModelSuccess:
    def test_no_fallback_counter_when_model_loaded(self, forecaster, mock_manager):
        mock_manager.model_loaded = True
        mock_manager.model_version = "v1"
        mock_manager.model_source = "mlflow"
        mock_manager.required_features = ["velocity_24h"]
        mock_manager.cached_feature_importance = None

        features = _mock_features()
        features.has_history = True

        with (
            patch.object(forecaster, "_fetch_features", return_value=features),
            patch.object(forecaster, "_predict_with_model", return_value=0.3),
        ):
            forecaster.predict(_make_request())

        assert heuristic_fallback_total.value == 0
        assert zero_fallback_total.value == 0
