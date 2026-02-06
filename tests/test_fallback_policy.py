"""Tests for request-level fallback policy (C5)."""

import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from api.schemas import SignalRequest
from forecast.services import SignalForecaster


class TestFallbackPolicy:
    """Tests for request-level fallback policy (C5)."""

    @pytest.fixture
    def forecaster(self):
        return SignalForecaster()

    @pytest.fixture
    def mock_manager(self):
        with patch("forecast.model_manager.get_model_manager") as mock:
            manager = MagicMock()
            mock.return_value = manager
            yield manager

    def test_fallback_zero_mode(self, forecaster, mock_manager):
        """Test that fallback_mode='zero' yields score=1 (C5)."""
        mock_manager.model_loaded = False

        request = SignalRequest(
            user_id="user1",
            amount=Decimal("100.0"),
            client_transaction_id="tx1",
            fallback_mode="zero",
        )

        with patch.object(forecaster, "_fetch_features") as mock_fetch:
            mock_features = MagicMock()
            mock_features.has_history = False
            mock_features.velocity_24h = 0
            mock_features.amount_to_avg_ratio_30d = 1.0
            mock_features.balance_volatility_z_score = 0.0
            mock_features.bank_connections_24h = 0
            mock_features.merchant_risk_score = 0
            mock_fetch.return_value = mock_features

            result = forecaster.predict(request)

            assert result["fallback_used"] is True
            assert result["model_score"] == 1
            assert result["diagnostics"]["raw_probability"] == 0.0
            assert result["diagnostics"]["fallback_mode_effective"] == "zero"

    def test_fallback_error_mode(self, forecaster, mock_manager):
        """Test that fallback_mode='error' raises RuntimeError (C5)."""
        mock_manager.model_loaded = False

        request = SignalRequest(
            user_id="user1",
            amount=Decimal("100.0"),
            client_transaction_id="tx1",
            fallback_mode="error",
        )

        with patch.object(forecaster, "_fetch_features") as mock_fetch:
            mock_features = MagicMock()
            mock_features.has_history = False
            mock_features.velocity_24h = 0
            mock_features.amount_to_avg_ratio_30d = 1.0
            mock_features.balance_volatility_z_score = 0.0
            mock_features.bank_connections_24h = 0
            mock_features.merchant_risk_score = 0
            mock_fetch.return_value = mock_features

            with pytest.raises(RuntimeError, match="Forecaster fallback triggered"):
                forecaster.predict(request)

    def test_fallback_probability_mode(self, forecaster, mock_manager):
        """Test that fallback_mode='probability' uses heuristic (C5)."""
        mock_manager.model_loaded = False

        request = SignalRequest(
            user_id="user1",
            amount=Decimal("100.0"),
            client_transaction_id="tx1",
            fallback_mode="probability",
        )

        with patch.object(forecaster, "_fetch_features") as mock_fetch:
            mock_features = MagicMock()
            mock_features.has_history = False
            mock_features.velocity_24h = 0
            mock_features.amount_to_avg_ratio_30d = 1.0
            mock_features.balance_volatility_z_score = 0.0
            mock_features.bank_connections_24h = 0
            mock_features.merchant_risk_score = 0
            mock_fetch.return_value = mock_features

            # Heuristic for no history adds 0.10 to base 0.05 -> 0.15
            # Score for 0.15 with default calibrator (power 3) is:
            # clip((0.15^3)*98 + 1, 1, 99)
            # = clip(0.003375*98 + 1, 1, 99)
            # = clip(0.33 + 1, 1, 99)
            # = 1

            result = forecaster.predict(request)

            assert result["fallback_used"] is True
            assert result["diagnostics"]["fallback_mode_effective"] == "probability"
            assert result["diagnostics"]["raw_probability"] > 0.0

    def test_fallback_env_var_default(self, forecaster, mock_manager):
        """Test that it uses env var if request-level fallback is missing (C5)."""
        mock_manager.model_loaded = False

        request = SignalRequest(
            user_id="user1",
            amount=Decimal("100.0"),
            client_transaction_id="tx1",
            fallback_mode=None,
        )

        with (
            patch.dict(os.environ, {"FORECASTER_FALLBACK_MODE": "zero"}),
            patch.object(forecaster, "_fetch_features") as mock_fetch,
        ):
            mock_features = MagicMock()
            mock_features.has_history = False
            mock_fetch.return_value = mock_features

            result = forecaster.predict(request)

            assert result["diagnostics"]["fallback_mode_effective"] == "zero"
            assert result["model_score"] == 1
