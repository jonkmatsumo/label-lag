"""Tests for feature importance in prediction responses (C4)."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from forecast.services import SignalForecaster
from training.schemas import SignalRequest


class TestFeatureImportanceResponse:
    """Tests for feature importance in prediction responses (C4)."""

    def test_include_importance_requested(self):
        """Test that feature importance is included when requested (C4)."""
        forecaster = SignalForecaster()
        request = SignalRequest(
            user_id="user1",
            amount=Decimal("100.0"),
            client_transaction_id="tx1",
            include_importance=True,
        )

        mock_manager = MagicMock()
        mock_manager.model_loaded = True
        mock_manager.cached_feature_importance = {"feat1": 0.5, "feat2": 0.3}

        with (
            patch(
                "forecast.model_manager.get_model_manager",
                return_value=mock_manager,
            ),
            patch.object(forecaster, "_fetch_features") as mock_fetch,
            patch.object(forecaster, "_predict_with_model", return_value=0.5),
        ):
            mock_features = MagicMock()
            mock_features.has_history = True
            mock_fetch.return_value = mock_features

            result = forecaster.predict(request)

            assert "feature_importance" in result
            assert result["feature_importance"] == {"feat1": 0.5, "feat2": 0.3}

    def test_include_importance_not_requested(self):
        """Test that feature importance is omitted when not requested (C4)."""
        forecaster = SignalForecaster()
        request = SignalRequest(
            user_id="user1",
            amount=Decimal("100.0"),
            client_transaction_id="tx1",
            include_importance=False,
        )

        mock_manager = MagicMock()
        mock_manager.model_loaded = True
        mock_manager.cached_feature_importance = {"feat1": 0.5, "feat2": 0.3}

        with (
            patch(
                "forecast.model_manager.get_model_manager",
                return_value=mock_manager,
            ),
            patch.object(forecaster, "_fetch_features") as mock_fetch,
            patch.object(forecaster, "_predict_with_model", return_value=0.5),
        ):
            mock_features = MagicMock()
            mock_features.has_history = True
            mock_fetch.return_value = mock_features

            result = forecaster.predict(request)

            assert "feature_importance" not in result

    def test_importance_unavailable(self):
        """Test diagnostics note when importance is requested but unavailable (C4)."""
        forecaster = SignalForecaster()
        request = SignalRequest(
            user_id="user1",
            amount=Decimal("100.0"),
            client_transaction_id="tx1",
            include_importance=True,
        )

        mock_manager = MagicMock()
        mock_manager.model_loaded = True
        mock_manager.cached_feature_importance = None

        with (
            patch(
                "forecast.model_manager.get_model_manager",
                return_value=mock_manager,
            ),
            patch.object(forecaster, "_fetch_features") as mock_fetch,
            patch.object(forecaster, "_predict_with_model", return_value=0.5),
        ):
            mock_features = MagicMock()
            mock_features.has_history = True
            mock_fetch.return_value = mock_features

            result = forecaster.predict(request)

            assert "feature_importance" not in result
            assert result["diagnostics"]["importance_unavailable"] is True
