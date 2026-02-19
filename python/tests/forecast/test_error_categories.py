from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from forecast.services import FeatureVector, SignalForecaster
from training.schemas import ErrorCategory, SignalRequest


class TestErrorCategories:
    @pytest.fixture
    def forecaster(self):
        return SignalForecaster()

    @pytest.fixture
    def mock_manager(self):
        manager = MagicMock()
        manager.model_version = "v1"
        return manager

    def test_model_not_loaded_category(self, forecaster):
        """Assert MODEL_NOT_LOADED is returned when model is not loaded."""
        with patch("forecast.model_manager.get_model_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.model_loaded = False
            mock_get_manager.return_value = mock_manager

            request = SignalRequest(
                user_id="user1",
                amount=Decimal("100.0"),
                client_transaction_id="tx1",
                fallback_mode="probability",
            )

            with patch.object(forecaster, "_fetch_features") as mock_fetch:
                mock_fetch.return_value = FeatureVector(has_history=True)

                response = forecaster.predict(request)

                assert (
                    response["diagnostics"]["fallback_reason"]
                    == ErrorCategory.MODEL_NOT_LOADED
                )
                assert response["fallback_used"] is True

    def test_missing_features_category(self, forecaster):
        """Assert MISSING_FEATURES is returned when features are missing."""
        with patch("forecast.model_manager.get_model_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.model_loaded = True
            mock_manager.required_features = ["velocity_24h"]
            mock_get_manager.return_value = mock_manager

            request = SignalRequest(
                user_id="user1",
                amount=Decimal("100.0"),
                client_transaction_id="tx1",
            )

            # Feature vector missing velocity_24h (simulated by having it as None)
            # Since FeatureVector has defaults, we need to manually construct one
            # with None if possible or rely on _predict_with_model logic.
            # Here I'll patch _fetch_features to return a vector where I manually
            # set attribute to None
            features = FeatureVector(has_history=True)
            features.velocity_24h = None

            # Also mock _calculate_probability to avoid crash (from Phase 1 issue)
            forecaster._calculate_probability = MagicMock(return_value=0.1)

            with patch.object(forecaster, "_fetch_features", return_value=features):
                response = forecaster.predict(request)

                assert (
                    response["diagnostics"]["fallback_reason"]
                    == ErrorCategory.MISSING_FEATURES
                )
                assert response["fallback_used"] is True
