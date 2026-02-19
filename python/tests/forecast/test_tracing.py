from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from forecast.services import FeatureVector, SignalForecaster
from training.schemas import ErrorCategory, SignalRequest


class TestTracing:
    @pytest.fixture
    def forecaster(self):
        return SignalForecaster()

    @pytest.fixture
    def mock_manager(self):
        manager = MagicMock()
        manager.model_version = "v1"
        manager.required_features = ["velocity_24h"]
        manager.model_loaded = True
        return manager

    def test_predict_tracing_success(self, forecaster, mock_manager):
        """Verify spans are created and attributes set on success."""
        with (
            patch(
                "forecast.model_manager.get_model_manager", return_value=mock_manager
            ),
            patch("forecast.services.mlflow") as mock_mlflow,
            patch.object(forecaster, "_fetch_features") as mock_fetch,
        ):
            mock_fetch.return_value = FeatureVector(has_history=True, velocity_24h=10)
            mock_manager.predict_single.return_value = 0.5

            mock_span = MagicMock()
            mock_mlflow.start_span.return_value.__enter__.return_value = mock_span

            request = SignalRequest(
                user_id="user1",
                amount=Decimal("100.0"),
                client_transaction_id="tx1",
            )

            forecaster.predict(request)

            # Check calls to start_span
            span_names = [
                call.kwargs.get("name") or call.args[0]
                for call in mock_mlflow.start_span.call_args_list
            ]
            assert "_predict_with_model" in span_names

            # Check attributes on span
            mock_span.set_attribute.assert_any_call("model_version", "v1")
            mock_span.set_attribute.assert_any_call("feature_count.total", 1)
            mock_span.set_attribute.assert_any_call("fallback_used", False)

    def test_predict_tracing_fallback(self, forecaster, mock_manager):
        """Verify fallback attributes are set on failure."""
        with (
            patch(
                "forecast.model_manager.get_model_manager", return_value=mock_manager
            ),
            patch("forecast.services.mlflow") as mock_mlflow,
            patch.object(forecaster, "_fetch_features") as mock_fetch,
        ):
            # Missing features
            features = FeatureVector(has_history=True)
            features.velocity_24h = None
            mock_fetch.return_value = features

            mock_span = MagicMock()
            mock_mlflow.start_span.return_value.__enter__.return_value = mock_span

            # Mock calculate_probability
            forecaster._calculate_probability = MagicMock(return_value=0.1)

            request = SignalRequest(
                user_id="user1",
                amount=Decimal("100.0"),
                client_transaction_id="tx1",
            )

            forecaster.predict(request)

            # Verify attributes
            mock_span.set_attribute.assert_any_call("fallback_used", True)
            mock_span.set_attribute.assert_any_call(
                "fallback_reason", ErrorCategory.MISSING_FEATURES
            )
