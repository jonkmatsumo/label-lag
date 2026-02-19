from unittest.mock import MagicMock, patch

import pytest

from forecast.metrics import inference_feature_coverage_ratio
from forecast.services import FeatureVector, SignalForecaster


class TestCoverageMetrics:
    @pytest.fixture
    def forecaster(self):
        return SignalForecaster()

    @pytest.fixture
    def mock_manager(self):
        manager = MagicMock()
        manager.model_version = "v1"
        return manager

    def test_full_coverage_ratio(self, forecaster, mock_manager):
        """Assert full coverage results in ratio 1.0."""
        mock_manager.required_features = ["velocity_24h", "merchant_risk_score"]
        mock_manager.predict_single.return_value = 0.5

        features = FeatureVector(
            velocity_24h=10,
            merchant_risk_score=50,
        )

        with patch.object(inference_feature_coverage_ratio, "observe") as mock_observe:
            forecaster._predict_with_model(mock_manager, features)
            mock_observe.assert_called_with(1.0)

    def test_partial_coverage_ratio(self, forecaster, mock_manager):
        """Assert partial coverage results in ratio between 0.0 and 1.0."""
        mock_manager.required_features = ["velocity_24h", "merchant_risk_score"]

        features = FeatureVector()
        features.velocity_24h = 10
        features.merchant_risk_score = None  # Missing

        forecaster._calculate_probability = MagicMock(return_value=0.1)

        with patch.object(inference_feature_coverage_ratio, "observe") as mock_observe:
            forecaster._predict_with_model(mock_manager, features)
            mock_observe.assert_called_with(0.5)

    def test_zero_coverage_ratio(self, forecaster, mock_manager):
        """Assert zero coverage results in ratio 0.0."""
        mock_manager.required_features = ["velocity_24h", "merchant_risk_score"]

        features = FeatureVector()
        features.velocity_24h = None
        features.merchant_risk_score = None

        forecaster._calculate_probability = MagicMock(return_value=0.1)

        with patch.object(inference_feature_coverage_ratio, "observe") as mock_observe:
            forecaster._predict_with_model(mock_manager, features)
            mock_observe.assert_called_with(0.0)

    def test_zero_required_features(self, forecaster, mock_manager):
        """Assert no required features results in ratio 1.0."""
        mock_manager.required_features = []
        mock_manager.predict_single.return_value = 0.5

        features = FeatureVector()

        with patch.object(inference_feature_coverage_ratio, "observe") as mock_observe:
            forecaster._predict_with_model(mock_manager, features)
            mock_observe.assert_called_with(1.0)
