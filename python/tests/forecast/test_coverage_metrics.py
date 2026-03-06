from unittest.mock import MagicMock, patch

import pytest

from forecast.metrics import (
    inference_feature_coverage_below_threshold_total,
    inference_feature_coverage_ratio,
)
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

    def test_partial_coverage_increments_threshold_counter(
        self, forecaster, mock_manager
    ):
        """Coverage below threshold increments bounded counter with bucket label."""
        mock_manager.required_features = ["velocity_24h", "merchant_risk_score"]
        features = FeatureVector()
        features.velocity_24h = 10
        features.merchant_risk_score = None

        forecaster._calculate_probability = MagicMock(return_value=0.1)

        with (
            patch("forecast.services.FEATURE_COVERAGE_WARN_THRESHOLD", 1.0),
            patch.object(
                inference_feature_coverage_below_threshold_total, "labels"
            ) as mock_labels,
        ):
            mock_counter = MagicMock()
            mock_labels.return_value = mock_counter

            forecaster._predict_with_model(mock_manager, features)

        mock_labels.assert_called_once_with(bucket="lt_0.8")
        mock_counter.inc.assert_called_once()

    def test_zero_coverage_increments_lowest_bucket(self, forecaster, mock_manager):
        """Zero coverage should increment the most severe bounded bucket."""
        mock_manager.required_features = ["velocity_24h", "merchant_risk_score"]
        features = FeatureVector()
        features.velocity_24h = None
        features.merchant_risk_score = None

        forecaster._calculate_probability = MagicMock(return_value=0.1)

        with (
            patch("forecast.services.FEATURE_COVERAGE_WARN_THRESHOLD", 1.0),
            patch.object(
                inference_feature_coverage_below_threshold_total, "labels"
            ) as mock_labels,
        ):
            mock_counter = MagicMock()
            mock_labels.return_value = mock_counter

            forecaster._predict_with_model(mock_manager, features)

        mock_labels.assert_called_once_with(bucket="lt_0.5")
        mock_counter.inc.assert_called_once()

    def test_counter_not_incremented_when_coverage_above_threshold(
        self, forecaster, mock_manager
    ):
        """Coverage above threshold should not increment guardrail counter."""
        mock_manager.required_features = ["velocity_24h", "merchant_risk_score"]
        mock_manager.predict_single.return_value = 0.5

        features = FeatureVector(
            velocity_24h=10,
            merchant_risk_score=50,
        )

        with (
            patch("forecast.services.FEATURE_COVERAGE_WARN_THRESHOLD", 0.4),
            patch.object(
                inference_feature_coverage_below_threshold_total, "labels"
            ) as mock_labels,
        ):
            forecaster._predict_with_model(mock_manager, features)

        mock_labels.assert_not_called()

    def test_coverage_ratio_forwarded_to_manager_diagnostics(
        self, forecaster, mock_manager
    ):
        """Coverage ratio should be forwarded for diagnostics health summaries."""
        mock_manager.required_features = ["velocity_24h", "merchant_risk_score"]
        features = FeatureVector()
        features.velocity_24h = 10
        features.merchant_risk_score = None
        forecaster._calculate_probability = MagicMock(return_value=0.1)

        forecaster._predict_with_model(mock_manager, features)

        mock_manager.update_feature_coverage_warning.assert_called_once_with(
            active=True,
            coverage_ratio=0.5,
        )
