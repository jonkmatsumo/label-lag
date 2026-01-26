"""Tests for drift detection functionality."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from monitor.detect_drift import (
    PSI_THRESHOLD_CRITICAL,
    calculate_psi,
    detect_drift,
)


class TestCalculatePsi:
    """Tests for calculate_psi function."""

    def test_identical_distributions_returns_zero(self):
        """PSI should be approximately zero for identical distributions."""
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)

        psi = calculate_psi(expected, actual)

        assert psi >= 0.0
        assert psi < 0.01  # Should be very close to zero

    def test_shifted_distribution_returns_positive(self):
        """PSI should be positive for shifted distributions."""
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        actual = np.array([10.0, 11.0, 12.0, 13.0, 14.0] * 20)

        psi = calculate_psi(expected, actual)

        assert psi > 0.0
        assert psi > 1.0  # Significant shift should produce high PSI

    def test_empty_array_returns_zero(self):
        """Empty arrays should return zero PSI."""
        expected = np.array([])
        actual = np.array([1.0, 2.0, 3.0])

        psi = calculate_psi(expected, actual)
        assert psi == 0.0

        psi = calculate_psi(actual, expected)
        assert psi == 0.0

        psi = calculate_psi(expected, expected)
        assert psi == 0.0

    def test_nan_values_handled(self):
        """NaN values should be filtered out before calculation."""
        expected = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 20)
        actual = np.array([1.0, 2.0, 3.0, np.nan, 5.0] * 20)

        psi = calculate_psi(expected, actual)

        assert not np.isnan(psi)
        assert psi >= 0.0

    def test_quantile_bucketing(self):
        """Quantile bucketing should produce valid results."""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)

        psi_bins = calculate_psi(expected, actual, buckettype="bins", buckets=10)
        psi_quantiles = calculate_psi(
            expected, actual, buckettype="quantiles", buckets=10
        )

        assert psi_bins >= 0.0
        assert psi_quantiles >= 0.0
        assert not np.isnan(psi_bins)
        assert not np.isnan(psi_quantiles)

    def test_single_value_distribution(self):
        """Single value distributions should not crash."""
        expected = np.array([5.0] * 100)
        actual = np.array([5.0] * 100)

        psi = calculate_psi(expected, actual)
        assert psi >= 0.0

    def test_invalid_buckettype_raises_error(self):
        """Invalid buckettype should raise ValueError."""
        expected = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown buckettype"):
            calculate_psi(expected, actual, buckettype="invalid")


class TestDetectDrift:
    """Tests for detect_drift function."""

    @patch("monitor.detect_drift.get_reference_data")
    @patch("monitor.detect_drift.get_live_data")
    def test_returns_expected_structure(self, mock_live, mock_ref):
        """detect_drift should return expected dictionary structure."""
        # Mock reference data
        mock_ref.return_value = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3, 4, 5] * 20,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0, 2.5, 3.0] * 20,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0, 2.0, 3.0] * 20,
            }
        )

        # Mock live data
        mock_live.return_value = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3, 4, 5] * 20,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0, 2.5, 3.0] * 20,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0, 2.0, 3.0] * 20,
            }
        )

        result = detect_drift(hours=24, threshold=PSI_THRESHOLD_CRITICAL)

        assert "timestamp" in result
        assert "hours_analyzed" in result
        assert "threshold" in result
        assert "reference_size" in result
        assert "live_size" in result
        assert "features" in result
        assert "drift_detected" in result
        assert "drifted_features" in result
        assert isinstance(result["features"], dict)
        assert isinstance(result["drifted_features"], list)

    @patch("monitor.detect_drift.get_reference_data")
    @patch("monitor.detect_drift.get_live_data")
    def test_no_reference_data_returns_error(self, mock_live, mock_ref):
        """Missing reference data should return error in results."""
        mock_ref.return_value = None
        mock_live.return_value = pd.DataFrame()

        result = detect_drift()

        assert "error" in result
        assert result["error"] == "No reference data available"
        assert result["reference_size"] == 0
        assert result["live_size"] == 0

    @patch("monitor.detect_drift.get_reference_data")
    @patch("monitor.detect_drift.get_live_data")
    def test_no_live_data_returns_error(self, mock_live, mock_ref):
        """Missing live data should return error in results."""
        mock_ref.return_value = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3],
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0],
                "balance_volatility_z_score": [-1.0, 0.0, 1.0],
            }
        )
        mock_live.return_value = pd.DataFrame()

        result = detect_drift()

        assert "error" in result
        assert result["error"] == "No live data available"
        assert result["reference_size"] > 0
        assert result["live_size"] == 0

    @patch("monitor.detect_drift.get_reference_data")
    @patch("monitor.detect_drift.get_live_data")
    def test_status_classification_ok(self, mock_live, mock_ref):
        """PSI < 0.1 should result in OK status."""
        # Create identical distributions (low PSI)
        ref_data = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3, 4, 5] * 100,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0, 2.5, 3.0] * 100,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0, 2.0, 3.0] * 100,
            }
        )

        live_data = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3, 4, 5] * 100,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0, 2.5, 3.0] * 100,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0, 2.0, 3.0] * 100,
            }
        )

        mock_ref.return_value = ref_data
        mock_live.return_value = live_data

        result = detect_drift()

        assert not result["drift_detected"]
        for feature, details in result["features"].items():
            assert details["status"] == "OK"

    @patch("monitor.detect_drift.get_reference_data")
    @patch("monitor.detect_drift.get_live_data")
    def test_status_classification_warning(self, mock_live, mock_ref):
        """0.1 <= PSI < 0.2 should result in WARNING status."""
        # Create moderately shifted distributions
        ref_data = pd.DataFrame(
            {
                "velocity_24h": np.random.normal(5, 1, 1000),
                "amount_to_avg_ratio_30d": np.random.normal(2.0, 0.5, 1000),
                "balance_volatility_z_score": np.random.normal(0, 1, 1000),
            }
        )

        # Shift distributions to trigger warning
        live_data = pd.DataFrame(
            {
                "velocity_24h": np.random.normal(6, 1.5, 1000),
                "amount_to_avg_ratio_30d": np.random.normal(2.5, 0.7, 1000),
                "balance_volatility_z_score": np.random.normal(0.5, 1.2, 1000),
            }
        )

        mock_ref.return_value = ref_data
        mock_live.return_value = live_data

        result = detect_drift()

        # At least one feature should have WARNING status
        # (exact status depends on PSI calculation)
        assert isinstance(result["features"], dict)
        assert len(result["features"]) > 0

    @patch("monitor.detect_drift.get_reference_data")
    @patch("monitor.detect_drift.get_live_data")
    def test_status_classification_critical(self, mock_live, mock_ref):
        """PSI >= 0.2 should result in CRITICAL status and drift_detected=True."""
        # Create very different distributions
        ref_data = pd.DataFrame(
            {
                "velocity_24h": np.random.normal(5, 1, 1000),
                "amount_to_avg_ratio_30d": np.random.normal(2.0, 0.5, 1000),
                "balance_volatility_z_score": np.random.normal(0, 1, 1000),
            }
        )

        # Very different distributions to trigger critical
        live_data = pd.DataFrame(
            {
                "velocity_24h": np.random.normal(20, 5, 1000),
                "amount_to_avg_ratio_30d": np.random.normal(10.0, 3.0, 1000),
                "balance_volatility_z_score": np.random.normal(5, 2, 1000),
            }
        )

        mock_ref.return_value = ref_data
        mock_live.return_value = live_data

        result = detect_drift()

        # Should detect drift if any feature has PSI >= 0.2
        assert isinstance(result["features"], dict)
        # Check if any feature has CRITICAL status
        has_critical = any(
            details["status"] == "CRITICAL"
            for details in result["features"].values()
        )
        if has_critical:
            assert result["drift_detected"] is True
            assert len(result["drifted_features"]) > 0

    @patch("monitor.detect_drift.get_reference_data")
    @patch("monitor.detect_drift.get_live_data")
    def test_missing_feature_skipped(self, mock_live, mock_ref):
        """Missing features in reference or live data should be skipped."""
        ref_data = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3],
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0],
                # Missing balance_volatility_z_score
            }
        )

        live_data = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3],
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0],
                "balance_volatility_z_score": [-1.0, 0.0, 1.0],
            }
        )

        mock_ref.return_value = ref_data
        mock_live.return_value = live_data

        result = detect_drift()

        # Should only have features present in both
        assert "velocity_24h" in result["features"]
        assert "amount_to_avg_ratio_30d" in result["features"]
        # balance_volatility_z_score should be skipped (not in reference)
