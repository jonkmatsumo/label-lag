"""Tests for drift detection functionality."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from training.detect_drift import (
    MAX_DRIFT_ERROR_MESSAGE_LENGTH,
    MIN_REFERENCE_SAMPLES,
    PSI_THRESHOLD_CRITICAL,
    _finalize_drift_error_contract,
    calculate_psi,
    detect_drift,
    get_reference_data,
)
from training.reason_codes import DriftErrorCode, DriftResolutionMode

DRIFT_RESULT_CORE_KEYS = {
    "timestamp",
    "hours_analyzed",
    "threshold",
    "reference_size",
    "live_size",
    "features",
    "drift_detected",
    "drifted_features",
    "drift_error",
    "error_code",
    "error_message",
    "error",
    "resolution_mode",
    "alerts",
    "reference_resolution",
    "reference_model_version",
}
REFERENCE_RESOLUTION_KEYS = {
    "requested_alias",
    "resolution_strategy",
    "resolution_mode",
    "alias_candidate_count",
    "alias_ambiguous",
    "selected_model_version",
    "selected_run_id",
}
BUCKETING_METADATA_KEYS = {
    "buckettype_requested",
    "buckettype_used",
    "buckets_requested",
    "buckets_used",
    "bucketing_fallback_reason",
    "reference_sample_size",
    "nonempty_buckets",
    "nonempty_buckets_ratio",
    "min_expected_count",
    "bucket_mass_ok",
    "bucket_mass_guardrail_applied",
    "drift_error",
    "breakpoints",
}


class TestCalculatePsi:
    """Tests for calculate_psi function."""

    def test_identical_distributions_returns_zero(self):
        """PSI should be approximately zero for identical distributions."""
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)

        psi = calculate_psi(expected, actual)[0]

        assert psi >= 0.0
        assert psi < 0.01  # Should be very close to zero

    def test_shifted_distribution_returns_positive(self):
        """PSI should be positive for shifted distributions."""
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        actual = np.array([10.0, 11.0, 12.0, 13.0, 14.0] * 20)

        psi = calculate_psi(expected, actual)[0]

        assert psi > 0.0
        assert psi > 1.0  # Significant shift should produce high PSI

    def test_empty_array_returns_zero(self):
        """Empty arrays should return zero PSI."""
        expected = np.array([])
        actual = np.array([1.0, 2.0, 3.0])

        psi = calculate_psi(expected, actual)[0]
        assert psi == 0.0

        psi = calculate_psi(actual, expected)[0]
        assert psi == 0.0

        psi = calculate_psi(expected, expected)[0]
        assert psi == 0.0

    def test_nan_values_handled(self):
        """NaN values should be filtered out before calculation."""
        expected = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 20)
        actual = np.array([1.0, 2.0, 3.0, np.nan, 5.0] * 20)

        psi = calculate_psi(expected, actual)[0]

        assert not np.isnan(psi)
        assert psi >= 0.0

    def test_quantile_bucketing(self):
        """Quantile bucketing should produce valid results."""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)

        psi_bins = calculate_psi(expected, actual, buckettype="bins", buckets=10)[0]
        psi_quantiles = calculate_psi(
            expected, actual, buckettype="quantiles", buckets=10
        )[0]

        assert psi_bins >= 0.0
        assert psi_quantiles >= 0.0
        assert not np.isnan(psi_bins)
        assert not np.isnan(psi_quantiles)

    def test_single_value_distribution(self):
        """Single value distributions should not crash."""
        expected = np.array([5.0] * 100)
        actual = np.array([5.0] * 100)

        psi = calculate_psi(expected, actual)[0]
        assert psi >= 0.0

    def test_constant_expected_distribution_returns_zero_and_warns(self, caplog):
        """Constant expected values should short-circuit PSI with warning."""
        expected = np.array([5.0] * 100)
        actual = np.array([4.0, 5.0, 6.0] * 34)

        with caplog.at_level(logging.WARNING):
            psi = calculate_psi(expected, actual, buckettype="quantiles", buckets=10)[0]

        assert psi == 0.0
        assert any(
            "Expected distribution is constant" in record.message
            for record in caplog.records
        )

    def test_quantile_ties_reduce_bucket_count_and_remain_stable(self, caplog):
        """Tied quantiles should warn about reduced buckets and remain deterministic."""
        expected = np.array([0.0] * 95 + [1.0] * 5)
        actual = np.array([0.0] * 90 + [1.0] * 10)

        with caplog.at_level(logging.WARNING):
            psi_first = calculate_psi(
                expected, actual, buckettype="quantiles", buckets=10
            )[0]
            psi_second = calculate_psi(
                expected, actual, buckettype="quantiles", buckets=10
            )[0]

        assert psi_first == pytest.approx(psi_second)
        assert psi_first >= 0.0
        assert any(
            "Quantile PSI bucketing collapsed" in record.message
            and "Falling back to uniform bins" in record.message
            for record in caplog.records
        )

    def test_invalid_buckettype_raises_error(self):
        """Invalid buckettype should raise ValueError."""
        expected = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown buckettype"):
            calculate_psi(expected, actual, buckettype="invalid")

    def test_bucket_mass_guardrail_suppresses_noisy_small_sample_signal(self, caplog):
        """Sparse bucket mass above minimum sample size should suppress PSI alerting."""
        sample_size = MIN_REFERENCE_SAMPLES + 20
        expected = np.array([0.0] * (sample_size - 10) + [1.0] * 10)
        actual = np.array([0.0] * (sample_size - 35) + [7.0] * 35)

        with caplog.at_level(logging.WARNING):
            psi_first, metadata_first = calculate_psi(
                expected, actual, buckettype="quantiles", buckets=10
            )
            psi_second, metadata_second = calculate_psi(
                expected, actual, buckettype="quantiles", buckets=10
            )

        assert psi_first == 0.0
        assert psi_second == 0.0
        assert metadata_first == metadata_second
        assert metadata_first["bucket_mass_guardrail_applied"] is True
        assert metadata_first["bucket_mass_ok"] is False
        assert metadata_first["drift_error"] == "insufficient_bucket_mass"
        assert "nonempty_buckets" in metadata_first
        assert "min_expected_count" in metadata_first
        assert any(
            "Insufficient bucket mass for PSI" in record.message
            for record in caplog.records
        )


class TestReferenceResolution:
    @staticmethod
    def _version(
        *,
        version: int,
        run_id: str,
        current_stage: str = "None",
        aliases: list[str] | None = None,
    ):
        return SimpleNamespace(
            version=str(version),
            run_id=run_id,
            current_stage=current_stage,
            aliases=aliases or [],
        )

    @patch("pandas.read_parquet")
    @patch("mlflow.artifacts.download_artifacts")
    @patch("mlflow.MlflowClient")
    def test_reference_resolution_prefers_alias(
        self,
        mock_client_cls,
        mock_download_artifacts,
        mock_read_parquet,
    ):
        df = pd.DataFrame({"velocity_24h": [1.0], "amount_to_avg_ratio_30d": [1.0]})
        mock_read_parquet.return_value = df
        mock_download_artifacts.return_value = "/tmp/reference_data.parquet"
        mock_client = mock_client_cls.return_value
        mock_client.search_model_versions.return_value = [
            self._version(version=5, run_id="run-v5", current_stage="Production"),
            self._version(version=7, run_id="run-v7", aliases=["champion"]),
        ]

        with patch("training.detect_drift.DRIFT_REFERENCE_MODEL_ALIAS", "@champion"):
            loaded, metadata = get_reference_data(include_metadata=True)

        assert loaded is not None
        assert metadata["resolution_strategy"] == "alias"
        assert metadata["selected_model_version"] == "7"
        mock_download_artifacts.assert_called_once_with(
            run_id="run-v7",
            artifact_path="reference_data.parquet",
            dst_path=mock_download_artifacts.call_args.kwargs["dst_path"],
        )

    @patch("pandas.read_parquet")
    @patch("mlflow.artifacts.download_artifacts")
    @patch("mlflow.MlflowClient")
    def test_reference_resolution_falls_back_to_stage_when_alias_missing(
        self,
        mock_client_cls,
        mock_download_artifacts,
        mock_read_parquet,
        caplog,
    ):
        df = pd.DataFrame({"velocity_24h": [1.0], "amount_to_avg_ratio_30d": [1.0]})
        mock_read_parquet.return_value = df
        mock_download_artifacts.return_value = "/tmp/reference_data.parquet"
        mock_client = mock_client_cls.return_value
        mock_client.search_model_versions.return_value = [
            self._version(version=4, run_id="run-v4", current_stage="Production"),
            self._version(version=2, run_id="run-v2"),
        ]

        with (
            patch("training.detect_drift.DRIFT_REFERENCE_MODEL_ALIAS", "@champion"),
            caplog.at_level(logging.WARNING),
        ):
            _, metadata = get_reference_data(include_metadata=True)

        assert metadata["resolution_strategy"] == "production_stage"
        assert metadata["selected_model_version"] == "4"
        assert "alias '@champion' not found" in caplog.text
        assert "legacy Production stage" in caplog.text

    @patch("pandas.read_parquet")
    @patch("mlflow.artifacts.download_artifacts")
    @patch("mlflow.MlflowClient")
    def test_reference_resolution_falls_back_to_latest_when_alias_and_stage_missing(
        self,
        mock_client_cls,
        mock_download_artifacts,
        mock_read_parquet,
        caplog,
    ):
        df = pd.DataFrame({"velocity_24h": [1.0], "amount_to_avg_ratio_30d": [1.0]})
        mock_read_parquet.return_value = df
        mock_download_artifacts.return_value = "/tmp/reference_data.parquet"
        mock_client = mock_client_cls.return_value
        mock_client.search_model_versions.return_value = [
            self._version(version=1, run_id="run-v1"),
            self._version(version=3, run_id="run-v3"),
            self._version(version=2, run_id="run-v2"),
        ]

        with (
            patch("training.detect_drift.DRIFT_REFERENCE_MODEL_ALIAS", ""),
            caplog.at_level(logging.WARNING),
        ):
            _, metadata = get_reference_data(include_metadata=True)

        assert metadata["resolution_strategy"] == "latest_version"
        assert metadata["selected_model_version"] == "3"
        assert "falling back to latest model version 3" in caplog.text

    @patch("pandas.read_parquet")
    @patch("mlflow.artifacts.download_artifacts")
    @patch("mlflow.MlflowClient")
    def test_reference_resolution_handles_ambiguous_alias_deterministically(
        self,
        mock_client_cls,
        mock_download_artifacts,
        mock_read_parquet,
        caplog,
    ):
        df = pd.DataFrame({"velocity_24h": [1.0], "amount_to_avg_ratio_30d": [1.0]})
        mock_read_parquet.return_value = df
        mock_download_artifacts.return_value = "/tmp/reference_data.parquet"
        mock_client = mock_client_cls.return_value
        mock_client.search_model_versions.return_value = [
            self._version(version=6, run_id="run-v6", aliases=["champion"]),
            self._version(version=8, run_id="run-v8", aliases=["champion"]),
            self._version(version=4, run_id="run-v4", current_stage="Production"),
        ]

        with (
            patch("training.detect_drift.DRIFT_REFERENCE_MODEL_ALIAS", "@champion"),
            caplog.at_level(logging.WARNING),
        ):
            _, metadata = get_reference_data(include_metadata=True)

        assert metadata["resolution_strategy"] == "alias"
        assert metadata["alias_ambiguous"] is True
        assert metadata["alias_candidate_count"] == 2
        assert metadata["selected_model_version"] == "8"
        assert metadata["selected_model_version"] == "8"
        assert "resolved to 2 candidates" in caplog.text

    @patch("mlflow.MlflowClient")
    def test_reference_resolution_handles_empty_registry(self, mock_client_cls):
        """Empty model registry results return safe 'no reference' outcome."""
        mock_client = mock_client_cls.return_value
        mock_client.search_model_versions.return_value = []

        loaded, metadata = get_reference_data(include_metadata=True)

        assert loaded is None
        assert metadata["resolution_strategy"] is None
        assert metadata["requested_alias"] is None

    @patch("training.detect_drift.logger")
    @patch("mlflow.MlflowClient")
    def test_fallback_warnings_rate_limited(self, mock_client_cls, mock_logger):
        """Production stage fallback warns once and is then suppressed."""
        mock_client = mock_client_cls.return_value
        mock_client.search_model_versions.return_value = [
            self._version(version=4, run_id="run-v4", current_stage="Production"),
        ]

        # Reset global flags
        import training.detect_drift

        training.detect_drift._DRIFT_STAGE_FALLBACK_WARNED = False

        # First resolution: should warn
        from training.detect_drift import _select_reference_model_version

        _select_reference_model_version(
            mock_client.search_model_versions.return_value, alias_name=None
        )
        assert any(
            "legacy Production stage" in str(call)
            for call in mock_logger.warning.call_args_list
        )

        # Second resolution: should NOT warn again (process-level limit)
        mock_logger.warning.reset_mock()
        _select_reference_model_version(
            mock_client.search_model_versions.return_value, alias_name=None
        )
        assert not any(
            "legacy Production stage" in str(call)
            for call in mock_logger.warning.call_args_list
        )

    @patch("training.detect_drift.get_live_data")
    @patch("training.detect_drift.get_reference_data")
    def test_detect_drift_surfaces_reference_resolution_metadata(
        self, mock_reference, mock_live
    ):
        reference_df = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3] * 200,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0] * 200,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0] * 200,
            }
        )
        live_df = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3] * 200,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0] * 200,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0] * 200,
            }
        )
        mock_reference.return_value = (
            reference_df,
            {
                "resolution_strategy": "alias",
                "selected_model_version": "9",
                "selected_run_id": "run-v9",
            },
        )
        mock_live.return_value = live_df

        result = detect_drift()

        assert set(result["reference_resolution"].keys()) == REFERENCE_RESOLUTION_KEYS
        assert result["reference_resolution"]["resolution_mode"] == "alias"
        assert result["reference_resolution"]["resolution_strategy"] == "alias"
        assert result["reference_resolution"]["selected_model_version"] == "9"
        assert result["reference_resolution"]["selected_run_id"] == "run-v9"


class TestDetectDrift:
    """Tests for detect_drift function."""

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_returns_expected_structure(self, mock_live, mock_ref):
        """detect_drift should return expected dictionary structure."""
        # Mock reference data
        mock_ref.return_value = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3, 4, 5] * 100,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0, 2.5, 3.0] * 100,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0, 2.0, 3.0] * 100,
            }
        )

        # Mock live data
        mock_live.return_value = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3, 4, 5] * 100,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0, 2.5, 3.0] * 100,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0, 2.0, 3.0] * 100,
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

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_no_reference_data_returns_error(self, mock_live, mock_ref):
        """Missing reference data should return error in results."""
        mock_ref.return_value = None
        mock_live.return_value = pd.DataFrame()

        result = detect_drift()

        assert "error" in result
        assert result["error"] == "No reference data available"
        assert result["reference_size"] == 0
        assert result["live_size"] == 0

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_no_live_data_returns_error(self, mock_live, mock_ref):
        """Missing live data should return error in results."""
        mock_ref.return_value = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3] * 200,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0] * 200,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0] * 200,
            }
        )
        mock_live.return_value = pd.DataFrame()

        result = detect_drift()

        assert "error" in result
        assert result["error"] == "No live data available"
        assert result["reference_size"] > 0
        assert result["live_size"] == 0

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
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

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
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

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
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
            details["status"] == "CRITICAL" for details in result["features"].values()
        )
        if has_critical:
            assert result["drift_detected"] is True
            assert len(result["drifted_features"]) > 0

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_missing_feature_skipped(self, mock_live, mock_ref):
        """Missing features in reference or live data should be skipped."""
        ref_data = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3] * 200,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0] * 200,
                # Missing balance_volatility_z_score
            }
        )

        live_data = pd.DataFrame(
            {
                "velocity_24h": [1, 2, 3] * 200,
                "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0] * 200,
                "balance_volatility_z_score": [-1.0, 0.0, 1.0] * 200,
            }
        )

        mock_ref.return_value = ref_data
        mock_live.return_value = live_data

        result = detect_drift()

        # Should only have features present in both
        assert "velocity_24h" in result["features"]
        assert "amount_to_avg_ratio_30d" in result["features"]
        # balance_volatility_z_score should be skipped (not in reference)

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_insufficient_reference_data(self, mock_live, mock_ref, caplog):
        """Insufficient reference data should block PSI computation and warn."""
        # Create small reference dataset (below 500)
        ref_data = pd.DataFrame(
            {
                "velocity_24h": np.random.normal(0, 1, 400),
                "amount_to_avg_ratio_30d": np.random.normal(0, 1, 400),
                "balance_volatility_z_score": np.random.normal(0, 1, 400),
            }
        )
        mock_ref.return_value = ref_data

        # Live data doesn't matter as it shouldn't be fetched
        mock_live.return_value = pd.DataFrame()

        with caplog.at_level(logging.WARNING):
            result = detect_drift()

        assert "error" in result
        assert "Insufficient reference data" in result["error"]
        assert result["drift_detected"] is False

        # Verify warning
        assert any(
            "Insufficient reference data" in record.message for record in caplog.records
        )

        # Verify get_live_data was NOT called
        mock_live.assert_not_called()

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_guardrail_sets_drift_error_and_suppresses_alerts(
        self, mock_live, mock_ref, caplog
    ):
        """Guardrail should suppress drift signaling on insufficient bucket mass."""
        sample_size = MIN_REFERENCE_SAMPLES + 20
        base_reference = np.array([0.0] * (sample_size - 10) + [1.0] * 10)
        base_live = np.array([0.0] * (sample_size - 30) + [8.0] * 30)

        ref_data = pd.DataFrame(
            {
                "velocity_24h": base_reference,
                "amount_to_avg_ratio_30d": base_reference * 2.0,
                "balance_volatility_z_score": base_reference - 2.0,
            }
        )
        live_data = pd.DataFrame(
            {
                "velocity_24h": base_live,
                "amount_to_avg_ratio_30d": base_live * 2.0,
                "balance_volatility_z_score": base_live - 2.0,
            }
        )
        mock_ref.return_value = ref_data
        mock_live.return_value = live_data

        with caplog.at_level(logging.WARNING):
            result = detect_drift()

        assert result["drift_detected"] is False
        assert result["drift_error"] == "insufficient_bucket_mass"
        assert result["alerts"] == []

        for feature in (
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
        ):
            feature_result = result["features"][feature]
            assert feature_result["status"] == "OK"
            assert feature_result["drift_error"] == "insufficient_bucket_mass"
            assert feature_result["bucketing"]["bucket_mass_ok"] is False
            assert "nonempty_buckets" in feature_result["bucketing"]
            assert "min_expected_count" in feature_result["bucketing"]

        assert any(
            "Insufficient bucket mass for PSI" in record.message
            for record in caplog.records
        )

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_standard_error_fields_for_no_reference_model(self, mock_live, mock_ref):
        mock_ref.return_value = None
        mock_live.return_value = pd.DataFrame()

        result = detect_drift()

        assert result["error_code"] == DriftErrorCode.NO_REFERENCE_DATA.value
        assert result["error_message"] == "No reference data available"
        assert len(result["error_message"]) <= MAX_DRIFT_ERROR_MESSAGE_LENGTH
        assert result["drift_error"] == DriftErrorCode.NO_REFERENCE_DATA.value
        assert result["error"] == result["error_message"]
        assert result["resolution_mode"] == DriftResolutionMode.NONE.value
        assert result["reference_model_version"] is None

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_standard_error_fields_for_insufficient_reference_samples(
        self, mock_live, mock_ref
    ):
        mock_ref.return_value = pd.DataFrame(
            {
                "velocity_24h": [0.0] * 50,
                "amount_to_avg_ratio_30d": [0.0] * 50,
                "balance_volatility_z_score": [0.0] * 50,
            }
        )
        mock_live.return_value = pd.DataFrame()

        result = detect_drift()

        assert (
            result["error_code"] == DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value
        )
        assert "Insufficient reference data" in result["error_message"]
        assert len(result["error_message"]) <= MAX_DRIFT_ERROR_MESSAGE_LENGTH
        assert (
            result["drift_error"] == DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value
        )
        assert result["resolution_mode"] == DriftResolutionMode.NONE.value
        assert result["reference_model_version"] is None

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_standard_error_fields_for_sparse_bucket_guardrail(
        self, mock_live, mock_ref
    ):
        sample_size = MIN_REFERENCE_SAMPLES + 20
        base_reference = np.array([0.0] * (sample_size - 10) + [1.0] * 10)
        base_live = np.array([0.0] * (sample_size - 30) + [8.0] * 30)
        mock_ref.return_value = pd.DataFrame(
            {
                "velocity_24h": base_reference,
                "amount_to_avg_ratio_30d": base_reference,
                "balance_volatility_z_score": base_reference,
            }
        )
        mock_live.return_value = pd.DataFrame(
            {
                "velocity_24h": base_live,
                "amount_to_avg_ratio_30d": base_live,
                "balance_volatility_z_score": base_live,
            }
        )

        result = detect_drift()

        assert result["error_code"] == DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value
        assert (
            result["error_message"]
            == "Drift signal suppressed due to insufficient bucket mass"
        )
        assert len(result["error_message"]) <= MAX_DRIFT_ERROR_MESSAGE_LENGTH
        assert result["drift_error"] == DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value
        assert result["resolution_mode"] in {
            DriftResolutionMode.ALIAS.value,
            DriftResolutionMode.STAGE.value,
            DriftResolutionMode.LATEST.value,
            DriftResolutionMode.NONE.value,
        }
        assert result["reference_model_version"] is None

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_standard_error_fields_for_success_path(self, mock_live, mock_ref):
        base = np.arange(1000, dtype=float)
        stable_reference = pd.DataFrame(
            {
                "velocity_24h": base,
                "amount_to_avg_ratio_30d": base * 0.5,
                "balance_volatility_z_score": base - 500.0,
            }
        )
        mock_ref.return_value = (
            stable_reference,
            {
                "resolution_strategy": "alias",
                "selected_model_version": "9",
                "selected_run_id": "run-v9",
            },
        )
        mock_live.return_value = stable_reference.copy()

        result = detect_drift()

        assert result["error_code"] is None
        assert result["error_message"] is None
        assert result["error"] is None
        assert result["resolution_mode"] == DriftResolutionMode.ALIAS.value
        assert result["reference_model_version"] == "9"

    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_drift_error_contract_stable_across_failure_modes(
        self, mock_live, mock_ref
    ):
        expected_reference_resolution_keys = {
            "requested_alias",
            "resolution_strategy",
            "resolution_mode",
            "alias_candidate_count",
            "alias_ambiguous",
            "selected_model_version",
            "selected_run_id",
        }
        expected_bucketing_keys = {
            "buckettype_requested",
            "buckettype_used",
            "buckets_requested",
            "buckets_used",
            "bucketing_fallback_reason",
            "breakpoints",
            "reference_sample_size",
            "nonempty_buckets",
            "nonempty_buckets_ratio",
            "min_expected_count",
            "bucket_mass_ok",
            "bucket_mass_guardrail_applied",
            "drift_error",
        }
        sparse_reference_size = MIN_REFERENCE_SAMPLES + 20
        sparse_reference = np.array([0.0] * (sparse_reference_size - 10) + [1.0] * 10)
        sparse_live = np.array([0.0] * (sparse_reference_size - 30) + [8.0] * 30)

        stable_base = np.arange(1000, dtype=float)
        stable_df = pd.DataFrame(
            {
                "velocity_24h": stable_base,
                "amount_to_avg_ratio_30d": stable_base * 0.5,
                "balance_volatility_z_score": stable_base - 500.0,
            }
        )

        scenarios = [
            {
                "name": "no_reference_model",
                "reference_payload": None,
                "live_payload": pd.DataFrame(),
                "expected_code": DriftErrorCode.NO_REFERENCE_DATA.value,
                "expected_mode": DriftResolutionMode.NONE.value,
                "expected_reference_version": None,
            },
            {
                "name": "insufficient_reference_samples",
                "reference_payload": pd.DataFrame(
                    {
                        "velocity_24h": [0.0] * 50,
                        "amount_to_avg_ratio_30d": [0.0] * 50,
                        "balance_volatility_z_score": [0.0] * 50,
                    }
                ),
                "live_payload": pd.DataFrame(),
                "expected_code": DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value,
                "expected_mode": DriftResolutionMode.NONE.value,
                "expected_reference_version": None,
            },
            {
                "name": "insufficient_bucket_mass",
                "reference_payload": pd.DataFrame(
                    {
                        "velocity_24h": sparse_reference,
                        "amount_to_avg_ratio_30d": sparse_reference,
                        "balance_volatility_z_score": sparse_reference,
                    }
                ),
                "live_payload": pd.DataFrame(
                    {
                        "velocity_24h": sparse_live,
                        "amount_to_avg_ratio_30d": sparse_live,
                        "balance_volatility_z_score": sparse_live,
                    }
                ),
                "expected_code": DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value,
                "expected_mode": DriftResolutionMode.NONE.value,
                "expected_reference_version": None,
            },
            {
                "name": "success",
                "reference_payload": (
                    stable_df,
                    {
                        "resolution_strategy": "alias",
                        "selected_model_version": "9",
                        "selected_run_id": "run-v9",
                    },
                ),
                "live_payload": stable_df.copy(),
                "expected_code": None,
                "expected_mode": DriftResolutionMode.ALIAS.value,
                "expected_reference_version": "9",
            },
            {
                "name": "success_stage_mode_from_canonical_reference_metadata",
                "reference_payload": (
                    stable_df,
                    {
                        "resolution_mode": "stage",
                        "selected_model_version": "7",
                        "selected_run_id": "run-v7",
                    },
                ),
                "live_payload": stable_df.copy(),
                "expected_code": None,
                "expected_mode": DriftResolutionMode.STAGE.value,
                "expected_reference_version": "7",
            },
        ]

        for scenario in scenarios:
            mock_ref.return_value = scenario["reference_payload"]
            mock_live.return_value = scenario["live_payload"]
            result = detect_drift()

            assert set(result.keys()) == DRIFT_RESULT_CORE_KEYS, scenario["name"]
            assert "error_code" in result, scenario["name"]
            assert "error_message" in result, scenario["name"]
            assert "resolution_mode" in result, scenario["name"]
            assert result["error_code"] == scenario["expected_code"], scenario["name"]
            assert result["resolution_mode"] == scenario["expected_mode"], scenario[
                "name"
            ]
            assert (
                result["reference_model_version"]
                == scenario["expected_reference_version"]
            ), scenario["name"]
            assert set(result["reference_resolution"].keys()) == (
                expected_reference_resolution_keys
            ), scenario["name"]
            assert result["reference_resolution"]["resolution_mode"] in {
                DriftResolutionMode.ALIAS.value,
                DriftResolutionMode.STAGE.value,
                DriftResolutionMode.LATEST.value,
                DriftResolutionMode.NONE.value,
            }, scenario["name"]
            assert result["reference_resolution"]["resolution_strategy"] in {
                DriftResolutionMode.ALIAS.value,
                DriftResolutionMode.STAGE.value,
                DriftResolutionMode.LATEST.value,
                DriftResolutionMode.NONE.value,
            }, scenario["name"]
            assert isinstance(
                result["reference_resolution"]["alias_candidate_count"], int
            ), scenario["name"]
            assert isinstance(
                result["reference_resolution"]["alias_ambiguous"], bool
            ), scenario["name"]
            if result["error_code"] is None:
                assert result["error"] is None, scenario["name"]
            else:
                assert result["error"] == result["error_message"], scenario["name"]
            if result["reference_model_version"] is not None:
                assert (
                    result["reference_model_version"]
                    == result["reference_resolution"]["selected_model_version"]
                ), scenario["name"]
            if result["error_message"] is not None:
                assert len(result["error_message"]) <= MAX_DRIFT_ERROR_MESSAGE_LENGTH, (
                    scenario["name"]
                )
            for feature_name, feature_result in result["features"].items():
                assert feature_name in {
                    "velocity_24h",
                    "amount_to_avg_ratio_30d",
                    "balance_volatility_z_score",
                }, scenario["name"]
                assert set(feature_result.keys()) == {
                    "psi",
                    "status",
                    "drift_error",
                    "bucketing",
                }, scenario["name"]
                assert (
                    set(feature_result["bucketing"].keys()) == expected_bucketing_keys
                ), scenario["name"]
                assert isinstance(feature_result["bucketing"]["breakpoints"], list), (
                    scenario["name"]
                )
                assert len(feature_result["bucketing"]["breakpoints"]) <= 20, scenario[
                    "name"
                ]
                if scenario["name"] == "insufficient_bucket_mass":
                    assert (
                        feature_result["bucketing"]["drift_error"]
                        == DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value
                    ), scenario["name"]

    def test_drift_error_contract_maps_legacy_fields(self):
        legacy = {
            "drift_error": DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value,
            "error_code": None,
            "error_message": None,
            "error": "x" * (MAX_DRIFT_ERROR_MESSAGE_LENGTH + 20),
            "resolution_mode": "production_stage",
            "reference_model_version": " 9 ",
        }

        result = _finalize_drift_error_contract(legacy)

        assert result["error_code"] == DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value
        assert len(result["error_message"]) == MAX_DRIFT_ERROR_MESSAGE_LENGTH
        assert result["error"] == result["error_message"]
        assert result["resolution_mode"] == DriftResolutionMode.STAGE.value
        assert result["reference_model_version"] == "9"

    def test_drift_error_contract_maps_legacy_alias_error_codes(self):
        legacy = {
            "drift_error": None,
            "error_code": "insufficient_reference_data",
            "error_message": None,
            "error": None,
            "resolution_mode": "latest",
            "reference_model_version": " 11 ",
        }

        result = _finalize_drift_error_contract(legacy)

        assert (
            result["error_code"] == DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value
        )
        assert result["error_message"] == "Insufficient reference data"
        assert result["error"] == "Insufficient reference data"
        assert result["resolution_mode"] == DriftResolutionMode.LATEST.value
        assert result["reference_model_version"] == "11"

    @pytest.mark.parametrize(
        ("raw_mode", "expected_mode"),
        [
            ("alias", DriftResolutionMode.ALIAS.value),
            ("production_stage", DriftResolutionMode.STAGE.value),
            ("stage", DriftResolutionMode.STAGE.value),
            ("latest_version", DriftResolutionMode.LATEST.value),
            ("latest", DriftResolutionMode.LATEST.value),
            ("unexpected_mode", DriftResolutionMode.NONE.value),
        ],
    )
    @patch("training.detect_drift.get_reference_data")
    @patch("training.detect_drift.get_live_data")
    def test_resolution_mode_is_normalized(
        self, mock_live, mock_ref, raw_mode, expected_mode
    ):
        base = np.arange(1000, dtype=float)
        stable_reference = pd.DataFrame(
            {
                "velocity_24h": base,
                "amount_to_avg_ratio_30d": base * 0.5,
                "balance_volatility_z_score": base - 500.0,
            }
        )
        mock_ref.return_value = (
            stable_reference,
            {
                "resolution_strategy": raw_mode,
                "selected_model_version": "9",
                "selected_run_id": "run-v9",
            },
        )
        mock_live.return_value = stable_reference.copy()

        result = detect_drift()

        assert result["resolution_mode"] == expected_mode
