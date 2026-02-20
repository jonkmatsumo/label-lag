import numpy as np
import pytest

from training.detect_drift import calculate_psi


class TestDriftDeterminism:
    def test_psi_deterministic_fallback_on_collapse(self):
        # Create data that will collapse quantiles (mostly zeros)
        # 95% zeros means bins based on quantiles will collapse if buckets=10
        expected = np.array([0] * 95 + [1, 2, 3, 4, 5])
        actual = np.array([0] * 90 + [1, 2, 3, 4, 10, 11, 12, 13, 14, 15])

        psi_first, meta_first = calculate_psi(
            expected, actual, buckettype="quantiles", buckets=10
        )
        psi_second, meta_second = calculate_psi(
            expected, actual, buckettype="quantiles", buckets=10
        )

        # 1. Determinism check
        assert psi_first == pytest.approx(psi_second)
        assert meta_first["breakpoints"] == meta_second["breakpoints"]

        # 2. Fallback metadata check
        assert meta_first["buckettype_requested"] == "quantiles"
        assert meta_first["buckettype_used"] == "bins"
        assert meta_first["bucketing_fallback_reason"] == "tied_quantiles"
        assert meta_first["buckets_requested"] == 10
        assert meta_first["buckets_used"] == 10
        assert len(meta_first["breakpoints"]) == 11
        assert meta_first["reference_sample_size"] == len(expected)

        # 3. Mass preservation check (no samples dropped)
        # Fallback linspace(min(0,0), max(5,15), 11) should cover everything
        assert meta_first["breakpoints"][0] <= min(expected.min(), actual.min())
        assert meta_first["breakpoints"][-1] >= max(expected.max(), actual.max())

    def test_psi_metadata_correctness_bins(self):
        expected = np.arange(100)
        actual = np.arange(100) + 10  # Slight shift

        psi, metadata = calculate_psi(expected, actual, buckettype="bins", buckets=5)

        assert metadata["buckettype_requested"] == "bins"
        assert metadata["buckettype_used"] == "bins"
        assert metadata["buckets_requested"] == 5
        assert metadata["buckets_used"] == 5
        assert len(metadata["breakpoints"]) == 6
        assert metadata["breakpoints"][0] == 0  # min(0, 10)
        assert metadata["breakpoints"][-1] == 109  # max(99, 109)
        assert metadata["bucketing_fallback_reason"] is None

    def test_psi_constant_data_returns_zero(self):
        expected = np.array([1, 1, 1])
        actual = np.array([1, 1, 1])

        psi, metadata = calculate_psi(expected, actual)
        assert psi == 0.0
        assert metadata == {}
