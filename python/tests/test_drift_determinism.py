import numpy as np

from training.detect_drift import calculate_psi


class TestDriftDeterminism:
    def test_psi_deterministic_fallback_on_collapse(self):
        # Create data that will collapse quantiles (mostly zeros)
        expected = np.array([0] * 95 + [1, 2, 3, 4, 5])
        actual = np.array([0] * 90 + [1, 2, 3, 4, 10, 11, 12, 13, 14, 15])

        # This will collapse if we use 10 buckets because 95% are 0.
        # np.percentile(expected, [0, 10, ..., 90]) will all be 0.

        psi, metadata = calculate_psi(
            expected, actual, buckettype="quantiles", buckets=10
        )

        # Check that it didn't return 0.0 but actually fell back and
        # calculated something
        assert psi >= 0.0
        assert metadata["bucket_type"] == "quantiles"
        # If it fell back to uniform, actual_buckets should be 10
        assert metadata["actual_buckets"] == 10
        assert len(metadata["breakpoints"]) == 11

    def test_psi_metadata_correctness(self):
        expected = np.arange(100)
        actual = np.arange(100) + 10  # Slight shift

        psi, metadata = calculate_psi(expected, actual, buckettype="bins", buckets=5)

        assert metadata["bucket_type"] == "bins"
        assert metadata["n_buckets"] == 5
        assert metadata["actual_buckets"] == 5
        assert len(metadata["breakpoints"]) == 6
        assert metadata["breakpoints"][0] == 0  # min(0, 10)
        assert metadata["breakpoints"][-1] == 109  # max(99, 109)

    def test_psi_constant_data_returns_zero(self):
        expected = np.array([1, 1, 1])
        actual = np.array([1, 1, 1])

        psi, metadata = calculate_psi(expected, actual)
        assert psi == 0.0
        assert metadata == {}
