"""Tests for drift monitoring API endpoint."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.drift_cache import DriftCache, get_drift_cache
from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear drift cache before and after each test."""
    from api.drift_cache import _drift_cache

    # Clear the module-level cache
    if _drift_cache is not None:
        _drift_cache.invalidate()
    yield
    # Clear again after test
    if _drift_cache is not None:
        _drift_cache.invalidate()


@pytest.fixture
def mock_drift_result():
    """Sample drift detection result."""
    return {
        "timestamp": "2024-01-01T12:00:00+00:00",
        "hours_analyzed": 24,
        "threshold": 0.2,
        "reference_size": 1000,
        "live_size": 500,
        "features": {
            "velocity_24h": {"psi": 0.05, "status": "OK"},
            "amount_to_avg_ratio_30d": {"psi": 0.15, "status": "WARNING"},
            "balance_volatility_z_score": {"psi": 0.25, "status": "CRITICAL"},
        },
        "drift_detected": True,
        "drifted_features": ["balance_volatility_z_score"],
    }


@pytest.fixture
def mock_drift_result_no_drift():
    """Sample drift detection result with no drift."""
    return {
        "timestamp": "2024-01-01T12:00:00+00:00",
        "hours_analyzed": 24,
        "threshold": 0.2,
        "reference_size": 1000,
        "live_size": 500,
        "features": {
            "velocity_24h": {"psi": 0.05, "status": "OK"},
            "amount_to_avg_ratio_30d": {"psi": 0.08, "status": "OK"},
            "balance_volatility_z_score": {"psi": 0.09, "status": "OK"},
        },
        "drift_detected": False,
        "drifted_features": [],
    }


class TestDriftEndpoint:
    """Tests for /monitoring/drift endpoint."""

    @patch("monitor.detect_drift.detect_drift")
    def test_returns_200(self, mock_detect, client, mock_drift_result):
        """Endpoint should return 200 on success."""
        mock_detect.return_value = mock_drift_result

        response = client.get("/monitoring/drift")

        assert response.status_code == 200

    @patch("monitor.detect_drift.detect_drift")
    def test_response_structure(self, mock_detect, client, mock_drift_result):
        """Response should have all required fields."""
        mock_detect.return_value = mock_drift_result

        response = client.get("/monitoring/drift")
        data = response.json()

        assert "status" in data
        assert "computed_at" in data
        assert "cached" in data
        assert "reference_window" in data
        assert "current_window" in data
        assert "reference_size" in data
        assert "live_size" in data
        assert "top_features" in data
        assert "thresholds" in data
        assert isinstance(data["top_features"], list)
        assert isinstance(data["thresholds"], dict)

    @patch("monitor.detect_drift.detect_drift")
    def test_cached_response_returns_cached_true(
        self, mock_detect, client, mock_drift_result, mock_drift_result_no_drift
    ):
        """Cached responses should have cached=True."""
        # First call - not cached
        mock_detect.return_value = mock_drift_result
        response1 = client.get("/monitoring/drift")
        assert response1.json()["cached"] is False

        # Second call - should be cached
        response2 = client.get("/monitoring/drift")
        assert response2.json()["cached"] is True

        # detect_drift should only be called once
        assert mock_detect.call_count == 1

    @patch("monitor.detect_drift.detect_drift")
    def test_force_refresh_bypasses_cache(self, mock_detect, client, mock_drift_result):
        """force_refresh=True should bypass cache."""
        mock_detect.return_value = mock_drift_result

        # First call
        response1 = client.get("/monitoring/drift")
        assert response1.json()["cached"] is False

        # Second call with force_refresh
        response2 = client.get("/monitoring/drift?force_refresh=true")
        assert response2.json()["cached"] is False

        # detect_drift should be called twice
        assert mock_detect.call_count == 2

    def test_invalid_hours_returns_422(self, client):
        """Invalid hours parameter should return 422."""
        # Too low
        response = client.get("/monitoring/drift?hours=0")
        assert response.status_code == 422

        # Too high
        response = client.get("/monitoring/drift?hours=200")
        assert response.status_code == 422

    @patch("monitor.detect_drift.detect_drift")
    def test_error_handling_returns_unknown_status(self, mock_detect, client):
        """Errors should return status='unknown' with error message."""
        error_result = {
            "timestamp": "2024-01-01T12:00:00+00:00",
            "hours_analyzed": 24,
            "threshold": 0.2,
            "reference_size": 0,
            "live_size": 0,
            "features": {},
            "drift_detected": False,
            "drifted_features": [],
            "error": "No reference data available",
        }
        mock_detect.return_value = error_result

        response = client.get("/monitoring/drift")
        data = response.json()

        assert data["status"] == "unknown"
        assert data["error"] == "No reference data available"

    @patch("monitor.detect_drift.detect_drift")
    def test_status_classification_ok(
        self, mock_detect, client, mock_drift_result_no_drift
    ):
        """No drift should result in status='ok'."""
        mock_detect.return_value = mock_drift_result_no_drift

        response = client.get("/monitoring/drift")
        data = response.json()

        assert data["status"] == "ok"
        assert data["error"] is None

    @patch("monitor.detect_drift.detect_drift")
    def test_status_classification_warn(self, mock_detect, client):
        """WARNING features should result in status='warn'."""
        warn_result = {
            "timestamp": "2024-01-01T12:00:00+00:00",
            "hours_analyzed": 24,
            "threshold": 0.2,
            "reference_size": 1000,
            "live_size": 500,
            "features": {
                "velocity_24h": {"psi": 0.12, "status": "WARNING"},
            },
            "drift_detected": False,
            "drifted_features": [],
        }
        mock_detect.return_value = warn_result

        response = client.get("/monitoring/drift")
        data = response.json()

        assert data["status"] == "warn"

    @patch("monitor.detect_drift.detect_drift")
    def test_status_classification_fail(self, mock_detect, client, mock_drift_result):
        """CRITICAL features should result in status='fail'."""
        mock_detect.return_value = mock_drift_result

        response = client.get("/monitoring/drift")
        data = response.json()

        assert data["status"] == "fail"

    @patch("monitor.detect_drift.detect_drift")
    def test_top_features_sorted_by_psi(self, mock_detect, client, mock_drift_result):
        """Top features should be sorted by PSI descending."""
        mock_detect.return_value = mock_drift_result

        response = client.get("/monitoring/drift")
        data = response.json()

        top_features = data["top_features"]
        assert len(top_features) == 3

        # Should be sorted by PSI descending
        psi_values = [f["psi"] for f in top_features]
        assert psi_values == sorted(psi_values, reverse=True)

        # Highest PSI should be first
        assert top_features[0]["psi"] == 0.25
        assert top_features[0]["feature"] == "balance_volatility_z_score"

    @patch("monitor.detect_drift.detect_drift")
    def test_thresholds_in_response(self, mock_detect, client, mock_drift_result):
        """Response should include threshold values."""
        mock_detect.return_value = mock_drift_result

        response = client.get("/monitoring/drift")
        data = response.json()

        assert "thresholds" in data
        assert "warn" in data["thresholds"]
        assert "fail" in data["thresholds"]
        assert data["thresholds"]["warn"] == 0.1
        assert data["thresholds"]["fail"] == 0.2

    @patch("monitor.detect_drift.detect_drift")
    def test_exception_handling(self, mock_detect, client):
        """Exceptions should return unknown status with error."""
        mock_detect.side_effect = Exception("Database connection failed")

        response = client.get("/monitoring/drift")
        data = response.json()

        assert data["status"] == "unknown"
        assert data["error"] is not None
        assert "error" in data["error"].lower() or "failed" in data["error"].lower()


class TestDriftCache:
    """Tests for DriftCache behavior."""

    def test_cache_hit_within_ttl(self):
        """Cache should return stored value within TTL."""
        cache = DriftCache(ttl_seconds=300)
        result = {"test": "data"}

        cache.set(24, 0.2, result)
        cached = cache.get(24, 0.2)

        assert cached == result

    def test_cache_miss_after_ttl(self):
        """Expired cache entries should not be returned."""
        import time

        cache = DriftCache(ttl_seconds=1)  # 1 second TTL
        result = {"test": "data"}

        cache.set(24, 0.2, result)
        # Should be cached immediately
        cached = cache.get(24, 0.2)
        assert cached == result

        # Wait for TTL to expire
        time.sleep(1.1)
        cached = cache.get(24, 0.2)
        assert cached is None

    def test_invalidate_clears_cache(self):
        """invalidate() should clear the cache."""
        cache = DriftCache()
        result = {"test": "data"}

        cache.set(24, 0.2, result)
        assert cache.get(24, 0.2) == result

        cache.invalidate()
        assert cache.get(24, 0.2) is None

    def test_different_params_separate_cache(self):
        """Cache stores only one entry - new entry overwrites old."""
        cache = DriftCache()
        result1 = {"hours": 24}
        result2 = {"hours": 48}

        cache.set(24, 0.2, result1)
        assert cache.get(24, 0.2) == result1

        # Setting a new entry with different params overwrites
        cache.set(48, 0.2, result2)
        assert cache.get(48, 0.2) == result2
        # Old entry is gone
        assert cache.get(24, 0.2) is None

        # Different threshold should miss
        assert cache.get(48, 0.3) is None

    def test_get_drift_cache_singleton(self):
        """get_drift_cache() should return singleton instance."""
        cache1 = get_drift_cache()
        cache2 = get_drift_cache()

        assert cache1 is cache2

        # Setting in one should affect the other
        cache1.set(24, 0.2, {"test": "data"})
        assert cache2.get(24, 0.2) == {"test": "data"}
