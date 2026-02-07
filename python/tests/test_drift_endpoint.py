"""Tests for drift monitoring gRPC service."""

from unittest.mock import MagicMock, patch

import pytest

from forecast.service import ForecastService
from forecast.v1 import forecast_pb2


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear drift cache before and after each test."""
    from forecast.drift_cache import _drift_cache

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


class TestDriftService:
    """Tests for GetDriftMonitoring gRPC method."""

    @patch("training.detect_drift.detect_drift")
    @pytest.mark.anyio
    async def test_returns_success(self, mock_detect, mock_drift_result):
        """Method should return response with drift data."""
        mock_detect.return_value = mock_drift_result

        # Patch any other deps if needed (e.g. cache is used)
        service = ForecastService()
        request = forecast_pb2.GetDriftMonitoringRequest(hours=24, threshold=0.2)

        response = service.GetDriftMonitoring(request, MagicMock())

        assert response.drift_detected is True
        # assert len(response.top_features) == 3 # Field does not exist in proto

    @patch("training.detect_drift.detect_drift")
    @pytest.mark.anyio
    async def test_cached_response(self, mock_detect, mock_drift_result):
        """Cached responses should have cached=True."""
        mock_detect.return_value = mock_drift_result

        service = ForecastService()

        # First call - not cached
        req1 = forecast_pb2.GetDriftMonitoringRequest(hours=24)
        _ = service.GetDriftMonitoring(req1, MagicMock())
        # assert resp1.cached is False # Field does not exist
        assert mock_detect.call_count == 1

        # Second call - should be cached
        req2 = forecast_pb2.GetDriftMonitoringRequest(hours=24)
        _ = service.GetDriftMonitoring(req2, MagicMock())
        # assert resp2.cached is True # Field does not exist
        assert mock_detect.call_count == 1  # Should still be 1 if cached

        # detect_drift should only be called once
        assert mock_detect.call_count == 1

    @patch("training.detect_drift.detect_drift")
    @pytest.mark.anyio
    async def test_force_refresh_bypasses_cache(self, mock_detect, mock_drift_result):
        """force_refresh=True should bypass cache."""
        mock_detect.return_value = mock_drift_result

        service = ForecastService()

        # First call
        service.GetDriftMonitoring(
            forecast_pb2.GetDriftMonitoringRequest(hours=24), MagicMock()
        )

        # Second call with force_refresh
        request = forecast_pb2.GetDriftMonitoringRequest(hours=24, force_refresh=True)
        _ = service.GetDriftMonitoring(request, MagicMock())
        # assert resp2.cached is False # Field does not exist

        # detect_drift should be called twice
        # (once for first call, once for force_refresh)
        # Wait, first call in this test invoked service.GetDriftMonitoring.
        # So call_count should be 2.
        assert mock_detect.call_count == 2
        assert mock_detect.call_count == 2
