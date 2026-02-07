"""Tests for score distribution monitoring (C3)."""

from unittest.mock import MagicMock, patch

import pytest

from forecast.service import ForecastService
from forecast.v1 import forecast_pb2


class TestScoreDistribution:
    """Tests for score distribution monitoring (C3)."""

    @pytest.mark.anyio
    async def test_score_distribution_calculation(self):
        """Test score distribution bucketization and divergence (C3)."""
        mock_manager = MagicMock()
        # Baseline: mostly low scores
        mock_manager.baseline_distribution = {
            "ratios": [0.8, 0.1, 0.05, 0.03, 0.02],
            "total": 1000,
        }

        mock_client = MagicMock()
        # Live: shifted toward higher scores
        # Buckets: [1-10, 11-30, 31-70, 71-90, 91-99]
        mock_resp = MagicMock()
        mock_resp.scores = [5] * 50 + [20] * 20 + [50] * 10 + [80] * 10 + [95] * 10
        mock_client.get_inference_scores.return_value = mock_resp

        with (
            patch("forecast.service.get_model_manager", return_value=mock_manager),
            patch("training.crud_client.get_crud_client", return_value=mock_client),
        ):
            service = ForecastService()
            request = forecast_pb2.GetScoreDistributionRequest(hours=24)
            mock_context = MagicMock()

            response = service.GetScoreDistribution(request, mock_context)

            assert response.divergence > 0
            assert len(response.distribution) == 5

            # Check bucket [91, 99] (index 4) -> "91-99"
            # Baseline ratio: 0.02
            # Observed count: 10. (Total 100). Ratio 0.10.
            # 0.10 > 2 * 0.02 (0.04), so shift should be detected
            assert response.shift_detected is True
            assert response.distribution["91-99"] == 10

    @pytest.mark.anyio
    async def test_score_distribution_missing_baseline(self):
        """Test graceful handling when baseline is missing (C3)."""
        mock_manager = MagicMock()
        mock_manager.baseline_distribution = None

        mock_client = MagicMock()
        mock_resp = MagicMock()
        # Uniform live scores to match uniform default baseline
        mock_resp.scores = [5, 20, 50, 80, 95] * 2
        mock_client.get_inference_scores.return_value = mock_resp

        with (
            patch("forecast.service.get_model_manager", return_value=mock_manager),
            patch("training.crud_client.get_crud_client", return_value=mock_client),
        ):
            service = ForecastService()
            request = forecast_pb2.GetScoreDistributionRequest(hours=24)
            mock_context = MagicMock()

            response = service.GetScoreDistribution(request, mock_context)

            # Baseline size is 0 in response if missing? Or 0. Implementation detail.
            # Original test asserted None, but protobuf int64 defaults to 0.
            # assert response.baseline_size == 0 # Field does not exist in proto
            assert response.divergence >= 0
            assert response.shift_detected is False  # Default baseline is uniform
