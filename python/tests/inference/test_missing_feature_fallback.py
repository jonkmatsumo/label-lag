from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forecast.services import SignalForecaster
from training.schemas import SignalRequest


def test_missing_required_features_sets_explicit_fallback_signal():
    forecaster = SignalForecaster()
    request = SignalRequest(
        user_id="user-1",
        amount=Decimal("100.00"),
        client_transaction_id="txn-1",
    )

    mock_manager = MagicMock()
    mock_manager.model_loaded = True
    mock_manager.model_version = "v-test"
    mock_manager.model_source = "mlflow"
    mock_manager.required_features = ["velocity_24h", "missing_feature"]
    mock_manager.calibrator_loaded = False
    mock_manager.cached_feature_importance = None
    mock_manager.predict_single = MagicMock(return_value=0.99)

    mock_calibrator = MagicMock()
    mock_calibrator.transform.return_value = np.array([42])
    mock_manager.calibrator = mock_calibrator

    features_override = {
        "velocity_24h": 3,
        "amount_to_avg_ratio_30d": 1.2,
        "balance_volatility_z_score": 0.1,
        "bank_connections_24h": 1,
        "merchant_risk_score": 20,
        "has_history": True,
    }

    with (
        patch("forecast.model_manager.get_model_manager", return_value=mock_manager),
        patch.object(forecaster, "_calculate_probability", return_value=0.42),
    ):
        result = forecaster.predict(request, features_override=features_override)

    assert result["fallback_used"] is True
    assert result["diagnostics"]["fallback_reason"] == "missing_features"
    assert result["diagnostics"]["fallback_mode_effective"] == "probability"
    assert result["diagnostics"]["raw_probability"] == pytest.approx(0.42)
    mock_manager.predict_single.assert_not_called()
