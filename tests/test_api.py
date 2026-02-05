"""Tests for FastAPI signal evaluation API."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import Currency, SignalRequest
from forecast.services import (
    FeatureVector,
    SignalForecaster,
    get_forecaster,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def forecaster():
    """Create signal forecaster."""
    return SignalForecaster()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_healthy(self, client):
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        # model_loaded is False in test environment (no MLflow/MinIO)
        assert isinstance(data["model_loaded"], bool)


class TestSignalEndpoint:
    """Tests for signal evaluation endpoint."""

    def test_predict_returns_200(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_123",
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 200

    def test_predict_response_structure(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_123",
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        data = response.json()

        assert "request_id" in data
        assert "model_score" in data
        assert "model_version" in data
        assert "model_loaded" in data
        assert "latency_ms" in data

    def test_predict_score_in_range(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_456",
                "amount": 500.00,
                "currency": "USD",
                "client_transaction_id": "txn_def",
            },
        )
        data = response.json()

        assert 1 <= data["model_score"] <= 99

    def test_predict_request_id_format(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_789",
                "amount": 250.00,
                "currency": "EUR",
                "client_transaction_id": "txn_ghi",
            },
        )
        data = response.json()

        assert data["request_id"].startswith("req_")
        assert len(data["request_id"]) == 16  # "req_" + 12 hex chars

    def test_predict_model_version_present(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_test",
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_test",
            },
        )
        data = response.json()

        assert data["model_version"] == "v1.0.0"

    def test_predict_idempotent_same_user(self, client):
        """Same user should get consistent prediction."""
        request_data = {
            "user_id": "user_consistent",
            "amount": 200.00,
            "currency": "USD",
            "client_transaction_id": "txn_1",
        }

        response1 = client.post("/predict/signal", json=request_data)
        response2 = client.post("/predict/signal", json=request_data)

        # Score should be the same for same user
        assert response1.json()["model_score"] == response2.json()["model_score"]

        # Request IDs should be different (each request gets new ID)
        assert response1.json()["request_id"] != response2.json()["request_id"]


class TestSignalValidation:
    """Tests for request validation."""

    def test_missing_user_id(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_missing_amount(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_123",
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_negative_amount(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_123",
                "amount": -100.00,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_zero_amount(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_123",
                "amount": 0,
                "currency": "USD",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_invalid_currency(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_123",
                "amount": 100.00,
                "currency": "INVALID",
                "client_transaction_id": "txn_abc",
            },
        )
        assert response.status_code == 422

    def test_missing_transaction_id(self, client):
        response = client.post(
            "/predict/signal",
            json={
                "user_id": "user_123",
                "amount": 100.00,
                "currency": "USD",
            },
        )
        assert response.status_code == 422


class TestSignalForecaster:
    """Tests for SignalForecaster service."""

    def test_predict_returns_response(self, forecaster):
        request = SignalRequest(
            user_id="user_test",
            amount=Decimal("100.00"),
            currency=Currency.USD,
            client_transaction_id="txn_test",
        )

        response = forecaster.predict(request)

        assert response["request_id"].startswith("req_")
        assert 1 <= response["model_score"] <= 99
        assert response["model_version"] == "v1.0.0"
        assert "fallback_used" in response

    def test_predict_fallback_mode_error(self, forecaster, monkeypatch):
        """Test that fallback_mode=error raises RuntimeError when no model/history."""
        monkeypatch.setenv("FORECASTER_FALLBACK_MODE", "error")
        # Ensure we trigger fallback by using unknown user
        request = SignalRequest(
            user_id="unknown_user_for_error",
            amount=Decimal("100.00"),
            currency=Currency.USD,
            client_transaction_id="txn_error",
        )

        with pytest.raises(RuntimeError, match="Forecaster fallback triggered"):
            forecaster.predict(request)

    def test_probability_calculation_base(self, forecaster):
        """Low-risk features should give low probability."""
        features = FeatureVector(
            velocity_24h=1,
            amount_to_avg_ratio_30d=1.0,
            balance_volatility_z_score=0.0,
            bank_connections_24h=1,
            merchant_risk_score=20,
            has_history=True,
        )

        prob = forecaster._calculate_probability(features)
        assert prob < 0.1

    def test_probability_high_velocity(self, forecaster):
        """High velocity should increase probability."""
        low_velocity = FeatureVector(velocity_24h=1, has_history=True)
        high_velocity = FeatureVector(velocity_24h=10, has_history=True)

        prob_low = forecaster._calculate_probability(low_velocity)
        prob_high = forecaster._calculate_probability(high_velocity)

        assert prob_high > prob_low

    def test_probability_high_amount_ratio(self, forecaster):
        """High amount ratio should increase probability."""
        normal = FeatureVector(amount_to_avg_ratio_30d=1.0, has_history=True)
        high = FeatureVector(amount_to_avg_ratio_30d=5.0, has_history=True)

        prob_normal = forecaster._calculate_probability(normal)
        prob_high = forecaster._calculate_probability(high)

        assert prob_high > prob_normal

    def test_probability_no_history(self, forecaster):
        """No history should increase probability."""
        with_history = FeatureVector(has_history=True)
        no_history = FeatureVector(has_history=False)

        prob_with = forecaster._calculate_probability(with_history)
        prob_without = forecaster._calculate_probability(no_history)

        assert prob_without > prob_with

    def test_probability_capped_at_099(self, forecaster):
        """Probability should be capped at 0.99."""
        extreme_features = FeatureVector(
            velocity_24h=100,
            amount_to_avg_ratio_30d=20.0,
            balance_volatility_z_score=-5.0,
            bank_connections_24h=20,
            merchant_risk_score=100,
            has_history=False,
        )

        prob = forecaster._calculate_probability(extreme_features)
        assert prob <= 0.99


class TestGetForecaster:
    """Tests for forecaster singleton."""

    def test_returns_forecaster(self):
        forecaster = get_forecaster()
        assert isinstance(forecaster, SignalForecaster)

    def test_returns_same_instance(self):
        forecaster1 = get_forecaster()
        forecaster2 = get_forecaster()
        assert forecaster1 is forecaster2


class TestGenerateDataEndpoint:
    """Tests for data generation endpoint."""

    def test_generate_data_request_validation(self, client):
        """Test that invalid requests are rejected."""
        # num_users too low
        response = client.post(
            "/data/generate",
            json={"num_users": 5, "fraud_rate": 0.05},
        )
        assert response.status_code == 422

        # num_users too high
        response = client.post(
            "/data/generate",
            json={"num_users": 20000, "fraud_rate": 0.05},
        )
        assert response.status_code == 422

        # fraud_rate too high
        response = client.post(
            "/data/generate",
            json={"num_users": 100, "fraud_rate": 0.8},
        )
        assert response.status_code == 422

        # fraud_rate negative
        response = client.post(
            "/data/generate",
            json={"num_users": 100, "fraud_rate": -0.1},
        )
        assert response.status_code == 422

    @patch("synthetic_pipeline.generator.DataGenerator")
    def test_generate_data_handles_error(self, mock_generator_cls, client):
        """Test error handling in data generation."""
        mock_generator = MagicMock()
        mock_generator.generate_dataset_with_sequences.side_effect = Exception(
            "Database error"
        )
        mock_generator_cls.return_value = mock_generator

        response = client.post(
            "/data/generate",
            json={"num_users": 100, "fraud_rate": 0.05},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Database error" in data["error"]

    def test_generate_data_default_values(self, client):
        """Test that default values are accepted."""
        # This will fail at the database level, but validates the request
        response = client.post("/data/generate", json={})
        # Should not be a validation error (422)
        assert response.status_code == 200


class TestClearDataEndpoint:
    """Tests for data clearing endpoint."""

    def test_clear_data_success(self, client, mock_crud_client):
        """Test successful data clearing via analytics client."""
        response = client.delete("/data/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "generated_records" in data["tables_cleared"]
        assert "evaluation_metadata" in data["tables_cleared"]

    def test_clear_data_handles_error(self, client, monkeypatch):
        """Test error handling in data clearing."""
        from api import crud_client

        def raise_error():
            raise Exception("Connection failed")

        mock = MagicMock()
        mock.clear_all_data.side_effect = Exception("Connection failed")
        monkeypatch.setattr(crud_client, "_client", mock)

        response = client.delete("/data/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Connection failed" in data["error"]


class TestTrainEndpoint:
    """Tests for model training endpoint."""

    def test_train_endpoint_accepts_selected_feature_columns(self, client):
        """Test that /train accepts selected_feature_columns parameter."""
        with patch("model.train.train_model") as mock_train:
            mock_train.return_value = "test_run_123"

            response = client.post(
                "/train",
                json={
                    "max_depth": 6,
                    "training_window_days": 30,
                    "selected_feature_columns": [
                        "velocity_24h",
                        "amount_to_avg_ratio_30d",
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["run_id"] == "test_run_123"

            # Verify train_model was called with feature_columns
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args[1]
            expected_cols = ["velocity_24h", "amount_to_avg_ratio_30d"]
            assert call_kwargs["feature_columns"] == expected_cols

    def test_train_endpoint_works_without_feature_columns(self, client):
        """Test /train works without selected_feature_columns (backward compatible)."""
        with patch("model.train.train_model") as mock_train:
            mock_train.return_value = "test_run_456"

            response = client.post(
                "/train",
                json={
                    "max_depth": 6,
                    "training_window_days": 30,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # Verify train_model was called with feature_columns=None
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs.get("feature_columns") is None

    def test_train_endpoint_rejects_empty_feature_columns(self, client):
        """Test that /train rejects empty selected_feature_columns list."""
        response = client.post(
            "/train",
            json={
                "max_depth": 6,
                "training_window_days": 30,
                "selected_feature_columns": [],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_train_endpoint_handles_invalid_columns_error(self, client):
        """Test that /train returns error when invalid columns are provided."""
        with patch("model.train.train_model") as mock_train:
            mock_train.side_effect = ValueError(
                "Requested feature columns not found in data: ['invalid_col']"
            )

            response = client.post(
                "/train",
                json={
                    "max_depth": 6,
                    "training_window_days": 30,
                    "selected_feature_columns": ["invalid_col"],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "invalid_col" in data["error"]
