"""Tests for parity gap closure features."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

class TestDatasetCorrelations:
    @patch("api.main.get_crud_client")
    def test_correlations_numeric(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock feature sample response structure
        # The endpoint calls MessageToDict on the response
        with patch("api.main.MessageToDict") as mock_to_dict:
            mock_to_dict.return_value = {
                "samples": [
                    {"velocity_24h": 1.0, "amount_to_avg_ratio_30d": 2.0},
                    {"velocity_24h": 2.0, "amount_to_avg_ratio_30d": 4.0}, # Perfect correlation
                    {"velocity_24h": 3.0, "amount_to_avg_ratio_30d": 6.0},
                    {"velocity_24h": 4.0, "amount_to_avg_ratio_30d": 8.0}
                ]
            }
            
            # Pass sample_size >= 100 to satisfy validation
            response = client.get("/analytics/correlations?sample_size=100")
            assert response.status_code == 200
            data = response.json()
            
            # Verify numeric columns detected
            assert "velocity_24h" in data["numeric_columns"]
            
            # Verify pearson correlation
            pearson = data["pearson"]
            corr = next((p for p in pearson if p["feature_a"] == "velocity_24h" and p["feature_b"] == "amount_to_avg_ratio_30d"), None)
            assert corr is not None
            assert corr["value"] > 0.99

    @patch("api.main.get_crud_client")
    def test_correlations_categorical(self, mock_get_client, client):
        with patch("api.main.MessageToDict") as mock_to_dict:
            # Create a larger perfect correlation dataset
            samples = []
            for _ in range(50):
                samples.append({"cat_a": "A", "cat_b": "X"})
            for _ in range(50):
                samples.append({"cat_a": "B", "cat_b": "Y"})
                
            mock_to_dict.return_value = {
                "samples": samples
            }
            
            response = client.get("/analytics/correlations?sample_size=100")
            assert response.status_code == 200
            data = response.json()
            
            assert "cat_a" in data["categorical_columns"]
            cramers = data["cramers_v"]
            # Perfect association
            assoc = next((p for p in cramers if p["feature_a"] == "cat_a" and p["feature_b"] == "cat_b"), None)
            assert assoc is not None
            assert assoc["value"] > 0.9

class TestTransactionSearch:
    @patch("api.main.get_crud_client")
    def test_search_transactions_filters(self, mock_get_client, client):
        from datetime import datetime, timezone

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock proto response
        mock_txn = MagicMock()
        mock_txn.record_id = "txn_1"
        mock_txn.user_id = "user_1"
        mock_txn.amount = 100.0
        mock_txn.is_fraudulent = False
        mock_txn.fraud_type = ""
        mock_txn.is_off_hours_txn = False
        mock_txn.merchant_risk_score = 10
        mock_txn.velocity_24h = 1
        mock_txn.amount_to_avg_ratio_30d = 1.0
        mock_txn.balance_volatility_z_score = 0.5
        mock_txn.created_at = MagicMock()
        mock_txn.created_at.ToDatetime.return_value = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_txn.is_train_eligible = True
        mock_txn.is_pre_fraud = False

        mock_response = MagicMock()
        mock_response.transactions = [mock_txn]
        mock_response.total = 1
        mock_client.search_transactions.return_value = mock_response

        response = client.post(
            "/analytics/transactions/search",
            json={"user_id": "user_1", "limit": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["transactions"]) == 1
        assert data["transactions"][0]["record_id"] == "txn_1"
        assert data["total"] == 1