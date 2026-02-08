from unittest.mock import MagicMock

import grpc
import pytest

from training.crud_client import AnalyticsCRUDClient


class MockRpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


@pytest.fixture
def mock_client():
    client = AnalyticsCRUDClient(target="localhost:50051")
    client.stub = MagicMock()
    return client


def test_generate_data_invalid_argument(mock_client):
    mock_client.stub.GenerateData.side_effect = MockRpcError(
        grpc.StatusCode.INVALID_ARGUMENT, "num_users must be positive"
    )
    with pytest.raises(ValueError, match="Invalid generation parameters"):
        mock_client.generate_data(num_users=-1, fraud_rate=0.1)


def test_generate_data_conflict(mock_client):
    mock_client.stub.GenerateData.side_effect = MockRpcError(
        grpc.StatusCode.ALREADY_EXISTS, "job in progress"
    )
    with pytest.raises(RuntimeError, match="Generation job conflict"):
        mock_client.generate_data(num_users=10, fraud_rate=0.1, idempotency_key="key")


def test_generate_data_timeout(mock_client):
    mock_client.stub.GenerateData.side_effect = MockRpcError(
        grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded"
    )
    with pytest.raises(TimeoutError, match="Generation timed out"):
        mock_client.generate_data(num_users=10, fraud_rate=0.1)


def test_generate_data_unimplemented(mock_client):
    mock_client.stub.GenerateData.side_effect = MockRpcError(
        grpc.StatusCode.UNIMPLEMENTED, "disabled"
    )
    with pytest.raises(NotImplementedError, match="Generation not enabled"):
        mock_client.generate_data(num_users=10, fraud_rate=0.1)
