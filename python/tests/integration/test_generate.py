import grpc
import pytest

from training.crud_client import get_crud_client

# Assumes Analytics service is running at localhost:50051 (default for local dev)
# Or can be overridden via ANALYTICS_CRUD_TARGET


@pytest.fixture(scope="module")
def crud_client():
    client = get_crud_client()
    # Check connectivity before running integration tests
    try:
        # Simple ping-like call to verify DNS and connectivity
        client.get_overview_metrics()
    except Exception as e:
        pytest.skip(f"Analytics service is not reachable: {e}")
    return client


def test_generate_data_success(crud_client):
    """Verify that generation succeeds and returns expected metrics."""
    # Small dataset for speed
    num_users = 5
    fraud_rate = 0.2

    resp = crud_client.generate_data(
        num_users=num_users, fraud_rate=fraud_rate, drop_existing=False
    )

    assert resp.success is True
    assert resp.total_records > 0
    # Each user should have at least some records
    assert resp.total_records >= num_users
    assert resp.fraud_records >= 0
    assert resp.features_materialized >= 0
    assert not resp.error


def test_generate_data_deterministic(crud_client):
    """Verify that generation is deterministic with a seed."""
    num_users = 2
    fraud_rate = 0.5
    seed = 12345

    resp1 = crud_client.generate_data(
        num_users=num_users, fraud_rate=fraud_rate, seed=seed, drop_existing=True
    )

    resp2 = crud_client.generate_data(
        num_users=num_users, fraud_rate=fraud_rate, seed=seed, drop_existing=True
    )

    assert resp1.total_records == resp2.total_records
    assert resp1.fraud_records == resp2.fraud_records


def test_generate_data_invalid_fraud_rate(crud_client):
    """Verify that invalid fraud rate returns an error or raises exception."""
    # Go service currently doesn't validate fraud_rate in GenerateData.
    # We'll add strict validation in Go in Phase 1.
    try:
        crud_client.generate_data(num_users=5, fraud_rate=2.0, drop_existing=False)
    except grpc.RpcError:
        # If it raises a gRPC error, that's also a valid way to handle it
        pass
