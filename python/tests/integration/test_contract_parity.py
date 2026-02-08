import pytest

from training.crud_client import get_crud_client


@pytest.fixture(scope="module")
def crud_client():
    return get_crud_client()


def test_generate_data_response_contract(crud_client):
    """Assert that GenerateDataResponse has all expected fields with correct types."""
    resp = crud_client.generate_data(num_users=1, fraud_rate=0.0)

    # Success field
    assert hasattr(resp, "success")
    assert isinstance(resp.success, bool)

    # Metrics fields (int64 in proto -> int in python)
    assert hasattr(resp, "total_records")
    assert isinstance(resp.total_records, int)

    assert hasattr(resp, "fraud_records")
    assert isinstance(resp.fraud_records, int)

    assert hasattr(resp, "features_materialized")
    assert isinstance(resp.features_materialized, int)

    # Error field
    assert hasattr(resp, "error")
    assert isinstance(resp.error, str)


def test_overview_metrics_contract(crud_client):
    """Verify overview metrics fields."""
    resp = crud_client.get_overview_metrics()

    fields = [
        "total_records",
        "fraud_records",
        "fraud_rate",
        "unique_users",
        "total_amount",
        "fraud_amount",
    ]
    for field in fields:
        assert hasattr(resp, field)
        val = getattr(resp, field)
        assert isinstance(val, (int, float))
