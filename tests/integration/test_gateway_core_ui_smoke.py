"""Smoke tests for the preferred BFF -> Gateway read path.

Usage:
    # Assumes docker compose is running
    pytest tests/integration/test_gateway_core_ui_smoke.py -v

    # Override BFF URL
    BFF_BASE_URL=http://localhost:3210 pytest tests/integration/test_gateway_core_ui_smoke.py
"""

import os

import pytest
import requests

BFF_BASE_URL = os.getenv("BFF_BASE_URL", "http://localhost:3210")


@pytest.fixture(scope="module")
def bff_url():
    """Verify BFF is reachable before running smoke tests."""
    try:
        response = requests.get(f"{BFF_BASE_URL}/health", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.skip(f"BFF is not reachable at {BFF_BASE_URL}: {e}")
    return BFF_BASE_URL


@pytest.mark.parametrize(
    "endpoint,params,required_fields",
    [
        ("/bff/v1/analytics/overview", None, ["total_records", "fraud_rate"]),
        ("/bff/v1/monitoring/drift", {"hours": 24}, ["status", "computed_at"]),
        ("/bff/v1/backtest/results", {"limit": 1}, ["results", "total"]),
    ],
    ids=["analytics_overview", "monitoring_drift", "backtest_results"],
)
def test_gateway_core_ui_smoke(bff_url, endpoint, params, required_fields):
    response = requests.get(f"{bff_url}{endpoint}", params=params, timeout=20)
    if response.status_code in (401, 403):
        pytest.skip("BFF requires authentication for smoke requests.")
    response.raise_for_status()
    payload = response.json()
    for field in required_fields:
        assert field in payload
