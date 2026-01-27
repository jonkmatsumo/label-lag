"""Smoke tests for all analytics endpoints.

This module provides baseline guardrails to detect regressions across
all analytics endpoints. Tests validate behavioral continuity (non-empty
responses, required top-level fields) rather than exact values.

Usage:
    # Assumes docker compose is running
    pytest tests/integration/test_analytics_smoke.py -v

    # Override API URL
    API_BASE_URL=http://localhost:8000 pytest tests/integration/test_analytics_smoke.py
"""

import os

import pytest
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8640")
ANALYTICS_RULE_ID = os.getenv("ANALYTICS_RULE_ID")


@pytest.fixture(scope="module")
def api_url():
    """Fixture to provide the API base URL and verify connectivity."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.skip(f"API is not reachable at {API_BASE_URL}: {e}")
    return API_BASE_URL


# =============================================================================
# Endpoint specifications for smoke tests
# Each tuple: (endpoint_path, query_params, required_top_level_fields, list_field)
# list_field is the key containing a list if the response is a list-based response
# =============================================================================

ANALYTICS_ENDPOINTS = [
    # gRPC-proxied endpoints (analytics-crud service)
    (
        "/analytics/daily-stats",
        {"days": 7},
        ["stats"],
        "stats",
    ),
    (
        "/analytics/transactions",
        {"days": 7, "limit": 10},
        ["transactions"],
        "transactions",
    ),
    (
        "/analytics/recent-alerts",
        {"limit": 10},
        ["alerts"],
        "alerts",
    ),
    (
        "/analytics/overview",
        {},
        ["total_records", "fraud_records", "fraud_rate"],
        None,
    ),
    (
        "/analytics/fingerprint",
        {},
        ["generated_records", "feature_snapshots"],
        None,
    ),
    (
        "/analytics/feature-sample",
        {"sample_size": 10, "stratify": "true"},
        ["samples"],
        "samples",
    ),
    (
        "/analytics/schema",
        {},
        ["columns"],
        "columns",
    ),
]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _make_request(url: str, params: dict) -> requests.Response:
    """Make a GET request with retry logic."""
    return requests.get(url, params=params, timeout=30)


def _resolve_rule_id(api_url: str) -> str | None:
    """Resolve a rule id for rule-specific analytics endpoints."""
    if ANALYTICS_RULE_ID:
        return ANALYTICS_RULE_ID

    try:
        response = requests.get(f"{api_url}/rules", timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    data = response.json()
    rules = data.get("rules") or []
    if not rules:
        return None

    return rules[0].get("id")


@pytest.mark.parametrize(
    "endpoint,params,required_fields,list_field",
    ANALYTICS_ENDPOINTS,
    ids=[ep[0] for ep in ANALYTICS_ENDPOINTS],
)
def test_analytics_endpoint_smoke(
    api_url, endpoint, params, required_fields, list_field
):
    """Smoke test for analytics endpoints.

    Validates:
    - HTTP 200 response
    - Required top-level fields present
    - List fields are actually lists (if applicable)
    - Response is non-empty JSON
    """
    url = f"{api_url}{endpoint}"
    response = _make_request(url, params)

    assert response.status_code == 200, (
        f"Expected 200 for {endpoint}, got {response.status_code}: {response.text}"
    )

    data = response.json()
    assert data is not None, f"Response for {endpoint} is None"
    assert isinstance(data, dict), (
        f"Expected dict response for {endpoint}, got {type(data)}"
    )

    # Verify required top-level fields
    for field in required_fields:
        assert field in data, f"Missing required field '{field}' in {endpoint} response"

    # Verify list fields are lists
    if list_field:
        assert isinstance(data[list_field], list), (
            f"Expected '{list_field}' to be a list in {endpoint}, "
            f"got {type(data[list_field])}"
        )


# =============================================================================
# Additional detailed validation tests
# =============================================================================


def test_daily_stats_structure(api_url):
    """Verify daily stats item structure when data exists."""
    url = f"{api_url}/analytics/daily-stats"
    response = _make_request(url, {"days": 30})
    assert response.status_code == 200
    data = response.json()

    if data["stats"]:
        item = data["stats"][0]
        expected_fields = ["date", "total_transactions", "fraud_count", "fraud_rate"]
        for field in expected_fields:
            assert field in item, f"Missing field '{field}' in daily stats item"

        # Type checks
        assert isinstance(item["total_transactions"], int)
        assert isinstance(item["fraud_count"], int)
        assert isinstance(item["fraud_rate"], (int, float))


def test_transactions_structure(api_url):
    """Verify transaction details item structure when data exists."""
    url = f"{api_url}/analytics/transactions"
    response = _make_request(url, {"days": 7, "limit": 5})
    assert response.status_code == 200
    data = response.json()

    if data["transactions"]:
        item = data["transactions"][0]
        expected_fields = [
            "record_id",
            "user_id",
            "amount",
            "is_fraudulent",
        ]
        for field in expected_fields:
            assert field in item, f"Missing field '{field}' in transaction item"


def test_overview_numeric_fields(api_url):
    """Verify overview metrics have valid numeric types."""
    url = f"{api_url}/analytics/overview"
    response = _make_request(url, {})
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["total_records"], int)
    assert isinstance(data["fraud_records"], int)
    assert isinstance(data["fraud_rate"], (int, float))

    # Fraud rate should be a non-negative number
    fraud_rate = data["fraud_rate"]
    assert fraud_rate >= 0, f"Fraud rate is negative: {fraud_rate}"


def test_fingerprint_structure(api_url):
    """Verify fingerprint has expected table structures."""
    url = f"{api_url}/analytics/fingerprint"
    response = _make_request(url, {})
    assert response.status_code == 200
    data = response.json()

    for table_key in ["generated_records", "feature_snapshots"]:
        assert table_key in data
        table = data[table_key]
        assert "count" in table, f"Missing 'count' in {table_key} fingerprint"
        assert isinstance(table["count"], int)


def test_feature_sample_structure(api_url):
    """Verify feature sample item structure when data exists."""
    url = f"{api_url}/analytics/feature-sample"
    response = _make_request(url, {"sample_size": 5, "stratify": "true"})
    assert response.status_code == 200
    data = response.json()

    if data["samples"]:
        sample = data["samples"][0]
        expected_fields = [
            "record_id",
            "is_fraudulent",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
        ]
        for field in expected_fields:
            assert field in sample, f"Missing field '{field}' in feature sample"


def test_schema_has_expected_tables(api_url):
    """Verify schema summary includes expected tables."""
    url = f"{api_url}/analytics/schema"
    response = _make_request(url, {})
    assert response.status_code == 200
    data = response.json()

    tables = {col["table_name"] for col in data["columns"]}
    assert "generated_records" in tables, "Missing generated_records table in schema"
    assert "feature_snapshots" in tables, "Missing feature_snapshots table in schema"


def test_schema_column_structure(api_url):
    """Verify schema column items have required fields."""
    url = f"{api_url}/analytics/schema"
    response = _make_request(url, {})
    assert response.status_code == 200
    data = response.json()

    if data["columns"]:
        col = data["columns"][0]
        expected_fields = ["table_name", "column_name", "data_type"]
        for field in expected_fields:
            assert field in col, f"Missing field '{field}' in column info"


def test_alerts_risk_score(api_url):
    """Verify alerts have computed risk scores."""
    url = f"{api_url}/analytics/recent-alerts"
    response = _make_request(url, {"limit": 5})
    assert response.status_code == 200
    data = response.json()

    if data["alerts"]:
        alert = data["alerts"][0]
        assert "computed_risk_score" in alert, "Missing computed_risk_score in alert"
        # Alerts should have risk score >= 80 (threshold from Go service)
        assert alert["computed_risk_score"] >= 80, (
            f"Alert risk score {alert['computed_risk_score']} below threshold 80"
        )


# =============================================================================
# Sanity checks for endpoint parameter limits
# =============================================================================


def test_daily_stats_respects_days_limit(api_url):
    """Verify daily stats respects days parameter."""
    url = f"{api_url}/analytics/daily-stats"
    response = _make_request(url, {"days": 3})
    assert response.status_code == 200
    data = response.json()

    # Should have at most 3 days of data
    assert len(data["stats"]) <= 3


def test_transactions_respects_limit(api_url):
    """Verify transactions endpoint respects limit parameter."""
    url = f"{api_url}/analytics/transactions"
    response = _make_request(url, {"days": 7, "limit": 5})
    assert response.status_code == 200
    data = response.json()

    assert len(data["transactions"]) <= 5


def test_feature_sample_respects_size(api_url):
    """Verify feature sample respects sample_size parameter."""
    url = f"{api_url}/analytics/feature-sample"
    response = _make_request(url, {"sample_size": 3, "stratify": "false"})
    assert response.status_code == 200
    data = response.json()

    assert len(data["samples"]) <= 3


# =============================================================================
# Rule-specific analytics endpoints
# =============================================================================


def test_rule_analytics_smoke(api_url):
    """Smoke test for rule analytics endpoint."""
    rule_id = _resolve_rule_id(api_url)
    if not rule_id:
        pytest.skip("No rule id available for rule analytics smoke test.")

    url = f"{api_url}/analytics/rules/{rule_id}"
    response = _make_request(url, {"days": 7})
    if response.status_code == 404:
        pytest.skip(f"Rule {rule_id} not found in active ruleset.")
    assert response.status_code == 200

    data = response.json()
    assert data.get("rule_id") == rule_id
    assert "health" in data
    assert "statistics" in data


def test_rule_attribution_smoke(api_url):
    """Smoke test for rule attribution endpoint."""
    rule_id = _resolve_rule_id(api_url)
    if not rule_id:
        pytest.skip("No rule id available for rule attribution smoke test.")

    url = f"{api_url}/analytics/attribution"
    response = _make_request(url, {"rule_id": rule_id, "days": 7})
    if response.status_code == 404:
        pytest.skip(f"No attribution data for rule {rule_id}.")
    assert response.status_code == 200

    data = response.json()
    assert data.get("rule_id") == rule_id
    assert "total_matches" in data
    assert "mean_impact" in data
