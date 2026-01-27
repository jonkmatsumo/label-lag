import os
import requests
import pytest
from tenacity import retry, stop_after_attempt, wait_fixed

# Default target is localhost:8000, but can be overridden
# In the provided docker ps, the API is mapped to port 8640
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8640")

@pytest.fixture(scope="module")
def api_url():
    """Fixture to provide the API base URL."""
    # Check if the API is reachable before starting tests
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.skip(f"API is not reachable at {API_BASE_URL}: {e}")
    return API_BASE_URL

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def test_analytics_daily_stats(api_url):
    """Verify that daily stats endpoint returns valid structure."""
    url = f"{api_url}/analytics/daily-stats"
    response = requests.get(url, params={"days": 30})
    assert response.status_code == 200
    data = response.json()
    
    assert "stats" in data
    assert isinstance(data["stats"], list)
    
    # If data exists, verify structure of first item
    if data["stats"]:
        item = data["stats"][0]
        assert "date" in item
        assert "total_transactions" in item
        assert "fraud_count" in item

def test_analytics_schema_summary(api_url):
    """Verify that schema summary returns valid column information."""
    url = f"{api_url}/analytics/schema"
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    
    assert "columns" in data
    assert isinstance(data["columns"], list)
    assert len(data["columns"]) > 0
    
    # Verify expected tables are present
    tables = {col["table_name"] for col in data["columns"]}
    assert "generated_records" in tables
    assert "feature_snapshots" in tables
    
    # Verify known column structure
    cols = {col["column_name"] for col in data["columns"] if col["table_name"] == "generated_records"}
    assert "record_id" in cols
    assert "is_fraudulent" in cols

def test_analytics_feature_sample(api_url):
    """Verify feature sampling endpoint."""
    url = f"{api_url}/analytics/feature-sample"
    params = {"sample_size": 10, "stratify": True}
    response = requests.get(url, params=params)
    assert response.status_code == 200
    data = response.json()
    
    assert "samples" in data
    assert isinstance(data["samples"], list)
    
    # Verify sample size limit is respected
    assert len(data["samples"]) <= 10
    
    if data["samples"]:
        sample = data["samples"][0]
        assert "record_id" in sample
        assert "velocity_24h" in sample
        assert "amount_to_avg_ratio_30d" in sample

def test_analytics_recent_alerts(api_url):
    """Verify recent alerts endpoint."""
    url = f"{api_url}/analytics/recent-alerts"
    response = requests.get(url, params={"limit": 5})
    assert response.status_code == 200
    data = response.json()
    
    assert "alerts" in data
    assert isinstance(data["alerts"], list)
    assert len(data["alerts"]) <= 5

def test_analytics_overview_metrics(api_url):
    """Verify overview metrics endpoint."""
    url = f"{api_url}/analytics/overview"
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    
    assert "total_records" in data
    assert "fraud_records" in data
    assert "fraud_rate" in data
