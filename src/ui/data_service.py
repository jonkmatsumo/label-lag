"""Data service layer for the dashboard.

Provides read-only database access and API client for the Streamlit UI.
This module is isolated from backend dependencies - uses raw SQL queries
instead of ORM models to avoid coupling.
"""

import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests
import streamlit as st
# DATABASE_URL is no longer used by the UI for analytics.
# Connection is now handled by the CRUD service.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# API timeout in seconds
API_TIMEOUT = 5.0  # Increased for proxying

# Risk score threshold for alerts
ALERT_THRESHOLD = 80


def fetch_daily_stats(days: int = 30) -> pd.DataFrame:
    """Fetch daily transaction statistics from API proxy."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/daily-stats",
            params={"days": days},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json().get("stats", [])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"API error in fetch_daily_stats: {e}")
        return pd.DataFrame()


def fetch_transaction_details(days: int = 7) -> pd.DataFrame:
    """Fetch individual transaction details from API proxy."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/transactions",
            params={"days": days, "limit": 1000},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json().get("transactions", [])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"API error in fetch_transaction_details: {e}")
        return pd.DataFrame()


def fetch_recent_alerts(limit: int = 50) -> pd.DataFrame:
    """Fetch recent high-risk transactions from API proxy."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/recent-alerts",
            params={"limit": limit},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json().get("alerts", [])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"API error in fetch_recent_alerts: {e}")
        return pd.DataFrame()


def fetch_fraud_summary() -> dict[str, Any]:
    """Fetch summary statistics for fraud metrics from API proxy overview."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/overview",
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "total_transactions": data.get("total_records", 0),
            "total_fraud": data.get("fraud_records", 0),
            "fraud_rate": data.get("fraud_rate", 0.0),
            "total_amount": data.get("total_amount", 0.0),
            "fraud_amount": data.get("fraud_amount", 0.0),
        }
    except Exception as e:
        print(f"API error in fetch_fraud_summary: {e}")
        return {
            "total_transactions": 0,
            "total_fraud": 0,
            "fraud_rate": 0.0,
            "total_amount": 0.0,
            "fraud_amount": 0.0,
        }


def fetch_schema_summary() -> pd.DataFrame:
    """Fetch schema summary for primary tables from API.

    Returns:
        DataFrame with columns: table_name, column_name, data_type,
        is_nullable, ordinal_position.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/schema",
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json().get("columns", [])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"API error in fetch_schema_summary: {e}")
        return pd.DataFrame()



    return pd.DataFrame()


@st.cache_data
def _cached_feature_sample(
    fingerprint: dict[str, Any],
    sample_size: int,
    stratify: bool,
) -> pd.DataFrame:
    """Internal cached function for feature sampling via API proxy."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/feature-sample",
            params={"sample_size": sample_size, "stratify": stratify},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json().get("samples", [])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"API error in _cached_feature_sample: {e}")
        return pd.DataFrame()


def fetch_feature_sample(
    sample_size: int,
    stratify: bool = True,
) -> pd.DataFrame:
    """Fetch a sampled feature frame for diagnostics and analysis.

    Returns a bounded-size DataFrame with numeric features and labels,
    optionally stratified by fraud class.

    Args:
        sample_size: Maximum number of rows to return.
        stratify: Whether to stratify sampling by fraud class. Default True.

    Returns:
        DataFrame with columns: record_id, is_fraudulent, velocity_24h,
        amount_to_avg_ratio_30d, balance_volatility_z_score.
        DataFrame will have at most sample_size rows.
    """
    fingerprint = get_dataset_fingerprint()
    return _cached_feature_sample(fingerprint, sample_size, stratify)


# --- API Client ---


def predict_risk(
    user_id: str,
    amount: float,
    currency: str = "USD",
    client_txn_id: str | None = None,
) -> dict[str, Any] | None:
    """Send a transaction for risk evaluation via the API.

    Args:
        user_id: User identifier.
        amount: Transaction amount.
        currency: Currency code (default: USD).
        client_txn_id: Optional client transaction ID. Auto-generated if None.

    Returns:
        API response dictionary with score and risk_components,
        or None if the request fails.

    Example response:
        {
            "request_id": "req_abc123",
            "score": 85,
            "risk_components": [
                {"key": "velocity", "label": "high_transaction_velocity"}
            ],
            "model_version": "v1.0.0"
        }
    """
    if client_txn_id is None:
        client_txn_id = f"ui_txn_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

    url = f"{API_BASE_URL}/evaluate/signal"
    payload = {
        "user_id": user_id,
        "amount": amount,
        "currency": currency,
        "client_transaction_id": client_txn_id,
    }

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        print(f"API timeout after {API_TIMEOUT}s for user {user_id}")
        return None
    except requests.ConnectionError:
        print(f"API connection error: Could not connect to {url}")
        return None
    except requests.HTTPError as e:
        print(f"API HTTP error: {e}")
        return None
    except requests.RequestException as e:
        print(f"API request error: {e}")
        return None


def check_api_health() -> dict[str, Any] | None:
    """Check the API health status.

    Returns:
        Health check response or None if unavailable.
    """
    url = f"{API_BASE_URL}/health"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


# =============================================================================
# Rule Inspector API Clients (Phase 1)
# =============================================================================


def fetch_rules() -> dict[str, Any] | None:
    """Fetch the current production ruleset.

    Returns:
        Dict with version and rules, or None if unavailable.
    """
    url = f"{API_BASE_URL}/rules"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching rules: {e}")
        return None


def sandbox_evaluate(
    features: dict[str, Any],
    base_score: int = 50,
    ruleset: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate rules in sandbox mode.

    Args:
        features: Dict of feature values.
        base_score: Base score before rule application.
        ruleset: Optional custom ruleset dict.

    Returns:
        Evaluation result dict or None if request failed.
    """
    url = f"{API_BASE_URL}/rules/sandbox/evaluate"

    payload = {
        "features": features,
        "base_score": base_score,
    }
    if ruleset is not None:
        payload["ruleset"] = ruleset

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT * 2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error in sandbox evaluation: {e}")
        return None


def fetch_shadow_comparison(
    start_date: str,
    end_date: str,
    rule_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Fetch shadow mode comparison metrics.

    Args:
        start_date: Start date (ISO format).
        end_date: End date (ISO format).
        rule_ids: Optional list of rule IDs to filter.

    Returns:
        Comparison report dict or None if unavailable.
    """
    url = f"{API_BASE_URL}/metrics/shadow/comparison"

    params = {
        "start_date": start_date,
        "end_date": end_date,
    }
    if rule_ids:
        params["rule_ids"] = ",".join(rule_ids)

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT * 2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching shadow comparison: {e}")
        return None


def fetch_backtest_results(
    rule_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
) -> dict[str, Any] | None:
    """Fetch backtest results list.

    Args:
        rule_id: Optional rule ID filter.
        start_date: Optional start date filter (ISO format).
        end_date: Optional end date filter (ISO format).
        limit: Maximum results to return.

    Returns:
        Dict with results list or None if unavailable.
    """
    url = f"{API_BASE_URL}/backtest/results"

    params = {"limit": limit}
    if rule_id:
        params["rule_id"] = rule_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT * 2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching backtest results: {e}")
        return None


def fetch_backtest_result(job_id: str) -> dict[str, Any] | None:
    """Fetch a specific backtest result.

    Args:
        job_id: Backtest job identifier.

    Returns:
        Backtest result dict or None if not found.
    """
    url = f"{API_BASE_URL}/backtest/results/{job_id}"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching backtest result {job_id}: {e}")
        return None


def fetch_draft_rules(
    status: str | None = None,
) -> dict[str, Any] | None:
    """Fetch draft rules.

    Args:
        status: Optional status filter (draft, pending_review, approved, etc.).

    Returns:
        Dict with rules list or None if unavailable.
    """
    url = f"{API_BASE_URL}/rules/draft"

    params = {}
    if status:
        params["status"] = status

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching draft rules: {e}")
        return None


def publish_rule(
    rule_id: str, actor: str, reason: str | None = None
) -> dict[str, Any] | None:
    """Publish an approved rule to production.

    Args:
        rule_id: Rule identifier.
        actor: Who is publishing the rule.
        reason: Optional reason for publishing.

    Returns:
        Publish response dict or None if request failed.
    """
    url = f"{API_BASE_URL}/rules/{rule_id}/publish"

    payload = {"actor": actor}
    if reason:
        payload["reason"] = reason

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error publishing rule: {e}")
        return None


def fetch_approval_signals(rule_id: str) -> dict[str, Any] | None:
    """Fetch approval quality signals for a rule.

    Args:
        rule_id: Rule identifier.

    Returns:
        Dict with signals data or None if unavailable.
    """
    url = f"{API_BASE_URL}/rules/draft/{rule_id}/signals"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching approval signals for {rule_id}: {e}")
        return None


def fetch_heuristic_suggestions(
    field: str | None = None,
    min_confidence: float = 0.7,
    min_samples: int = 100,
) -> dict[str, Any] | None:
    """Fetch heuristic rule suggestions.

    Args:
        field: Optional feature field to filter.
        min_confidence: Minimum confidence threshold.
        min_samples: Minimum samples required for analysis.

    Returns:
        Dict with suggestions list or None if unavailable.
    """
    url = f"{API_BASE_URL}/suggestions/heuristic"

    params = {
        "min_confidence": min_confidence,
        "min_samples": min_samples,
    }
    if field:
        params["field"] = field

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT * 3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching suggestions: {e}")
        return None


def fetch_drift_status(hours: int = 24) -> dict[str, Any] | None:
    """Fetch drift status from the API.

    Args:
        hours: Hours of live data to analyze.

    Returns:
        Drift status dict or None if unavailable.
    """
    url = f"{API_BASE_URL}/monitoring/drift"
    params = {"hours": hours}

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT * 3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching drift status: {e}")
        return None


@st.cache_data(ttl=60)
def _cached_fetch_drift_status(hours: int = 24) -> dict[str, Any] | None:
    """Cached wrapper for fetch_drift_status to avoid redundant API calls.

    Args:
        hours: Hours of live data to analyze.

    Returns:
        Drift status dict or None if unavailable.
    """
    return fetch_drift_status(hours)


# =============================================================================
# Rule Version Diff Functions
# =============================================================================


def fetch_rule_versions(rule_id: str) -> list[dict[str, Any]] | None:
    """Fetch all versions of a rule.

    Args:
        rule_id: Rule identifier.

    Returns:
        List of version dicts or None if request failed.
    """
    url = f"{API_BASE_URL}/rules/{rule_id}/versions"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("versions", [])
    except requests.RequestException as e:
        print(f"Error fetching versions for rule {rule_id}: {e}")
        return None


def fetch_rule_diff(
    rule_id: str,
    version_a: str | None = None,
    version_b: str | None = None,
) -> dict[str, Any] | None:
    """Fetch diff between two rule versions.

    Args:
        rule_id: Rule identifier.
        version_a: Newer version ID (optional, defaults to latest).
        version_b: Older version ID (optional, defaults to predecessor).

    Returns:
        Diff result dict or None if request failed.
    """
    url = f"{API_BASE_URL}/rules/{rule_id}/diff"

    params = {}
    if version_a:
        params["version_a"] = version_a
    if version_b:
        params["version_b"] = version_b

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching diff for rule {rule_id}: {e}")
        return None


def run_backtest(
    ruleset_version: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    rule_id: str | None = None,
) -> dict[str, Any] | None:
    """Run a backtest via API.

    Args:
        ruleset_version: Ruleset version to test.
        start_date: Start date (ISO format).
        end_date: End date (ISO format).
        rule_id: Optional rule identifier (test single rule).

    Returns:
        Backtest result dict or None if request failed.
    """
    url = f"{API_BASE_URL}/backtest/run"

    payload = {
        "ruleset_version": ruleset_version,
        "start_date": start_date,
        "end_date": end_date,
        "rule_id": rule_id,
    }

    try:
        # Long timeout for backtests
        response = requests.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error running backtest: {e}")
        return None


def compare_backtests(
    base_version: str | None,
    candidate_version: str,
    start_date: str,
    end_date: str,
    rule_id: str | None = None,
) -> dict[str, Any] | None:
    """Compare two backtests (what-if simulation).

    Args:
        base_version: Baseline version (None = production).
        candidate_version: Candidate version to compare.
        start_date: Start date (ISO format).
        end_date: End date (ISO format).
        rule_id: Optional rule identifier.

    Returns:
        Comparison result dict with deltas or None if request failed.
    """
    url = f"{API_BASE_URL}/backtest/compare"

    payload = {
        "base_version": base_version,
        "candidate_version": candidate_version,
        "start_date": start_date,
        "end_date": end_date,
        "rule_id": rule_id,
    }

    try:
        # Long timeout for running two backtests
        response = requests.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error comparing backtests: {e}")
        return None


def get_rule_analytics(rule_id: str, days: int = 7) -> dict[str, Any] | None:
    """Fetch analytics for a single rule.

    Args:
        rule_id: Rule ID.
        days: Number of days to look back.

    Returns:
        Dict with rule analytics or None if failed.
    """
    url = f"{API_BASE_URL}/analytics/rules/{rule_id}"
    params = {"days": days}

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching analytics for rule {rule_id}: {e}")
        return None


def get_readiness_report(rule_id: str) -> dict[str, Any] | None:
    """Fetch readiness report for a rule.

    Args:
        rule_id: Rule ID.

    Returns:
        Readiness report dict or None if failed.
    """
    url = f"{API_BASE_URL}/rules/{rule_id}/readiness"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching readiness report for {rule_id}: {e}")
        return None


def get_rule_attribution(rule_id: str, days: int = 7) -> dict[str, Any] | None:
    """Fetch attribution metrics for a rule.

    Args:
        rule_id: Rule ID.
        days: Days lookback.

    Returns:
        Attribution dict or None.
    """
    url = f"{API_BASE_URL}/analytics/attribution"
    params = {"rule_id": rule_id, "days": days}

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching attribution for {rule_id}: {e}")
        return None
