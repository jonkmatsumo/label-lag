"""Tests for backtest comparison logic."""

from datetime import datetime, timedelta, timezone

import pytest

from api.backtest import BacktestComparator, BacktestMetrics, BacktestResult
from api.schemas import BacktestComparisonResult, BacktestDelta, BacktestResultResponse


@pytest.fixture
def base_result():
    """Create a baseline backtest result."""
    return BacktestResult(
        job_id="job_base",
        rule_id="rule1",
        ruleset_version="v1",
        start_date=datetime.now(timezone.utc) - timedelta(days=7),
        end_date=datetime.now(timezone.utc),
        metrics=BacktestMetrics(
            total_records=1000,
            matched_count=100,
            match_rate=0.10,
            score_distribution={},
            score_mean=50.0,
            score_std=10.0,
            score_min=10,
            score_max=90,
            rejected_count=10,
            rejected_rate=0.01,
        ),
        completed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def candidate_result():
    """Create a candidate backtest result with different metrics."""
    return BacktestResult(
        job_id="job_candidate",
        rule_id="rule1",
        ruleset_version="v2",
        start_date=datetime.now(timezone.utc) - timedelta(days=7),
        end_date=datetime.now(timezone.utc),
        metrics=BacktestMetrics(
            total_records=1000,
            matched_count=150,  # +50 matches
            match_rate=0.15,    # +0.05 rate
            score_distribution={},
            score_mean=55.0,    # +5.0 mean
            score_std=12.0,     # +2.0 std
            score_min=10,
            score_max=95,
            rejected_count=20,  # +10 rejections
            rejected_rate=0.02, # +0.01 rate
        ),
        completed_at=datetime.now(timezone.utc),
    )


def test_backtest_delta_schema():
    """Test that BacktestDelta schema accepts expected values."""
    delta = BacktestDelta(
        match_rate_delta=0.05,
        rejected_rate_delta=0.01,
        score_mean_delta=5.0,
        score_std_delta=2.0,
        matched_count_delta=50,
        rejected_count_delta=10,
    )

    assert delta.match_rate_delta == 0.05
    assert delta.rejected_count_delta == 10


def test_backtest_comparison_result_schema():
    """Test that BacktestComparisonResult schema is valid."""
    # Mock result response objects (simplified)
    # Note: We use dicts here because Pydantic models with extra fields might
    # fail validation if strictly checked.
    # For schema test we just verify the structure.

    base_response = BacktestResultResponse(
        job_id="job1",
        ruleset_version="v1",
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-07T00:00:00Z",
        completed_at="2024-01-07T12:00:00Z",
        metrics={
            "total_records": 100,
            "matched_count": 10,
            "match_rate": 0.1,
            "rejected_count": 1,
            "rejected_rate": 0.01,
            "score_distribution": {},
            "score_mean": 50.0,
            "score_std": 10.0,
            "score_min": 10,
            "score_max": 90
        }
    )

    candidate_response = BacktestResultResponse(
        job_id="job2",
        ruleset_version="v2",
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-07T00:00:00Z",
        completed_at="2024-01-07T12:00:00Z",
        metrics={
            "total_records": 100,
            "matched_count": 20,
            "match_rate": 0.2,
            "rejected_count": 2,
            "rejected_rate": 0.02,
             "score_distribution": {},
            "score_mean": 50.0,
            "score_std": 10.0,
            "score_min": 10,
            "score_max": 90
        }
    )

    delta = BacktestDelta(
        match_rate_delta=0.1,
        rejected_rate_delta=0.01,
        score_mean_delta=0.0,
        score_std_delta=0.0,
        matched_count_delta=10,
        rejected_count_delta=1
    )

    comparison = BacktestComparisonResult(
        base_result=base_response,
        candidate_result=candidate_response,
        delta=delta
    )

    assert comparison.delta.match_rate_delta == 0.1


def test_comparator_compute_delta(base_result, candidate_result):
    """Test that comparator correctly computes deltas."""
    comparator = BacktestComparator()
    delta = comparator.compute_delta(base_result.metrics, candidate_result.metrics)

    assert delta.match_rate_delta == pytest.approx(0.05)
    assert delta.matched_count_delta == 50
    assert delta.rejected_rate_delta == pytest.approx(0.01)
    pass
