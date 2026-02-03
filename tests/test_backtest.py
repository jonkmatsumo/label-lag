"""Tests for backtesting infrastructure.

Tests mock the analytics client to verify backtest computation logic
without requiring a live analytics service.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from api.backtest import BacktestMetrics, BacktestResult, BacktestRunner, BacktestStore
from api.rules import Rule, RuleSet


class TestBacktestMetrics:
    """Tests for BacktestMetrics dataclass."""

    def test_backtest_metrics_creation(self):
        """Test creating backtest metrics."""
        metrics = BacktestMetrics(
            total_records=100,
            matched_count=25,
            match_rate=0.25,
            score_distribution={
                "1-20": 10,
                "21-40": 20,
                "41-60": 30,
                "61-80": 25,
                "81-99": 15,
            },
            score_mean=50.0,
            score_std=20.0,
            score_min=10,
            score_max=95,
            rejected_count=5,
            rejected_rate=0.05,
        )

        assert metrics.total_records == 100
        assert metrics.match_rate == 0.25
        assert metrics.rejected_rate == 0.05


def _make_mock_feature(
    velocity_24h: int,
    amount_to_avg_ratio_30d: float,
    balance_volatility_z_score: float,
    experimental_signals_json: str = "",
) -> MagicMock:
    """Create a mock proto feature vector."""
    feature = MagicMock()
    feature.velocity_24h = velocity_24h
    feature.amount_to_avg_ratio_30d = amount_to_avg_ratio_30d
    feature.balance_volatility_z_score = balance_volatility_z_score
    feature.experimental_signals_json = experimental_signals_json
    return feature


class TestBacktestRunner:
    """Tests for BacktestRunner class."""

    @pytest.fixture
    def sample_ruleset(self):
        """Create a sample ruleset for testing."""
        rules = [
            Rule(
                id="velocity_high",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
        ]
        return RuleSet(version="v1", rules=rules)

    def test_run_backtest_with_matching_rules(self, sample_ruleset, monkeypatch):
        """Test running a backtest where rules match."""
        from api import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.features = [
            _make_mock_feature(10, 3.5, -1.5),  # velocity > 5, rule matches
            _make_mock_feature(2, 1.0, 0.0),   # velocity <= 5, no match
        ]
        mock_client.stub.GetBacktestFeatures.return_value = mock_response
        mock_client.stub.SaveBacktestResult.return_value = MagicMock(success=True)
        monkeypatch.setattr(crud_client, "_client", mock_client)

        runner = BacktestRunner()
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(
            sample_ruleset, start_date, end_date, base_score=50
        )

        assert result.job_id is not None
        assert result.ruleset_version == "v1"
        assert result.metrics.total_records == 2
        assert result.metrics.matched_count == 1  # First record matches
        assert result.metrics.match_rate == 0.5
        assert result.error is None

    def test_run_backtest_with_no_matches(self, monkeypatch):
        """Test running a backtest where no rules match."""
        from api import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.features = [
            _make_mock_feature(2, 1.0, 0.0),
            _make_mock_feature(3, 1.5, 0.5),
        ]
        mock_client.stub.GetBacktestFeatures.return_value = mock_response
        mock_client.stub.SaveBacktestResult.return_value = MagicMock(success=True)
        monkeypatch.setattr(crud_client, "_client", mock_client)

        rules = [
            Rule(
                id="high_threshold",
                field="velocity_24h",
                op=">",
                value=100,  # Very high threshold
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        runner = BacktestRunner()
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(ruleset, start_date, end_date, base_score=50)

        assert result.metrics.matched_count == 0
        assert result.metrics.match_rate == 0.0

    def test_run_backtest_computes_score_distribution(self, sample_ruleset, monkeypatch):
        """Test that backtest computes score distribution."""
        from api import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.features = [
            _make_mock_feature(10, 3.5, -1.5),  # rule matches -> high score
            _make_mock_feature(2, 1.0, 0.0),   # no match -> base score
        ]
        mock_client.stub.GetBacktestFeatures.return_value = mock_response
        mock_client.stub.SaveBacktestResult.return_value = MagicMock(success=True)
        monkeypatch.setattr(crud_client, "_client", mock_client)

        runner = BacktestRunner()
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(
            sample_ruleset, start_date, end_date, base_score=50
        )

        # Score distribution should have bins populated
        assert "1-20" in result.metrics.score_distribution
        assert "21-40" in result.metrics.score_distribution
        assert "41-60" in result.metrics.score_distribution
        assert "61-80" in result.metrics.score_distribution
        assert "81-99" in result.metrics.score_distribution

    def test_run_backtest_handles_empty_data(self, sample_ruleset, monkeypatch):
        """Test that backtest handles empty historical data."""
        from api import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.features = []
        mock_client.stub.GetBacktestFeatures.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        runner = BacktestRunner()
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(
            sample_ruleset, start_date, end_date, base_score=50
        )

        assert result.metrics.total_records == 0
        assert result.metrics.match_rate == 0.0

    def test_run_backtest_handles_errors_gracefully(self, sample_ruleset, monkeypatch):
        """Test that backtest handles errors gracefully."""
        from api import crud_client

        mock_client = MagicMock()
        mock_client.stub.GetBacktestFeatures.side_effect = Exception("Analytics error")
        monkeypatch.setattr(crud_client, "_client", mock_client)

        runner = BacktestRunner()
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(
            sample_ruleset, start_date, end_date, base_score=50
        )

        assert result.error is not None
        assert "Analytics error" in result.error

    def test_compute_metrics_calculates_statistics(self):
        """Test that metrics computation is correct."""
        runner = BacktestRunner()

        scores = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
        matched_counts = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        rejected_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        metrics = runner._compute_metrics(10, scores, matched_counts, rejected_counts)

        assert metrics.total_records == 10
        assert metrics.matched_count == 5
        assert metrics.match_rate == 0.5
        assert metrics.rejected_count == 1
        assert metrics.rejected_rate == 0.1
        assert metrics.score_min == 10
        assert metrics.score_max == 99
        assert metrics.score_mean == pytest.approx(54.9, abs=0.1)


def _make_backtest_result_pb(
    job_id: str,
    rule_id: str,
    metrics: BacktestMetrics,
    completed_at: datetime,
) -> MagicMock:
    """Create a mock proto BacktestResult."""
    result = MagicMock()
    result.job_id = job_id
    result.rule_id = rule_id
    result.ruleset_version = "v1"
    result.start_date = MagicMock()
    result.start_date.ToDatetime.return_value = datetime.now(timezone.utc) - timedelta(days=7)
    result.end_date = MagicMock()
    result.end_date.ToDatetime.return_value = datetime.now(timezone.utc)
    result.completed_at = MagicMock()
    result.completed_at.ToDatetime.return_value = completed_at
    result.error = ""

    result.metrics = MagicMock()
    result.metrics.total_records = metrics.total_records
    result.metrics.matched_count = metrics.matched_count
    result.metrics.match_rate = metrics.match_rate
    result.metrics.score_distribution = metrics.score_distribution
    result.metrics.score_mean = metrics.score_mean
    result.metrics.score_std = metrics.score_std
    result.metrics.score_min = metrics.score_min
    result.metrics.score_max = metrics.score_max
    result.metrics.rejected_count = metrics.rejected_count
    result.metrics.rejected_rate = metrics.rejected_rate

    return result


class TestBacktestStore:
    """Tests for BacktestStore class."""

    @pytest.fixture
    def sample_metrics(self):
        return BacktestMetrics(
            total_records=100,
            matched_count=25,
            match_rate=0.25,
            score_distribution={},
            score_mean=50.0,
            score_std=20.0,
            score_min=10,
            score_max=95,
            rejected_count=5,
            rejected_rate=0.05,
        )

    def test_save_and_get_result(self, sample_metrics, monkeypatch):
        """Test retrieving backtest results via analytics service."""
        from api import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.result = _make_backtest_result_pb(
            "test_job", "test_rule", sample_metrics, datetime.now(timezone.utc)
        )
        mock_client.stub.GetBacktestResult.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        store = BacktestStore()
        retrieved = store.get("test_job")

        assert retrieved is not None
        assert retrieved.job_id == "test_job"
        assert retrieved.rule_id == "test_rule"

    def test_list_results_with_filters(self, sample_metrics, monkeypatch):
        """Test listing results with filters via analytics service."""
        from api import crud_client

        base_time = datetime.now(timezone.utc)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = [
            _make_backtest_result_pb("job1", "rule1", sample_metrics, base_time - timedelta(days=1)),
        ]
        mock_client.stub.ListBacktestResults.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        store = BacktestStore()
        results = store.list_results(rule_id="rule1")

        assert len(results) == 1
        assert results[0].job_id == "job1"

    def test_get_returns_none_on_error(self, monkeypatch):
        """Test that get returns None when analytics service errors."""
        from api import crud_client

        mock_client = MagicMock()
        mock_client.stub.GetBacktestResult.side_effect = Exception("Not found")
        monkeypatch.setattr(crud_client, "_client", mock_client)

        store = BacktestStore()
        result = store.get("nonexistent_job")

        assert result is None
