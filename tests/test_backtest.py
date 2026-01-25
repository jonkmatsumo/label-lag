"""Tests for backtesting infrastructure."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


class TestBacktestRunner:
    """Tests for BacktestRunner class."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.get_session.return_value.__enter__.return_value.execute.return_value = (
            iter(
                [
                    MagicMock(
                        velocity_24h=10,
                        amount_to_avg_ratio_30d=3.5,
                        balance_volatility_z_score=-1.5,
                        experimental_signals={
                            "bank_connections_24h": 5,
                            "merchant_risk_score": 75,
                            "has_history": True,
                        },
                    ),
                    MagicMock(
                        velocity_24h=2,
                        amount_to_avg_ratio_30d=1.0,
                        balance_volatility_z_score=0.0,
                        experimental_signals=None,
                    ),
                ]
            )
        )
        return session

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

    def test_run_backtest_with_matching_rules(self, mock_db_session, sample_ruleset):
        """Test running a backtest where rules match."""
        runner = BacktestRunner(db_session=mock_db_session)
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(
            sample_ruleset, start_date, end_date, base_score=50
        )

        assert result.job_id is not None
        assert result.ruleset_version == "v1"
        assert result.metrics.total_records == 2
        assert result.metrics.matched_count == 1  # First record matches
        assert result.metrics.match_rate > 0
        assert result.error is None

    def test_run_backtest_with_no_matches(self, mock_db_session):
        """Test running a backtest where no rules match."""
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

        runner = BacktestRunner(db_session=mock_db_session)
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(ruleset, start_date, end_date, base_score=50)

        assert result.metrics.matched_count == 0
        assert result.metrics.match_rate == 0.0

    def test_run_backtest_computes_score_distribution(
        self, mock_db_session, sample_ruleset
    ):
        """Test that backtest computes score distribution."""
        runner = BacktestRunner(db_session=mock_db_session)
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(
            sample_ruleset, start_date, end_date, base_score=50
        )

        assert "1-20" in result.metrics.score_distribution
        assert "21-40" in result.metrics.score_distribution
        assert "41-60" in result.metrics.score_distribution
        assert "61-80" in result.metrics.score_distribution
        assert "81-99" in result.metrics.score_distribution

    def test_run_backtest_handles_empty_data(self, mock_db_session, sample_ruleset):
        """Test that backtest handles empty historical data."""
        empty_session = MagicMock()
        enter_mock = empty_session.get_session.return_value.__enter__.return_value
        enter_mock.execute.return_value = iter([])

        runner = BacktestRunner(db_session=empty_session)
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(
            sample_ruleset, start_date, end_date, base_score=50
        )

        assert result.metrics.total_records == 0
        assert result.metrics.match_rate == 0.0

    def test_run_backtest_handles_errors_gracefully(self, sample_ruleset):
        """Test that backtest handles errors gracefully."""
        error_session = MagicMock()
        error_session.get_session.side_effect = Exception("Database error")

        runner = BacktestRunner(db_session=error_session)
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = runner.run_backtest(
            sample_ruleset, start_date, end_date, base_score=50
        )

        assert result.error is not None
        assert "Database error" in result.error

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


class TestBacktestStore:
    """Tests for BacktestStore class."""

    def test_save_and_get_result(self):
        """Test saving and retrieving backtest results."""
        store = BacktestStore()

        result = BacktestResult(
            job_id="test_job",
            rule_id="test_rule",
            ruleset_version="v1",
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc),
            metrics=BacktestMetrics(
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
            ),
            completed_at=datetime.now(timezone.utc),
        )

        store.save(result)
        retrieved = store.get("test_job")

        assert retrieved is not None
        assert retrieved.job_id == "test_job"
        assert retrieved.rule_id == "test_rule"

    def test_list_results_with_filters(self):
        """Test listing results with filters."""
        store = BacktestStore()

        base_time = datetime.now(timezone.utc)

        result1 = BacktestResult(
            job_id="job1",
            rule_id="rule1",
            ruleset_version="v1",
            start_date=base_time - timedelta(days=7),
            end_date=base_time,
            metrics=BacktestMetrics(
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
            ),
            completed_at=base_time - timedelta(days=1),
        )

        result2 = BacktestResult(
            job_id="job2",
            rule_id="rule2",
            ruleset_version="v1",
            start_date=base_time - timedelta(days=7),
            end_date=base_time,
            metrics=BacktestMetrics(
                total_records=100,
                matched_count=30,
                match_rate=0.30,
                score_distribution={},
                score_mean=55.0,
                score_std=20.0,
                score_min=10,
                score_max=95,
                rejected_count=3,
                rejected_rate=0.03,
            ),
            completed_at=base_time,
        )

        store.save(result1)
        store.save(result2)

        # Filter by rule_id
        results = store.list_results(rule_id="rule1")
        assert len(results) == 1
        assert results[0].job_id == "job1"

        # Filter by date range
        results = store.list_results(
            start_date=base_time - timedelta(hours=12),
            end_date=base_time + timedelta(hours=1),
        )
        assert len(results) == 1
        assert results[0].job_id == "job2"

    def test_persistent_storage(self):
        """Test saving and loading results from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "backtests.json"

            # Create store and save result
            store1 = BacktestStore(storage_path=storage_path)
            result = BacktestResult(
                job_id="test_job",
                rule_id="test_rule",
                ruleset_version="v1",
                start_date=datetime.now(timezone.utc) - timedelta(days=7),
                end_date=datetime.now(timezone.utc),
                metrics=BacktestMetrics(
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
                ),
                completed_at=datetime.now(timezone.utc),
            )
            store1.save(result)

            # Create new store and load
            store2 = BacktestStore(storage_path=storage_path)
            retrieved = store2.get("test_job")

            assert retrieved is not None
            assert retrieved.job_id == "test_job"
            assert retrieved.rule_id == "test_rule"
