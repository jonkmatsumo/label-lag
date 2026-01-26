"""Tests for approval quality signals computation."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from api.backtest import BacktestMetrics, BacktestResult, BacktestStore
from api.metrics import MetricsCollector, RuleMetrics
from api.rules import Rule, RuleSet, RuleStatus
from api.signals import (
    compute_approval_signals,
    compute_coverage_signals,
    compute_governance_signals,
    compute_structural_signals,
)


class TestStructuralSignals:
    """Tests for structural validity signals."""

    def test_no_conflicts_no_redundancies(self):
        """Test signals when rule has no conflicts or redundancies."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=10,
            action="clamp_min",
            score=80,
        )
        production_ruleset = RuleSet(
            version="v1",
            rules=[
                Rule(
                    id="other_rule",
                    field="amount_to_avg_ratio_30d",
                    op=">",
                    value=3.0,
                    action="clamp_min",
                    score=75,
                )
            ],
        )
        draft_ruleset = RuleSet(version="draft", rules=[])

        signals, unavailable = compute_structural_signals(
            rule, production_ruleset, draft_ruleset
        )

        assert len(unavailable) == 0
        has_conflicts_signal = next(
            (s for s in signals if s.signal_id == "has_conflicts"), None
        )
        assert has_conflicts_signal is not None
        assert has_conflicts_signal.value is False
        assert has_conflicts_signal.severity == "info"

    def test_conflicts_detected(self):
        """Test that conflicts are detected and marked as risk."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="reject",
        )
        production_ruleset = RuleSet(
            version="v1",
            rules=[
                Rule(
                    id="conflicting_rule",
                    field="velocity_24h",
                    op=">",
                    value=3,
                    action="clamp_min",
                    score=80,
                )
            ],
        )
        draft_ruleset = RuleSet(version="draft", rules=[])

        signals, unavailable = compute_structural_signals(
            rule, production_ruleset, draft_ruleset
        )

        has_conflicts_signal = next(
            (s for s in signals if s.signal_id == "has_conflicts"), None
        )
        assert has_conflicts_signal is not None
        assert has_conflicts_signal.value is True
        assert has_conflicts_signal.severity == "risk"

        conflict_count_signal = next(
            (s for s in signals if s.signal_id == "conflict_count"), None
        )
        assert conflict_count_signal is not None
        assert conflict_count_signal.value > 0

    def test_redundancies_detected(self):
        """Test that redundancies are detected and marked as warning."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        production_ruleset = RuleSet(
            version="v1",
            rules=[
                Rule(
                    id="redundant_rule",
                    field="velocity_24h",
                    op=">",
                    value=3,  # Subset condition
                    action="clamp_min",
                    score=80,
                )
            ],
        )
        draft_ruleset = RuleSet(version="draft", rules=[])

        signals, unavailable = compute_structural_signals(
            rule, production_ruleset, draft_ruleset
        )

        has_redundancies_signal = next(
            (s for s in signals if s.signal_id == "has_redundancies"), None
        )
        assert has_redundancies_signal is not None
        assert has_redundancies_signal.value is True
        assert has_redundancies_signal.severity == "warning"


class TestCoverageSignals:
    """Tests for coverage/match impact signals."""

    @patch("api.signals.get_metrics_collector")
    @patch("api.signals.get_draft_store")
    def test_shadow_metrics_available(self, mock_draft_store, mock_metrics_collector):
        """Test shadow metrics when available."""
        # Mock metrics collector
        mock_collector = MagicMock(spec=MetricsCollector)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        mock_collector.get_rule_metrics.return_value = RuleMetrics(
            rule_id="test_rule",
            period_start=start_date,
            period_end=end_date,
            shadow_matches=1247,
            production_matches=0,
            overlap_count=0,
        )
        mock_metrics_collector.return_value = mock_collector

        # Mock draft store
        mock_store = MagicMock()
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status=RuleStatus.SHADOW.value,
        )
        mock_store.get.return_value = rule
        mock_draft_store.return_value = mock_store

        # Mock audit logger
        with patch("api.signals.get_audit_logger") as mock_audit:
            mock_audit_logger = MagicMock()
            mock_audit_logger.query.return_value = []
            mock_audit.return_value = mock_audit_logger

            signals, unavailable = compute_coverage_signals("test_rule")

        shadow_match_signal = next(
            (s for s in signals if s.signal_id == "shadow_match_count"), None
        )
        assert shadow_match_signal is not None
        assert shadow_match_signal.value == 1247

    @patch("api.signals.BacktestStore")
    def test_backtest_available(self, mock_backtest_store_class):
        """Test backtest signals when available."""
        # Mock backtest store
        mock_store = MagicMock(spec=BacktestStore)
        mock_result = BacktestResult(
            job_id="job1",
            rule_id="test_rule",
            ruleset_version="v1",
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc),
            metrics=BacktestMetrics(
                total_records=1000,
                matched_count=23,
                match_rate=0.023,
                score_distribution={},
                score_mean=50.0,
                score_std=10.0,
                score_min=1,
                score_max=99,
                rejected_count=0,
                rejected_rate=0.0,
            ),
            completed_at=datetime.now(timezone.utc),
        )
        mock_store.list_results.return_value = [mock_result]
        mock_backtest_store_class.return_value = mock_store

        signals, unavailable = compute_coverage_signals("test_rule")

        has_backtest_signal = next(
            (s for s in signals if s.signal_id == "has_backtest"), None
        )
        assert has_backtest_signal is not None
        assert has_backtest_signal.value is True

        match_rate_signal = next(
            (s for s in signals if s.signal_id == "backtest_match_rate"), None
        )
        assert match_rate_signal is not None
        assert match_rate_signal.value == 0.023

    @patch("api.signals.BacktestStore")
    def test_no_backtest(self, mock_backtest_store_class):
        """Test backtest signals when no backtest exists."""
        mock_store = MagicMock(spec=BacktestStore)
        mock_store.list_results.return_value = []
        mock_backtest_store_class.return_value = mock_store

        signals, unavailable = compute_coverage_signals("test_rule")

        has_backtest_signal = next(
            (s for s in signals if s.signal_id == "has_backtest"), None
        )
        assert has_backtest_signal is not None
        assert has_backtest_signal.value is False


class TestGovernanceSignals:
    """Tests for governance/process signals."""

    @patch("api.signals.get_version_store")
    def test_version_count(self, mock_version_store):
        """Test version count signal."""
        mock_store = MagicMock()
        mock_store.list_versions.return_value = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        mock_version_store.return_value = mock_store

        signals, unavailable = compute_governance_signals("test_rule")

        version_count_signal = next(
            (s for s in signals if s.signal_id == "version_count"), None
        )
        assert version_count_signal is not None
        assert version_count_signal.value == 3

    @patch("api.signals.get_audit_logger")
    def test_submitter_and_days_in_review(self, mock_audit_logger):
        """Test submitter and days in review signals."""
        mock_logger = MagicMock()
        submitted_at = datetime.now(timezone.utc) - timedelta(days=5)
        mock_record = MagicMock()
        mock_record.action = "state_change"
        mock_record.after_state = {"status": RuleStatus.PENDING_REVIEW.value}
        mock_record.timestamp = submitted_at
        mock_record.actor = "submitter_user"
        mock_logger.query.return_value = [mock_record]
        mock_audit_logger.return_value = mock_logger

        signals, unavailable = compute_governance_signals("test_rule")

        submitter_signal = next(
            (s for s in signals if s.signal_id == "submitter_actor"), None
        )
        assert submitter_signal is not None
        assert submitter_signal.value == "submitter_user"

        days_signal = next(
            (s for s in signals if s.signal_id == "days_in_review"), None
        )
        assert days_signal is not None
        assert days_signal.value == 5


class TestComputeApprovalSignals:
    """Tests for the main compute_approval_signals function."""

    @patch("api.signals.get_draft_store")
    @patch("api.signals.compute_structural_signals")
    @patch("api.signals.compute_coverage_signals")
    @patch("api.signals.compute_governance_signals")
    def test_compute_all_signals(
        self,
        mock_governance,
        mock_coverage,
        mock_structural,
        mock_draft_store,
    ):
        """Test that all signal categories are computed."""
        # Mock draft store
        mock_store = MagicMock()
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        mock_store.get.return_value = rule
        mock_draft_store.return_value = mock_store

        # Mock signal computations
        mock_structural.return_value = (
            [
                MagicMock(signal_id="has_conflicts", severity="info", value=False),
            ],
            [],
        )
        mock_coverage.return_value = (
            [
                MagicMock(signal_id="has_backtest", severity="info", value=True),
            ],
            [],
        )
        mock_governance.return_value = (
            [
                MagicMock(signal_id="version_count", severity="info", value=1),
            ],
            [],
        )

        response = compute_approval_signals("test_rule")

        assert response.rule_id == "test_rule"
        assert len(response.signals) == 3
        assert response.summary.info_count == 3
        assert response.summary.risk_count == 0
        assert response.summary.warning_count == 0
        assert response.summary.has_blockers is False

    @patch("api.signals.get_draft_store")
    def test_rule_not_found(self, mock_draft_store):
        """Test that ValueError is raised when rule not found."""
        mock_store = MagicMock()
        mock_store.get.return_value = None
        mock_draft_store.return_value = mock_store

        with pytest.raises(ValueError, match="Rule not found"):
            compute_approval_signals("nonexistent_rule")

    @patch("api.signals.get_draft_store")
    @patch("api.signals.compute_structural_signals")
    @patch("api.signals.compute_coverage_signals")
    @patch("api.signals.compute_governance_signals")
    def test_partial_signals(
        self,
        mock_governance,
        mock_coverage,
        mock_structural,
        mock_draft_store,
    ):
        """Test partial signal availability."""
        mock_store = MagicMock()
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        mock_store.get.return_value = rule
        mock_draft_store.return_value = mock_store

        # Some signals unavailable
        mock_structural.return_value = ([], ["has_conflicts"])
        mock_coverage.return_value = ([], ["shadow_match_count"])
        mock_governance.return_value = ([], [])

        response = compute_approval_signals("test_rule")

        assert response.partial is True
        assert len(response.unavailable_signals) == 2
        assert "has_conflicts" in response.unavailable_signals
        assert "shadow_match_count" in response.unavailable_signals
