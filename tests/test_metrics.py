"""Tests for metrics collection and comparison."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from api.metrics import (
    MetricsCollector,
    RuleMetrics,
    get_metrics_collector,
    set_metrics_collector,
)


class TestRuleMetrics:
    """Tests for RuleMetrics dataclass."""

    def test_rule_metrics_creation(self):
        """Test creating rule metrics."""
        metrics = RuleMetrics(
            rule_id="test_rule",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            production_matches=100,
            shadow_matches=80,
            overlap_count=60,
        )

        assert metrics.rule_id == "test_rule"
        assert metrics.production_matches == 100
        assert metrics.shadow_matches == 80
        assert metrics.overlap_count == 60
        assert metrics.production_only_count == 40  # 100 - 60
        assert metrics.shadow_only_count == 20  # 80 - 60


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a metrics collector for testing."""
        return MetricsCollector()

    def test_record_production_match(self, collector):
        """Test recording a production rule match."""
        collector.record_match("rule1", is_production=True, is_shadow=False)

        metrics = collector.get_rule_metrics(
            "rule1",
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )

        assert metrics.production_matches == 1
        assert metrics.shadow_matches == 0

    def test_record_shadow_match(self, collector):
        """Test recording a shadow rule match."""
        collector.record_match("rule1", is_production=False, is_shadow=True)

        metrics = collector.get_rule_metrics(
            "rule1",
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )

        assert metrics.production_matches == 0
        assert metrics.shadow_matches == 1

    def test_record_overlap(self, collector):
        """Test recording overlap (both production and shadow match)."""
        collector.record_match("rule1", is_production=True, is_shadow=True)

        metrics = collector.get_rule_metrics(
            "rule1",
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )

        assert metrics.production_matches == 1
        assert metrics.shadow_matches == 1
        assert metrics.overlap_count == 1

    def test_record_request_matches(self, collector):
        """Test recording matches for a single request."""
        collector.record_request_matches(
            production_matched=["rule1", "rule2"],
            shadow_matched=["rule1", "rule3"],
        )

        # rule1 should have overlap
        metrics1 = collector.get_rule_metrics(
            "rule1",
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )
        assert metrics1.production_matches == 1
        assert metrics1.shadow_matches == 1
        assert metrics1.overlap_count == 1

        # rule2 should be production only
        metrics2 = collector.get_rule_metrics(
            "rule2",
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )
        assert metrics2.production_matches == 1
        assert metrics2.shadow_matches == 0

        # rule3 should be shadow only
        metrics3 = collector.get_rule_metrics(
            "rule3",
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc),
        )
        assert metrics3.production_matches == 0
        assert metrics3.shadow_matches == 1

    def test_get_rule_metrics_multiple_days(self, collector):
        """Test getting metrics across multiple days."""
        base_time = datetime.now(timezone.utc)

        # Record matches on different days
        collector.record_match(
            "rule1", True, False, timestamp=base_time - timedelta(days=2)
        )
        collector.record_match(
            "rule1", True, False, timestamp=base_time - timedelta(days=1)
        )
        collector.record_match("rule1", False, True, timestamp=base_time)

        metrics = collector.get_rule_metrics(
            "rule1", base_time - timedelta(days=3), base_time + timedelta(days=1)
        )

        assert metrics.production_matches == 2
        assert metrics.shadow_matches == 1

    def test_generate_comparison_report(self, collector):
        """Test generating a comparison report."""
        # Record matches for multiple rules
        collector.record_request_matches(["rule1", "rule2"], ["rule1", "rule3"])
        collector.record_request_matches(["rule1"], ["rule2"])

        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc) + timedelta(days=1)

        report = collector.generate_comparison_report(
            ["rule1", "rule2", "rule3"], start_date, end_date
        )

        assert len(report.rule_metrics) == 3
        rule1_metrics = next(m for m in report.rule_metrics if m.rule_id == "rule1")
        assert rule1_metrics.production_matches > 0
        assert rule1_metrics.shadow_matches > 0

    def test_persistent_storage(self):
        """Test saving and loading metrics from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "metrics.json"

            # Create collector and record matches
            collector1 = MetricsCollector(storage_path=storage_path)
            collector1.record_match("rule1", True, False)
            collector1.record_match("rule2", False, True)

            # Create new collector and load
            collector2 = MetricsCollector(storage_path=storage_path)
            metrics1 = collector2.get_rule_metrics(
                "rule1",
                datetime.now(timezone.utc) - timedelta(days=1),
                datetime.now(timezone.utc),
            )

            assert metrics1.production_matches == 1


class TestGlobalMetricsCollector:
    """Tests for global metrics collector."""

    def test_get_metrics_collector_returns_singleton(self):
        """Test that get_metrics_collector returns a singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

    def test_set_metrics_collector_for_testing(self):
        """Test that set_metrics_collector allows replacing for testing."""
        original = get_metrics_collector()
        test_collector = MetricsCollector()

        set_metrics_collector(test_collector)
        assert get_metrics_collector() is test_collector

        # Restore original
        set_metrics_collector(original)
        assert get_metrics_collector() is original
