"""Metrics collection and comparison for rule effectiveness."""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RuleMetrics:
    """Metrics for a single rule over a time period."""

    rule_id: str
    period_start: datetime
    period_end: datetime
    production_matches: int = 0
    shadow_matches: int = 0
    overlap_count: int = 0  # Records where both production and shadow matched

    @property
    def production_only_count(self) -> int:
        """Count of production-only matches."""
        return self.production_matches - self.overlap_count

    @property
    def shadow_only_count(self) -> int:
        """Count of shadow-only matches."""
        return self.shadow_matches - self.overlap_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["period_start"] = self.period_start.isoformat()
        data["period_end"] = self.period_end.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleMetrics":
        """Create from dictionary."""
        for field in ["period_start", "period_end"]:
            if isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


@dataclass
class ComparisonReport:
    """Comparison report between production and shadow rules."""

    period_start: datetime
    period_end: datetime
    rule_metrics: list[RuleMetrics]
    total_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "rule_metrics": [rm.to_dict() for rm in self.rule_metrics],
            "total_requests": self.total_requests,
        }


class MetricsCollector:
    """Collects metrics for rule matches in production and shadow mode."""

    def __init__(self, storage_path: Path | str | None = None):
        """Initialize metrics collector.

        Args:
            storage_path: Path to file for persistent storage. If None, uses in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        # In-memory counters: (rule_id, date) -> (production_count, shadow_count, overlap_count)
        self._counters: dict[tuple[str, str], tuple[int, int, int]] = defaultdict(
            lambda: (0, 0, 0)
        )
        # Track request-level matches for overlap calculation
        self._request_matches: list[dict[str, Any]] = []

        # Load existing metrics if file exists
        if self.storage_path and self.storage_path.exists():
            self._load_metrics()

    def _load_metrics(self) -> None:
        """Load metrics from storage file."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                self._counters = {}
                for key, value in data.get("counters", {}).items():
                    # Key format: "rule_id|date_str"
                    rule_id, date_str = key.split("|", 1)
                    prod, shadow, overlap = value
                    self._counters[(rule_id, date_str)] = (prod, shadow, overlap)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load metrics from {self.storage_path}: {e}")

    def _save_metrics(self) -> None:
        """Save metrics to storage file."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert counters to serializable format
            counters_data = {}
            for (rule_id, date), (prod, shadow, overlap) in self._counters.items():
                key = f"{rule_id}|{date}"
                counters_data[key] = [prod, shadow, overlap]

            data = {"counters": counters_data}

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(f"Failed to save metrics to {self.storage_path}: {e}")

    def record_match(
        self,
        rule_id: str,
        is_production: bool,
        is_shadow: bool,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a rule match.

        Args:
            rule_id: ID of the rule that matched.
            is_production: Whether this is a production rule match.
            is_shadow: Whether this is a shadow rule match.
            timestamp: Timestamp of the match. If None, uses current time.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        date_str = timestamp.date().isoformat()
        key = (rule_id, date_str)

        current_prod, current_shadow, current_overlap = self._counters[key]

        if is_production:
            current_prod += 1
        if is_shadow:
            current_shadow += 1
        if is_production and is_shadow:
            current_overlap += 1

        self._counters[key] = (current_prod, current_shadow, current_overlap)
        self._save_metrics()

    def record_request_matches(
        self,
        production_matched: list[str],
        shadow_matched: list[str],
        timestamp: datetime | None = None,
    ) -> None:
        """Record matches for a single request (for overlap calculation).

        Args:
            production_matched: List of production rule IDs that matched.
            shadow_matched: List of shadow rule IDs that matched.
            timestamp: Timestamp of the request. If None, uses current time.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        all_rules = set(production_matched) | set(shadow_matched)
        overlap = set(production_matched) & set(shadow_matched)

        # Record each rule once, with correct flags
        for rule_id in all_rules:
            is_prod = rule_id in production_matched
            is_shadow = rule_id in shadow_matched
            self.record_match(rule_id, is_prod, is_shadow, timestamp=timestamp)

    def get_rule_metrics(
        self, rule_id: str, start_date: datetime, end_date: datetime
    ) -> RuleMetrics:
        """Get metrics for a specific rule over a time period.

        Args:
            rule_id: ID of the rule.
            start_date: Start of period.
            end_date: End of period.

        Returns:
            RuleMetrics for the rule.
        """
        production_total = 0
        shadow_total = 0
        overlap_total = 0

        current_date = start_date.date()
        end_date_only = end_date.date()

        while current_date <= end_date_only:
            date_str = current_date.isoformat()
            key = (rule_id, date_str)

            prod, shadow, overlap = self._counters.get(key, (0, 0, 0))
            production_total += prod
            shadow_total += shadow
            overlap_total += overlap

            # Move to next day
            from datetime import timedelta

            current_date += timedelta(days=1)

        return RuleMetrics(
            rule_id=rule_id,
            period_start=start_date,
            period_end=end_date,
            production_matches=production_total,
            shadow_matches=shadow_total,
            overlap_count=overlap_total,
        )

    def generate_comparison_report(
        self, rule_ids: list[str], start_date: datetime, end_date: datetime
    ) -> ComparisonReport:
        """Generate a comparison report for multiple rules.

        Args:
            rule_ids: List of rule IDs to include.
            start_date: Start of period.
            end_date: End of period.

        Returns:
            ComparisonReport with metrics for all rules.
        """
        rule_metrics = [
            self.get_rule_metrics(rule_id, start_date, end_date) for rule_id in rule_ids
        ]

        return ComparisonReport(
            period_start=start_date,
            period_end=end_date,
            rule_metrics=rule_metrics,
        )


# Global metrics collector instance
_global_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Global MetricsCollector instance.
    """
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector instance (for testing).

    Args:
        collector: MetricsCollector instance to use.
    """
    global _global_metrics_collector
    _global_metrics_collector = collector
