"""Rule health analysis and classification."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from api.metrics import RuleMetrics
from api.rules import Rule, RuleStatus

logger = logging.getLogger(__name__)


class RuleHealth(str, Enum):
    """Health status of a rule."""

    HEALTHY = "healthy"
    STALE = "stale"  # No matches in X days
    NOISY = "noisy"  # Too many matches
    INEFFECTIVE = "ineffective"  # Matches but zero impact
    ERROR = "error"  # High error rate (future)
    UNKNOWN = "unknown"


@dataclass
class HealthReport:
    """Health report for a single rule."""

    rule_id: str
    status: RuleHealth
    reason: str
    metrics: RuleMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "status": self.status.value,
            "reason": self.reason,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class RuleHealthEvaluator:
    """Evaluates the health of rules based on metrics."""

    def __init__(
        self,
        stale_days: int = 7,
        noisy_threshold: float = 0.20,  # 20% match rate
    ):
        """Initialize evaluator.

        Args:
            stale_days: Days without matches to consider stale.
            noisy_threshold: Match rate threshold to consider noisy.
        """
        self.stale_days = stale_days
        self.noisy_threshold = noisy_threshold

    def evaluate(
        self, rule: Rule, metrics: RuleMetrics, total_requests: int
    ) -> HealthReport:
        """Evaluate rule health.

        Args:
            rule: The rule object.
            metrics: Metrics for the rule.
            total_requests: Total requests in the period.

        Returns:
            HealthReport.
        """
        if rule.status == RuleStatus.DISABLED.value:
            return HealthReport(rule.id, RuleHealth.UNKNOWN, "Rule is disabled")

        total_matches = metrics.production_matches + metrics.shadow_matches

        # Check for STALE
        if total_matches == 0:
            # Only stale if it's been active for long enough
            # For simplicity, we assume the metric window covers the stale period
            return HealthReport(
                rule.id,
                RuleHealth.STALE,
                f"No matches in the last {self.stale_days} days",
                metrics,
            )

        # Check for NOISY
        if total_requests > 0:
            match_rate = total_matches / total_requests
            if match_rate > self.noisy_threshold:
                return HealthReport(
                    rule.id,
                    RuleHealth.NOISY,
                    f"Match rate {match_rate:.1%} exceeds threshold "
                    f"{self.noisy_threshold:.1%}",
                    metrics,
                )

        # Check for INEFFECTIVE (Active but zero score impact)
        # Only applies to active rules that modify score
        if (
            rule.status == RuleStatus.ACTIVE.value
            and rule.action in ["override_score", "clamp_min", "clamp_max"]
            and metrics.total_score_delta == 0
        ):
            return HealthReport(
                rule.id,
                RuleHealth.INEFFECTIVE,
                "Rule matched but had zero impact on final scores",
                metrics,
            )

        return HealthReport(rule.id, RuleHealth.HEALTHY, "Operating normally", metrics)
