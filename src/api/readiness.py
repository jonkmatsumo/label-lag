"""Promotion readiness evaluation logic."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from api.rules import Rule, RuleStatus
from api.metrics import RuleMetrics
from api.audit import AuditLogger, AuditRecord

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    """Types of readiness policies."""
    
    STABILITY = "stability"      # Has it been unchanged for X hours?
    VOLUME = "volume"            # Has it seen enough traffic?
    PERFORMANCE = "performance"  # Are match rates within safe bounds?
    APPROVAL = "approval"        # Has it been manually approved?


class CheckStatus(str, Enum):
    """Status of a specific check."""
    
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single policy check."""
    
    policy_type: PolicyType
    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadinessReport:
    """Full readiness report for a rule."""
    
    rule_id: str
    timestamp: datetime
    overall_status: CheckStatus
    checks: list[CheckResult]
    
    @property
    def passed(self) -> bool:
        """True if all critical checks passed."""
        return self.overall_status in [CheckStatus.PASS, CheckStatus.WARN]


class ReadinessEvaluator:
    """Evaluates if a rule is ready for promotion to production."""
    
    def __init__(self, audit_logger: AuditLogger | None = None):
        """Initialize evaluator.
        
        Args:
            audit_logger: Logger to check rule history.
        """
        self.audit_logger = audit_logger
        
        # Policy thresholds
        self.min_shadow_hours = 24
        self.min_matches = 50
        self.max_match_rate = 0.50  # 50% match rate is suspicious for a rule
        
    def evaluate(
        self, 
        rule: Rule, 
        metrics: RuleMetrics | None, 
        total_requests: int = 0
    ) -> ReadinessReport:
        """Run all readiness checks.
        
        Args:
            rule: Rule to check.
            metrics: Operational metrics (optional).
            total_requests: Total traffic volume during period.
            
        Returns:
            ReadinessReport.
        """
        checks = []
        
        # 1. Stability Check (Time since creation/modification)
        # We need audit log to know when it was last changed.
        # Fallback to current time if naive, but ideally we check audit history.
        checks.append(self._check_stability(rule))
        
        # 2. Volume Check (Metrics)
        checks.append(self._check_volume(metrics))
        
        # 3. Performance Check (Match Rate)
        checks.append(self._check_performance(metrics, total_requests))
        
        # 4. Approval Check
        checks.append(self._check_approval(rule))
        
        # Determine overall status
        statuses = [c.status for c in checks]
        if CheckStatus.FAIL in statuses:
            overall = CheckStatus.FAIL
        elif CheckStatus.WARN in statuses:
            overall = CheckStatus.WARN
        else:
            overall = CheckStatus.PASS
            
        return ReadinessReport(
            rule_id=rule.id,
            timestamp=datetime.now(timezone.utc),
            overall_status=overall,
            checks=checks
        )

    def _check_stability(self, rule: Rule) -> CheckResult:
        """Check if rule has been stable (in shadow) for long enough."""
        # Without AuditLogger or detailed rule metadata, we can't fully check this.
        # We'll rely on AuditLogger if provided.
        
        if not self.audit_logger:
            return CheckResult(
                PolicyType.STABILITY, 
                "Minimum Aging", 
                CheckStatus.SKIP, 
                "Audit logger not available to verify age"
            )
            
        history = self.audit_logger.get_rule_history(rule.id)
        if not history:
             # Created but no history? Means just created.
             return CheckResult(
                PolicyType.STABILITY, 
                "Minimum Aging", 
                CheckStatus.FAIL, 
                "New rule with no history"
            )
            
        # Find last modification
        last_change = history[-1].timestamp
        age_hours = (datetime.now(timezone.utc) - last_change).total_seconds() / 3600
        
        status = CheckStatus.PASS if age_hours >= self.min_shadow_hours else CheckStatus.FAIL
        
        return CheckResult(
            PolicyType.STABILITY,
            "Minimum Aging",
            status,
            f"Rule age: {age_hours:.1f} hours (Required: {self.min_shadow_hours}h)",
            {"age_hours": age_hours}
        )

    def _check_volume(self, metrics: RuleMetrics | None) -> CheckResult:
        """Check if rule has seen enough traffic/matches."""
        if not metrics:
            return CheckResult(
                PolicyType.VOLUME,
                "Traffic Volume",
                CheckStatus.FAIL,
                "No metrics data available"
            )
            
        total_matches = metrics.production_matches + metrics.shadow_matches
        
        # If it's a shadow rule, we expect shadow matches
        if total_matches >= self.min_matches:
            return CheckResult(
                PolicyType.VOLUME,
                "Match Count",
                CheckStatus.PASS,
                f"Matches: {total_matches} (Required: {self.min_matches})",
                {"matches": total_matches}
            )
        else:
             return CheckResult(
                PolicyType.VOLUME,
                "Match Count",
                CheckStatus.WARN, # Warn only, as rare rules are valid
                f"Low match count: {total_matches} (Required: {self.min_matches})",
                 {"matches": total_matches}
            )

    def _check_performance(self, metrics: RuleMetrics | None, total_requests: int) -> CheckResult:
        """Check for anomalies in match rate."""
        if not metrics or total_requests == 0:
             return CheckResult(
                PolicyType.PERFORMANCE,
                "Anomaly Detection",
                CheckStatus.SKIP,
                "Insufficient traffic data"
            )
            
        total_matches = metrics.production_matches + metrics.shadow_matches
        match_rate = total_matches / total_requests
        
        if match_rate > self.max_match_rate:
             return CheckResult(
                PolicyType.PERFORMANCE,
                "Match Rate Safety",
                CheckStatus.FAIL,
                f"Match rate {match_rate:.1%} exceeds safety limit {self.max_match_rate:.1%}",
                {"match_rate": match_rate}
            )
            
        return CheckResult(
            PolicyType.PERFORMANCE,
            "Match Rate Safety",
            CheckStatus.PASS,
            f"Match rate {match_rate:.1%} within limits",
            {"match_rate": match_rate}
        )

    def _check_approval(self, rule: Rule) -> CheckResult:
        """Check if rule is approved."""
        if rule.status == "approved":
             return CheckResult(
                PolicyType.APPROVAL,
                "Manual Approval",
                CheckStatus.PASS,
                "Rule is approved"
            )
        elif rule.status == "active":
             return CheckResult(
                PolicyType.APPROVAL,
                "Manual Approval",
                CheckStatus.PASS,
                "Rule is already active"
            )
        else:
             return CheckResult(
                PolicyType.APPROVAL,
                "Manual Approval",
                CheckStatus.FAIL,
                f"Rule status is '{rule.status}', requires 'approved'",
                {"status": rule.status}
            )
