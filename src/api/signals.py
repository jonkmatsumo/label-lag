"""Approval quality signals for rule review decisions."""

import logging
from datetime import datetime, timezone
from typing import Any

from api.backtest import BacktestStore
from api.draft_store import get_draft_store
from api.metrics import get_metrics_collector
from api.rules import Rule, RuleSet, RuleStatus
from api.schemas import ApprovalSignalItem, ApprovalSignalsResponse, ApprovalSignalsSummary
from api.validation import detect_conflicts, detect_redundancies
from api.versioning import get_version_store

logger = logging.getLogger(__name__)


def compute_structural_signals(
    rule: Rule, production_ruleset: RuleSet | None, draft_ruleset: RuleSet | None
) -> list[ApprovalSignalItem]:
    """Compute structural validity signals (conflicts, redundancies).

    Args:
        rule: Rule to analyze.
        production_ruleset: Current production ruleset.
        draft_ruleset: Current draft ruleset (excluding this rule).

    Returns:
        List of structural signals.
    """
    signals = []
    unavailable = []

    try:
        # Build test ruleset with this rule + all existing rules
        all_rules = [rule]
        if production_ruleset:
            all_rules.extend(production_ruleset.rules)
        if draft_ruleset:
            # Exclude the rule we're analyzing
            all_rules.extend(
                [r for r in draft_ruleset.rules if r.id != rule.id]
            )

        test_ruleset = RuleSet(version="test", rules=all_rules)

        # Detect conflicts
        conflicts = detect_conflicts(test_ruleset)
        rule_conflicts = [
            c for c in conflicts if c.rule1_id == rule.id or c.rule2_id == rule.id
        ]

        has_conflicts = len(rule_conflicts) > 0
        conflict_count = len(rule_conflicts)

        signals.append(
            ApprovalSignalItem(
                signal_id="has_conflicts",
                category="structural",
                severity="risk" if has_conflicts else "info",
                value=has_conflicts,
                label="Has Conflicts",
                description=(
                    f"Rule conflicts with {conflict_count} existing rule(s)"
                    if has_conflicts
                    else "No conflicts detected"
                ),
            )
        )

        signals.append(
            ApprovalSignalItem(
                signal_id="conflict_count",
                category="structural",
                severity="info",
                value=conflict_count,
                label="Conflict Count",
                description=f"Number of conflicting rules: {conflict_count}",
            )
        )

        # Detect redundancies
        redundancies = detect_redundancies(test_ruleset)
        rule_redundancies = [
            r for r in redundancies if r.rule_id == rule.id
        ]

        has_redundancies = len(rule_redundancies) > 0
        redundancy_count = len(rule_redundancies)

        signals.append(
            ApprovalSignalItem(
                signal_id="has_redundancies",
                category="structural",
                severity="warning" if has_redundancies else "info",
                value=has_redundancies,
                label="Has Redundancies",
                description=(
                    f"Rule is redundant with {redundancy_count} existing rule(s)"
                    if has_redundancies
                    else "No redundancies detected"
                ),
            )
        )

        signals.append(
            ApprovalSignalItem(
                signal_id="redundancy_count",
                category="structural",
                severity="info",
                value=redundancy_count,
                label="Redundancy Count",
                description=f"Number of redundant rules: {redundancy_count}",
            )
        )

    except Exception as e:
        logger.warning(f"Failed to compute structural signals: {e}")
        unavailable.extend(["has_conflicts", "conflict_count", "has_redundancies", "redundancy_count"])

    return signals, unavailable


def compute_coverage_signals(rule_id: str) -> tuple[list[ApprovalSignalItem], list[str]]:
    """Compute coverage/match impact signals (shadow metrics, backtest).

    Args:
        rule_id: Rule ID to analyze.

    Returns:
        Tuple of (signals list, unavailable signal IDs).
    """
    signals = []
    unavailable = []

    # Shadow metrics
    try:
        collector = get_metrics_collector()
        # Get metrics for last 30 days
        end_date = datetime.now(timezone.utc)
        from datetime import timedelta

        start_date = end_date - timedelta(days=30)

        # Get rule metrics
        rule_metrics = collector.get_rule_metrics(rule_id, start_date, end_date)
        shadow_match_count = rule_metrics.shadow_matches if rule_metrics.shadow_matches > 0 else None

        # Check if rule has been in shadow mode and count days
        draft_store = get_draft_store()
        rule = draft_store.get(rule_id)
        shadow_days_active = 0

        if rule and rule.status == RuleStatus.SHADOW.value:
            # Check audit log for when rule was moved to shadow
            from api.audit import get_audit_logger

            audit_logger = get_audit_logger()
            records = audit_logger.query(rule_id=rule_id, action="state_change")
            for record in reversed(records):  # Check most recent first
                if (
                    record.after_state
                    and record.after_state.get("status") == RuleStatus.SHADOW.value
                ):
                    shadow_days_active = (datetime.now(timezone.utc) - record.timestamp).days
                    break

        signals.append(
            ApprovalSignalItem(
                signal_id="shadow_match_count",
                category="coverage",
                severity="info",
                value=shadow_match_count,
                label="Shadow Match Count",
                description=(
                    f"Matches in shadow mode: {shadow_match_count}"
                    if shadow_match_count is not None
                    else "No shadow metrics available"
                ),
            )
        )

        signals.append(
            ApprovalSignalItem(
                signal_id="shadow_days_active",
                category="coverage",
                severity="info",
                value=shadow_days_active,
                label="Shadow Days Active",
                description=f"Days rule has been in shadow mode: {shadow_days_active}",
            )
        )

    except Exception as e:
        logger.warning(f"Failed to compute shadow metrics: {e}")
        unavailable.extend(["shadow_match_count", "shadow_days_active"])

    # Backtest results
    try:
        backtest_store = BacktestStore()
        # Get most recent backtest for this rule
        backtest_results = backtest_store.list_results(rule_id=rule_id)

        # Sort by completed_at descending and take first
        backtest_results.sort(key=lambda r: r.completed_at, reverse=True)
        has_backtest = len(backtest_results) > 0
        backtest_match_rate = None

        if has_backtest and backtest_results:
            latest = backtest_results[0]
            backtest_match_rate = latest.metrics.match_rate

        signals.append(
            ApprovalSignalItem(
                signal_id="has_backtest",
                category="coverage",
                severity="info",
                value=has_backtest,
                label="Has Backtest",
                description=(
                    "Backtest results available" if has_backtest else "No backtest run"
                ),
            )
        )

        signals.append(
            ApprovalSignalItem(
                signal_id="backtest_match_rate",
                category="coverage",
                severity="info",
                value=backtest_match_rate,
                label="Backtest Match Rate",
                description=(
                    f"Match rate from backtest: {backtest_match_rate:.2%}"
                    if backtest_match_rate is not None
                    else "No backtest data"
                ),
            )
        )

    except Exception as e:
        logger.warning(f"Failed to compute backtest signals: {e}")
        unavailable.extend(["has_backtest", "backtest_match_rate"])

    return signals, unavailable


def compute_governance_signals(rule_id: str) -> tuple[list[ApprovalSignalItem], list[str]]:
    """Compute governance/process signals (version count, days in review, submitter).

    Args:
        rule_id: Rule ID to analyze.

    Returns:
        Tuple of (signals list, unavailable signal IDs).
    """
    signals = []
    unavailable = []

    # Version count
    try:
        version_store = get_version_store()
        versions = version_store.list_versions(rule_id)
        version_count = len(versions)

        signals.append(
            ApprovalSignalItem(
                signal_id="version_count",
                category="governance",
                severity="info",
                value=version_count,
                label="Version Count",
                description=f"Number of prior versions: {version_count}",
            )
        )
    except Exception as e:
        logger.warning(f"Failed to compute version count: {e}")
        unavailable.append("version_count")

    # Days in review and submitter
    try:
        from api.audit import get_audit_logger

        audit_logger = get_audit_logger()
        # Get audit records for this rule
        records = audit_logger.query(rule_id=rule_id)

        # Find when rule was submitted (state_change to pending_review)
        submitted_at = None
        submitter_actor = None
        for record in records:
            if (
                record.action == "state_change"
                and record.after_state
                and record.after_state.get("status") == RuleStatus.PENDING_REVIEW.value
            ):
                submitted_at = record.timestamp
                submitter_actor = record.actor
                break

        days_in_review = None
        if submitted_at:
            days_in_review = (datetime.now(timezone.utc) - submitted_at).days

        signals.append(
            ApprovalSignalItem(
                signal_id="submitter_actor",
                category="governance",
                severity="info",
                value=submitter_actor,
                label="Submitter",
                description=(
                    f"Submitted by: {submitter_actor}"
                    if submitter_actor
                    else "No submitter found"
                ),
            )
        )

        signals.append(
            ApprovalSignalItem(
                signal_id="days_in_review",
                category="governance",
                severity="info",
                value=days_in_review,
                label="Days in Review",
                description=(
                    f"Days since submission: {days_in_review}"
                    if days_in_review is not None
                    else "Not submitted"
                ),
            )
        )

    except Exception as e:
        logger.warning(f"Failed to compute governance signals: {e}")
        unavailable.extend(["submitter_actor", "days_in_review"])

    return signals, unavailable


def compute_approval_signals(
    rule_id: str,
    production_ruleset: RuleSet | None = None,
    draft_ruleset: RuleSet | None = None,
) -> ApprovalSignalsResponse:
    """Compute all approval quality signals for a rule.

    Args:
        rule_id: Rule ID to analyze.
        production_ruleset: Current production ruleset (optional).
        draft_ruleset: Current draft ruleset (optional).

    Returns:
        ApprovalSignalsResponse with all computed signals.
    """
    all_signals = []
    all_unavailable = []

    # Get the rule
    draft_store = get_draft_store()
    rule = draft_store.get(rule_id)

    if rule is None:
        raise ValueError(f"Rule not found: {rule_id}")

    # Compute structural signals
    structural_signals, structural_unavailable = compute_structural_signals(
        rule, production_ruleset, draft_ruleset
    )
    all_signals.extend(structural_signals)
    all_unavailable.extend(structural_unavailable)

    # Compute coverage signals
    coverage_signals, coverage_unavailable = compute_coverage_signals(rule_id)
    all_signals.extend(coverage_signals)
    all_unavailable.extend(coverage_unavailable)

    # Compute governance signals
    governance_signals, governance_unavailable = compute_governance_signals(rule_id)
    all_signals.extend(governance_signals)
    all_unavailable.extend(governance_unavailable)

    # Compute summary
    risk_count = sum(1 for s in all_signals if s.severity == "risk")
    warning_count = sum(1 for s in all_signals if s.severity == "warning")
    info_count = sum(1 for s in all_signals if s.severity == "info")

    summary = ApprovalSignalsSummary(
        risk_count=risk_count,
        warning_count=warning_count,
        info_count=info_count,
        has_blockers=risk_count > 0,
    )

    return ApprovalSignalsResponse(
        rule_id=rule_id,
        computed_at=datetime.now(timezone.utc).isoformat(),
        signals=all_signals,
        summary=summary,
        partial=len(all_unavailable) > 0,
        unavailable_signals=all_unavailable,
    )
