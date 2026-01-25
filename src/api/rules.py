"""Rule-based decision engine for hybrid model inference.

Provides deterministic rules that can override, clamp, or gate model scores
in a versioned and explainable way.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RuleStatus(str, Enum):
    """Lifecycle status of a rule."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    DISABLED = "disabled"
    ARCHIVED = "archived"


@dataclass
class Rule:
    """A single decision rule."""

    id: str
    field: str
    op: str  # >, >=, <, <=, ==, in, not_in
    value: Any  # scalar or list (for in/not_in)
    action: str  # override_score, clamp_min, clamp_max, reject
    score: int | None = None
    severity: str = "medium"  # low, medium, high
    reason: str = ""
    status: str = field(default="active")  # draft, pending_review, active, disabled, archived

    def __post_init__(self):
        """Validate rule configuration."""
        # Validate status
        valid_statuses = [s.value for s in RuleStatus]
        if self.status not in valid_statuses:
            raise ValueError(
                f"Invalid status: {self.status}. Must be one of {valid_statuses}"
            )

        valid_ops = [">", ">=", "<", "<=", "==", "in", "not_in"]
        if self.op not in valid_ops:
            raise ValueError(f"Invalid operator: {self.op}. Must be one of {valid_ops}")

        valid_actions = ["override_score", "clamp_min", "clamp_max", "reject"]
        if self.action not in valid_actions:
            raise ValueError(
                f"Invalid action: {self.action}. Must be one of {valid_actions}"
            )

        score_actions = ["override_score", "clamp_min", "clamp_max"]
        if self.action in score_actions and self.score is None:
            raise ValueError(f"Action {self.action} requires 'score' field")

        if self.op in ["in", "not_in"] and not isinstance(self.value, list):
            raise ValueError(f"Operator {self.op} requires 'value' to be a list")

        if self.severity not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid severity: {self.severity}. Use low/medium/high")


@dataclass
class RuleResult:
    """Result of rule evaluation."""

    final_score: int
    matched_rules: list[str]  # rule IDs
    explanations: list[dict[str, Any]]  # {rule_id, severity, reason}
    rejected: bool = False


@dataclass
class RuleSet:
    """A collection of decision rules with versioning."""

    version: str
    rules: list[Rule]

    @classmethod
    def from_dict(cls, data: dict[str, Any], validate: bool = False, strict: bool = False) -> "RuleSet":
        """Create RuleSet from dictionary.

        Args:
            data: Dictionary with 'version' and 'rules' keys.
            validate: If True, run conflict and redundancy detection.
            strict: If True and validate=True, raise ValueError on conflicts.

        Returns:
            RuleSet instance.

        Raises:
            ValueError: If required fields are missing or invalid, or if strict=True
                and conflicts are detected.
        """
        if "version" not in data:
            raise ValueError("RuleSet must have 'version' field")
        if "rules" not in data:
            raise ValueError("RuleSet must have 'rules' field")

        if not isinstance(data["rules"], list):
            raise ValueError("'rules' must be a list")

        rules = []
        for i, rule_dict in enumerate(data["rules"]):
            try:
                # Backward compatibility: if status is missing, default to "active"
                if "status" not in rule_dict:
                    rule_dict = rule_dict.copy()
                    rule_dict["status"] = "active"
                rule = Rule(**rule_dict)
                rules.append(rule)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid rule at index {i}: {e}") from e

        ruleset = cls(version=data["version"], rules=rules)

        # Run validation if requested
        if validate:
            from api.validation import validate_ruleset

            validate_ruleset(ruleset, strict=strict)

        return ruleset

    @classmethod
    def load_from_file(cls, path: str | Path) -> "RuleSet":
        """Load RuleSet from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            RuleSet instance.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file is invalid JSON or missing required fields.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {path}")

        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in rules file: {e}") from e

    @classmethod
    def empty(cls, version: str = "v1") -> "RuleSet":
        """Create an empty RuleSet.

        Args:
            version: Version string for the empty ruleset.

        Returns:
            Empty RuleSet instance.
        """
        return cls(version=version, rules=[])


def evaluate_rules(
    features: dict[str, Any],
    current_score: int,
    ruleset: RuleSet,
) -> RuleResult:
    """Evaluate rules against features and adjust score.

    Rules are evaluated in order. Action precedence:
    - reject > override_score > clamp_min/clamp_max
    - Multiple rules can match; all are collected for explainability
    - If a feature is missing, the rule does not match (no error)

    Args:
        features: Dictionary of feature name -> value.
        current_score: Current score (1-99) before rule application.
        ruleset: RuleSet to evaluate.

    Returns:
        RuleResult with final score, matched rules, and explanations.
    """
    if not ruleset.rules:
        return RuleResult(
            final_score=current_score,
            matched_rules=[],
            explanations=[],
        )

    score = current_score
    matched_rules = []
    explanations = []
    rejected = False
    override_applied = False

    for rule in ruleset.rules:
        # Skip non-active rules
        if rule.status != RuleStatus.ACTIVE.value:
            continue

        # Check if feature exists
        if rule.field not in features:
            continue  # Skip rule if feature missing

        feature_value = features[rule.field]

        # Evaluate condition
        matches = False
        try:
            if rule.op == ">":
                matches = feature_value > rule.value
            elif rule.op == ">=":
                matches = feature_value >= rule.value
            elif rule.op == "<":
                matches = feature_value < rule.value
            elif rule.op == "<=":
                matches = feature_value <= rule.value
            elif rule.op == "==":
                matches = feature_value == rule.value
            elif rule.op == "in":
                matches = feature_value in rule.value
            elif rule.op == "not_in":
                matches = feature_value not in rule.value
        except (TypeError, ValueError):
            # Type mismatch - rule does not match
            continue

        if not matches:
            continue

        # Rule matched - record it
        matched_rules.append(rule.id)
        explanations.append(
            {
                "rule_id": rule.id,
                "severity": rule.severity,
                "reason": rule.reason or f"rule_matched:{rule.id}",
            }
        )

        # Apply action based on precedence
        if rule.action == "reject":
            rejected = True
            score = 99
            # Continue evaluating to collect all matched rules
        elif rule.action == "override_score" and not override_applied:
            score = rule.score
            override_applied = True
            # Continue evaluating to collect all matched rules
        elif rule.action == "clamp_min" and not override_applied:
            score = max(score, rule.score)
        elif rule.action == "clamp_max" and not override_applied:
            score = min(score, rule.score)

    # Ensure score is in valid range
    score = max(1, min(99, score))

    return RuleResult(
        final_score=score,
        matched_rules=matched_rules,
        explanations=explanations,
        rejected=rejected,
    )
