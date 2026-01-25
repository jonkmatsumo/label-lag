"""Tests for rule lifecycle states."""

import json
import tempfile
from pathlib import Path

import pytest

from api.rules import Rule, RuleSet, RuleStatus, evaluate_rules


class TestRuleStatus:
    """Tests for rule status field."""

    def test_default_status_is_active(self):
        """Test that default status is active."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        assert rule.status == "active"

    def test_status_can_be_set(self):
        """Test that status can be explicitly set."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="draft",
        )
        assert rule.status == "draft"

    def test_invalid_status_raises(self):
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            Rule(
                id="test_rule",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
                status="invalid_status",
            )

    def test_all_valid_statuses(self):
        """Test that all valid statuses can be set."""
        for status in RuleStatus:
            rule = Rule(
                id="test_rule",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
                status=status.value,
            )
            assert rule.status == status.value


class TestRuleStatusBackwardCompatibility:
    """Tests for backward compatibility with rules without status."""

    def test_ruleset_from_dict_without_status(self):
        """Test that rules without status default to active."""
        data = {
            "version": "v1",
            "rules": [
                {
                    "id": "test_rule",
                    "field": "velocity_24h",
                    "op": ">",
                    "value": 5,
                    "action": "clamp_min",
                    "score": 80,
                }
            ],
        }

        ruleset = RuleSet.from_dict(data)
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].status == "active"

    def test_ruleset_from_dict_with_status(self):
        """Test that rules with status use provided status."""
        data = {
            "version": "v1",
            "rules": [
                {
                    "id": "test_rule",
                    "field": "velocity_24h",
                    "op": ">",
                    "value": 5,
                    "action": "clamp_min",
                    "score": 80,
                    "status": "draft",
                }
            ],
        }

        ruleset = RuleSet.from_dict(data)
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].status == "draft"

    def test_load_from_file_backward_compatible(self):
        """Test that loading from file handles missing status."""
        data = {
            "version": "v1",
            "rules": [
                {
                    "id": "test_rule",
                    "field": "velocity_24h",
                    "op": ">",
                    "value": 5,
                    "action": "clamp_min",
                    "score": 80,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            ruleset = RuleSet.load_from_file(temp_path)
            assert ruleset.rules[0].status == "active"
        finally:
            Path(temp_path).unlink()


class TestRuleStatusEvaluation:
    """Tests for rule evaluation with different statuses."""

    def test_only_active_rules_evaluated(self):
        """Test that only active rules are evaluated."""
        rules = [
            Rule(
                id="active_rule",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
                status="active",
            ),
            Rule(
                id="draft_rule",
                field="velocity_24h",
                op=">",
                value=3,
                action="clamp_min",
                score=70,
                status="draft",
            ),
            Rule(
                id="disabled_rule",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=3.0,
                action="clamp_min",
                score=75,
                status="disabled",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        features = {"velocity_24h": 10, "amount_to_avg_ratio_30d": 5.0}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        # Only active_rule should match
        assert result.final_score == 80
        assert "active_rule" in result.matched_rules
        assert "draft_rule" not in result.matched_rules
        assert "disabled_rule" not in result.matched_rules
        assert len(result.matched_rules) == 1

    def test_pending_review_rules_not_evaluated(self):
        """Test that pending_review rules are not evaluated."""
        rule = Rule(
            id="pending_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="pending_review",
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 50  # Unchanged
        assert len(result.matched_rules) == 0

    def test_archived_rules_not_evaluated(self):
        """Test that archived rules are not evaluated."""
        rule = Rule(
            id="archived_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="archived",
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 50  # Unchanged
        assert len(result.matched_rules) == 0

    def test_mixed_status_rules(self):
        """Test evaluation with mix of active and non-active rules."""
        rules = [
            Rule(
                id="active1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
                status="active",
            ),
            Rule(
                id="draft1",
                field="velocity_24h",
                op=">",
                value=10,
                action="clamp_min",
                score=90,
                status="draft",
            ),
            Rule(
                id="active2",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=3.0,
                action="clamp_min",
                score=75,
                status="active",
            ),
            Rule(
                id="disabled1",
                field="bank_connections_24h",
                op=">",
                value=4,
                action="reject",
                status="disabled",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        features = {
            "velocity_24h": 15,
            "amount_to_avg_ratio_30d": 5.0,
            "bank_connections_24h": 10,
        }
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        # active1 and active2 should match, draft1 and disabled1 should not
        assert result.final_score == 80  # Max of 80 and 75
        assert "active1" in result.matched_rules
        assert "active2" in result.matched_rules
        assert "draft1" not in result.matched_rules
        assert "disabled1" not in result.matched_rules
        assert len(result.matched_rules) == 2
