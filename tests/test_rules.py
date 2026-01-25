"""Tests for rule-based decision engine."""

import json
import tempfile
from pathlib import Path

import pytest

from api.rules import Rule, RuleResult, RuleSet, evaluate_rules


class TestRuleValidation:
    """Tests for Rule validation."""

    def test_valid_rule(self):
        """Test that a valid rule can be created."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        assert rule.id == "test_rule"
        assert rule.field == "velocity_24h"
        assert rule.op == ">"
        assert rule.value == 5
        assert rule.action == "clamp_min"
        assert rule.score == 80

    def test_invalid_operator(self):
        """Test that invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator"):
            Rule(
                id="test",
                field="velocity_24h",
                op="invalid",
                value=5,
                action="clamp_min",
                score=80,
            )

    def test_invalid_action(self):
        """Test that invalid action raises ValueError."""
        with pytest.raises(ValueError, match="Invalid action"):
            Rule(
                id="test",
                field="velocity_24h",
                op=">",
                value=5,
                action="invalid",
                score=80,
            )

    def test_missing_score_for_override(self):
        """Test that override_score requires score field."""
        with pytest.raises(ValueError, match="requires 'score' field"):
            Rule(
                id="test",
                field="velocity_24h",
                op=">",
                value=5,
                action="override_score",
            )

    def test_in_operator_requires_list(self):
        """Test that 'in' operator requires list value."""
        with pytest.raises(ValueError, match="requires 'value' to be a list"):
            Rule(
                id="test",
                field="velocity_24h",
                op="in",
                value=5,  # Should be a list
                action="clamp_min",
                score=80,
            )

    def test_invalid_severity(self):
        """Test that invalid severity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            Rule(
                id="test",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
                severity="invalid",
            )


class TestRuleEvaluation:
    """Tests for rule evaluation logic."""

    def test_single_rule_match_clamp_min(self):
        """Test that clamp_min increases score when rule matches."""
        rule = Rule(
            id="min_score",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 80
        assert "min_score" in result.matched_rules

    def test_single_rule_match_clamp_max(self):
        """Test that clamp_max decreases score when rule matches."""
        rule = Rule(
            id="max_score",
            field="velocity_24h",
            op="<",
            value=2,
            action="clamp_max",
            score=20,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 1}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 20
        assert "max_score" in result.matched_rules

    def test_single_rule_override_score(self):
        """Test that override_score sets score exactly."""
        rule = Rule(
            id="override",
            field="velocity_24h",
            op=">",
            value=10,
            action="override_score",
            score=95,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 15}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 95
        assert "override" in result.matched_rules

    def test_reject_action_sets_score_99(self):
        """Test that reject action sets score to 99."""
        rule = Rule(
            id="reject",
            field="velocity_24h",
            op=">",
            value=20,
            action="reject",
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 25}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 99
        assert result.rejected is True
        assert "reject" in result.matched_rules

    def test_multiple_rules_all_match(self):
        """Test that multiple matching rules are all recorded."""
        rules = [
            Rule(
                id="rule1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=70,
            ),
            Rule(
                id="rule2",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=3.0,
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        features = {"velocity_24h": 10, "amount_to_avg_ratio_30d": 5.0}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 80  # Max of 70 and 80
        assert "rule1" in result.matched_rules
        assert "rule2" in result.matched_rules
        assert len(result.matched_rules) == 2

    def test_precedence_reject_over_override(self):
        """Test that reject takes precedence over override."""
        rules = [
            Rule(
                id="override",
                field="velocity_24h",
                op=">",
                value=5,
                action="override_score",
                score=90,
            ),
            Rule(
                id="reject",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=10.0,
                action="reject",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        features = {"velocity_24h": 10, "amount_to_avg_ratio_30d": 15.0}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 99
        assert result.rejected is True
        assert "override" in result.matched_rules
        assert "reject" in result.matched_rules

    def test_precedence_override_over_clamp(self):
        """Test that override takes precedence over clamp."""
        rules = [
            Rule(
                id="clamp",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=70,
            ),
            Rule(
                id="override",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=3.0,
                action="override_score",
                score=85,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        features = {"velocity_24h": 10, "amount_to_avg_ratio_30d": 5.0}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 85  # Override wins
        assert "clamp" in result.matched_rules
        assert "override" in result.matched_rules

    def test_missing_feature_skips_rule(self):
        """Test that missing feature causes rule to be skipped."""
        rule = Rule(
            id="missing",
            field="nonexistent_field",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 50  # Unchanged
        assert len(result.matched_rules) == 0

    def test_op_greater_than(self):
        """Test greater than operator."""
        rule = Rule(
            id="gt",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        # Should match
        result = evaluate_rules({"velocity_24h": 10}, current_score=50, ruleset=ruleset)
        assert result.final_score == 80

        # Should not match
        result = evaluate_rules({"velocity_24h": 3}, current_score=50, ruleset=ruleset)
        assert result.final_score == 50

    def test_op_less_than(self):
        """Test less than operator."""
        rule = Rule(
            id="lt",
            field="velocity_24h",
            op="<",
            value=5,
            action="clamp_max",
            score=20,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        # Should match
        result = evaluate_rules({"velocity_24h": 3}, current_score=50, ruleset=ruleset)
        assert result.final_score == 20

        # Should not match
        result = evaluate_rules({"velocity_24h": 10}, current_score=50, ruleset=ruleset)
        assert result.final_score == 50

    def test_op_equals(self):
        """Test equals operator."""
        rule = Rule(
            id="eq",
            field="velocity_24h",
            op="==",
            value=5,
            action="override_score",
            score=75,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        # Should match
        result = evaluate_rules({"velocity_24h": 5}, current_score=50, ruleset=ruleset)
        assert result.final_score == 75

        # Should not match
        result = evaluate_rules({"velocity_24h": 10}, current_score=50, ruleset=ruleset)
        assert result.final_score == 50

    def test_op_in_list(self):
        """Test 'in' operator with list."""
        rule = Rule(
            id="in_list",
            field="velocity_24h",
            op="in",
            value=[5, 10, 15],
            action="clamp_min",
            score=80,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        # Should match
        result = evaluate_rules({"velocity_24h": 10}, current_score=50, ruleset=ruleset)
        assert result.final_score == 80

        # Should not match
        result = evaluate_rules({"velocity_24h": 7}, current_score=50, ruleset=ruleset)
        assert result.final_score == 50

    def test_op_not_in_list(self):
        """Test 'not_in' operator with list."""
        rule = Rule(
            id="not_in_list",
            field="velocity_24h",
            op="not_in",
            value=[1, 2, 3],
            action="clamp_min",
            score=80,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        # Should match (value not in list)
        result = evaluate_rules({"velocity_24h": 10}, current_score=50, ruleset=ruleset)
        assert result.final_score == 80

        # Should not match (value in list)
        result = evaluate_rules({"velocity_24h": 2}, current_score=50, ruleset=ruleset)
        assert result.final_score == 50

    def test_empty_ruleset_returns_original_score(self):
        """Test that empty ruleset returns original score unchanged."""
        ruleset = RuleSet(version="v1", rules=[])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 50
        assert len(result.matched_rules) == 0
        assert len(result.explanations) == 0

    def test_score_clamped_to_valid_range(self):
        """Test that final score is clamped to 1-99 range."""
        rule = Rule(
            id="high_score",
            field="velocity_24h",
            op=">",
            value=5,
            action="override_score",
            score=150,  # Invalid, should be clamped
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 99  # Clamped

    def test_type_mismatch_skips_rule(self):
        """Test that type mismatch in comparison skips rule."""
        rule = Rule(
            id="type_mismatch",
            field="velocity_24h",
            op=">",
            value="invalid",  # String vs int
            action="clamp_min",
            score=80,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert result.final_score == 50  # Unchanged, rule skipped
        assert len(result.matched_rules) == 0


class TestRuleSetLoading:
    """Tests for RuleSet loading from JSON."""

    def test_load_from_valid_json(self):
        """Test loading RuleSet from valid JSON dict."""
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

        assert ruleset.version == "v1"
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].id == "test_rule"

    def test_load_from_invalid_json_raises(self):
        """Test that invalid JSON structure raises ValueError."""
        data = {"version": "v1"}  # Missing 'rules'

        with pytest.raises(ValueError, match="must have 'rules' field"):
            RuleSet.from_dict(data)

    def test_load_from_file(self):
        """Test loading RuleSet from JSON file."""
        data = {
            "version": "v2",
            "rules": [
                {
                    "id": "file_rule",
                    "field": "velocity_24h",
                    "op": ">",
                    "value": 10,
                    "action": "clamp_min",
                    "score": 85,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            ruleset = RuleSet.load_from_file(temp_path)
            assert ruleset.version == "v2"
            assert len(ruleset.rules) == 1
            assert ruleset.rules[0].id == "file_rule"
        finally:
            Path(temp_path).unlink()

    def test_load_from_nonexistent_file_raises(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            RuleSet.load_from_file("/nonexistent/path/rules.json")

    def test_empty_ruleset(self):
        """Test creating empty RuleSet."""
        ruleset = RuleSet.empty(version="v1")

        assert ruleset.version == "v1"
        assert len(ruleset.rules) == 0

    def test_missing_required_fields_in_rule_raises(self):
        """Test that missing required fields in rule raises ValueError."""
        data = {
            "version": "v1",
            "rules": [
                {
                    "id": "incomplete",
                    "field": "velocity_24h",
                    # Missing op, value, action
                }
            ],
        }

        with pytest.raises(ValueError, match="Invalid rule"):
            RuleSet.from_dict(data)


class TestRuleExplanations:
    """Tests for rule explanations in results."""

    def test_explanations_include_rule_metadata(self):
        """Test that explanations include rule ID, severity, and reason."""
        rule = Rule(
            id="explain_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            severity="high",
            reason="High transaction velocity detected",
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert len(result.explanations) == 1
        exp = result.explanations[0]
        assert exp["rule_id"] == "explain_rule"
        assert exp["severity"] == "high"
        assert exp["reason"] == "High transaction velocity detected"

    def test_explanations_default_reason(self):
        """Test that explanations use default reason if not provided."""
        rule = Rule(
            id="no_reason",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert len(result.explanations) == 1
        exp = result.explanations[0]
        assert exp["reason"] == "rule_matched:no_reason"
