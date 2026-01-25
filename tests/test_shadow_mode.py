"""Tests for shadow mode rule evaluation."""

from api.rules import Rule, RuleSet, evaluate_rules


class TestShadowModeEvaluation:
    """Tests for shadow mode rule evaluation."""

    def test_shadow_rules_evaluated_separately(self):
        """Test that shadow rules are evaluated but don't affect score."""
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
                id="shadow_rule",
                field="velocity_24h",
                op=">",
                value=3,
                action="clamp_min",
                score=90,
                status="shadow",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        # Active rule should affect score
        assert result.final_score == 80
        assert "active_rule" in result.matched_rules

        # Shadow rule should be recorded but not affect score
        assert "shadow_rule" in result.shadow_matched_rules
        assert len(result.shadow_explanations) == 1
        assert result.shadow_explanations[0]["rule_id"] == "shadow_rule"

    def test_shadow_rules_do_not_affect_score(self):
        """Test that shadow rules never change the final score."""
        # Shadow rule that would set score to 99
        rule = Rule(
            id="shadow_reject",
            field="velocity_24h",
            op=">",
            value=5,
            action="reject",
            status="shadow",
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        # Score should be unchanged (shadow rule doesn't apply)
        assert result.final_score == 50
        assert result.rejected is False
        assert "shadow_reject" in result.shadow_matched_rules

    def test_shadow_override_does_not_apply(self):
        """Test that shadow override rules don't override the score."""
        rules = [
            Rule(
                id="active_clamp",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=70,
                status="active",
            ),
            Rule(
                id="shadow_override",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=3.0,
                action="override_score",
                score=95,
                status="shadow",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        features = {"velocity_24h": 10, "amount_to_avg_ratio_30d": 5.0}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        # Active rule should apply (score = 70)
        assert result.final_score == 70

        # Shadow override should be recorded but not applied
        assert "shadow_override" in result.shadow_matched_rules
        # Score should not be 95 (shadow override didn't apply)

    def test_shadow_rules_with_missing_features(self):
        """Test that shadow rules skip when features are missing."""
        rule = Rule(
            id="shadow_missing",
            field="nonexistent_field",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="shadow",
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        # Shadow rule should not match (feature missing)
        assert len(result.shadow_matched_rules) == 0
        assert result.final_score == 50

    def test_mixed_active_and_shadow_rules(self):
        """Test evaluation with mix of active and shadow rules."""
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
                id="shadow1",
                field="velocity_24h",
                op=">",
                value=3,
                action="clamp_min",
                score=90,
                status="shadow",
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
                id="shadow2",
                field="bank_connections_24h",
                op=">",
                value=4,
                action="reject",
                status="shadow",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        features = {
            "velocity_24h": 10,
            "amount_to_avg_ratio_30d": 5.0,
            "bank_connections_24h": 10,
        }
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        # Active rules should affect score
        assert result.final_score == 80  # Max of 80 and 75
        assert "active1" in result.matched_rules
        assert "active2" in result.matched_rules

        # Shadow rules should be recorded
        assert "shadow1" in result.shadow_matched_rules
        assert "shadow2" in result.shadow_matched_rules
        assert len(result.shadow_matched_rules) == 2

    def test_shadow_rules_empty_when_no_shadow_rules(self):
        """Test that shadow fields are empty when no shadow rules exist."""
        rule = Rule(
            id="active_only",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="active",
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert len(result.shadow_matched_rules) == 0
        assert len(result.shadow_explanations) == 0

    def test_shadow_rules_explanations_format(self):
        """Test that shadow explanations have correct format."""
        rule = Rule(
            id="shadow_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="shadow",
            severity="high",
            reason="Test shadow rule",
        )
        ruleset = RuleSet(version="v1", rules=[rule])

        features = {"velocity_24h": 10}
        result = evaluate_rules(features, current_score=50, ruleset=ruleset)

        assert len(result.shadow_explanations) == 1
        exp = result.shadow_explanations[0]
        assert exp["rule_id"] == "shadow_rule"
        assert exp["severity"] == "high"
        assert exp["reason"] == "Test shadow rule"
