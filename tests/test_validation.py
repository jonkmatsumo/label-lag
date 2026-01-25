"""Tests for rule validation and conflict detection."""

import pytest

from api.rules import Rule, RuleSet
from api.validation import (
    Conflict,
    Redundancy,
    detect_conflicts,
    detect_redundancies,
    validate_ruleset,
)


class TestConflictDetection:
    """Tests for conflict detection."""

    def test_no_conflicts(self):
        """Test that rules with non-overlapping conditions have no conflicts."""
        rules = [
            Rule(
                id="rule1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="rule2",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=3.0,
                action="clamp_min",
                score=75,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts = detect_conflicts(ruleset)
        assert len(conflicts) == 0

    def test_conflict_overlapping_conditions_different_actions(self):
        """Test that overlapping conditions with different actions create conflicts."""
        rules = [
            Rule(
                id="rule1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="rule2",
                field="velocity_24h",
                op=">=",
                value=3,
                action="reject",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts = detect_conflicts(ruleset)
        assert len(conflicts) == 1
        assert conflicts[0].rule1_id == "rule1"
        assert conflicts[0].rule2_id == "rule2"
        assert conflicts[0].conflict_type == "action_conflict"
        assert "velocity_24h" in conflicts[0].description

    def test_conflict_reject_vs_override(self):
        """Test that reject conflicts with override."""
        rules = [
            Rule(
                id="reject_rule",
                field="velocity_24h",
                op=">",
                value=10,
                action="reject",
            ),
            Rule(
                id="override_rule",
                field="velocity_24h",
                op=">=",
                value=8,
                action="override_score",
                score=90,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts = detect_conflicts(ruleset)
        assert len(conflicts) == 1
        assert "reject" in conflicts[0].description.lower()
        assert "override" in conflicts[0].description.lower()

    def test_conflict_multiple_overrides(self):
        """Test that multiple override actions on same field conflict."""
        rules = [
            Rule(
                id="override1",
                field="velocity_24h",
                op=">",
                value=5,
                action="override_score",
                score=80,
            ),
            Rule(
                id="override2",
                field="velocity_24h",
                op=">=",
                value=3,
                action="override_score",
                score=90,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts = detect_conflicts(ruleset)
        assert len(conflicts) == 1

    def test_no_conflict_clamp_actions(self):
        """Test that clamp actions don't conflict with each other."""
        rules = [
            Rule(
                id="clamp1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="clamp2",
                field="velocity_24h",
                op=">",
                value=10,
                action="clamp_min",
                score=90,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts = detect_conflicts(ruleset)
        # Clamp actions are composable, so no conflict
        assert len(conflicts) == 0

    def test_conflict_equality_overlap(self):
        """Test that equality conditions can overlap with ranges."""
        rules = [
            Rule(
                id="eq_rule",
                field="velocity_24h",
                op="==",
                value=5,
                action="override_score",
                score=80,
            ),
            Rule(
                id="range_rule",
                field="velocity_24h",
                op=">=",
                value=3,
                action="reject",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts = detect_conflicts(ruleset)
        assert len(conflicts) == 1

    def test_no_conflict_non_overlapping_ranges(self):
        """Test that non-overlapping ranges don't conflict."""
        rules = [
            Rule(
                id="low",
                field="velocity_24h",
                op="<",
                value=5,
                action="clamp_max",
                score=20,
            ),
            Rule(
                id="high",
                field="velocity_24h",
                op=">",
                value=10,
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts = detect_conflicts(ruleset)
        assert len(conflicts) == 0

    def test_conflict_list_operators(self):
        """Test conflict detection with list operators."""
        rules = [
            Rule(
                id="in_rule",
                field="velocity_24h",
                op="in",
                value=[5, 10, 15],
                action="override_score",
                score=80,
            ),
            Rule(
                id="reject_rule",
                field="velocity_24h",
                op="in",
                value=[10, 20],
                action="reject",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts = detect_conflicts(ruleset)
        # Should conflict because sets overlap (10 is in both)
        assert len(conflicts) == 1


class TestRedundancyDetection:
    """Tests for redundancy detection."""

    def test_no_redundancies(self):
        """Test that non-redundant rules are not flagged."""
        rules = [
            Rule(
                id="rule1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="rule2",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=3.0,
                action="clamp_min",
                score=75,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        redundancies = detect_redundancies(ruleset)
        assert len(redundancies) == 0

    def test_redundancy_subset_range(self):
        """Test that subset ranges are detected as redundant."""
        rules = [
            Rule(
                id="broad",
                field="velocity_24h",
                op=">",
                value=3,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="narrow",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        redundancies = detect_redundancies(ruleset)
        assert len(redundancies) == 1
        assert redundancies[0].rule_id == "narrow"
        assert redundancies[0].redundant_with == "broad"

    def test_redundancy_equality_in_range(self):
        """Test that equality in a range is redundant."""
        rules = [
            Rule(
                id="range",
                field="velocity_24h",
                op=">=",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="equality",
                field="velocity_24h",
                op="==",
                value=10,
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        redundancies = detect_redundancies(ruleset)
        assert len(redundancies) == 1
        assert redundancies[0].rule_id == "equality"

    def test_redundancy_list_subset(self):
        """Test that subset lists are detected as redundant."""
        rules = [
            Rule(
                id="broad_list",
                field="velocity_24h",
                op="in",
                value=[5, 10, 15, 20],
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="narrow_list",
                field="velocity_24h",
                op="in",
                value=[5, 10],
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        redundancies = detect_redundancies(ruleset)
        assert len(redundancies) == 1
        assert redundancies[0].rule_id == "narrow_list"

    def test_no_redundancy_different_actions(self):
        """Test that same conditions with different actions are not redundant."""
        rules = [
            Rule(
                id="clamp",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="override",
                field="velocity_24h",
                op=">",
                value=5,
                action="override_score",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        redundancies = detect_redundancies(ruleset)
        assert len(redundancies) == 0

    def test_no_redundancy_different_fields(self):
        """Test that same conditions on different fields are not redundant."""
        rules = [
            Rule(
                id="field1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="field2",
                field="amount_to_avg_ratio_30d",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        redundancies = detect_redundancies(ruleset)
        assert len(redundancies) == 0


class TestValidateRuleset:
    """Tests for validate_ruleset function."""

    def test_validate_no_issues(self):
        """Test validation of ruleset with no issues."""
        rules = [
            Rule(
                id="rule1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts, redundancies = validate_ruleset(ruleset)
        assert len(conflicts) == 0
        assert len(redundancies) == 0

    def test_validate_with_conflicts(self):
        """Test validation returns conflicts without raising."""
        rules = [
            Rule(
                id="rule1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="rule2",
                field="velocity_24h",
                op=">=",
                value=3,
                action="reject",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts, redundancies = validate_ruleset(ruleset, strict=False)
        assert len(conflicts) > 0

    def test_validate_strict_mode_raises(self):
        """Test that strict mode raises ValueError on conflicts."""
        rules = [
            Rule(
                id="rule1",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="rule2",
                field="velocity_24h",
                op=">=",
                value=3,
                action="reject",
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        with pytest.raises(ValueError, match="Rule conflicts detected"):
            validate_ruleset(ruleset, strict=True)

    def test_validate_with_redundancies(self):
        """Test validation returns redundancies."""
        rules = [
            Rule(
                id="broad",
                field="velocity_24h",
                op=">",
                value=3,
                action="clamp_min",
                score=80,
            ),
            Rule(
                id="narrow",
                field="velocity_24h",
                op=">",
                value=5,
                action="clamp_min",
                score=80,
            ),
        ]
        ruleset = RuleSet(version="v1", rules=rules)

        conflicts, redundancies = validate_ruleset(ruleset)
        assert len(redundancies) > 0
