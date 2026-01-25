"""Test harness for deterministic rule testing with golden files."""

import json
from pathlib import Path
from typing import Any

import pytest

from api.rules import Rule, RuleSet, evaluate_rules


class RuleTestCase:
    """Helper class for constructing rule test scenarios."""

    def __init__(
        self,
        name: str,
        features: dict[str, Any],
        rules: list[Rule] | RuleSet,
        current_score: int,
        expected_score: int | None = None,
        expected_matched_rules: list[str] | None = None,
        expected_rejected: bool | None = None,
    ):
        """Initialize a test case.

        Args:
            name: Test case name.
            features: Feature dictionary.
            rules: List of rules or RuleSet.
            current_score: Score before rule application.
            expected_score: Expected final score (if None, uses current_score).
            expected_matched_rules: Expected matched rule IDs.
            expected_rejected: Expected rejected flag.
        """
        self.name = name
        self.features = features
        if isinstance(rules, RuleSet):
            self.ruleset = rules
        else:
            self.ruleset = RuleSet(version="v1", rules=rules)
        self.current_score = current_score
        self.expected_score = (
            expected_score if expected_score is not None else current_score
        )
        self.expected_matched_rules = expected_matched_rules or []
        self.expected_rejected = (
            expected_rejected if expected_rejected is not None else False
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert test case to dictionary for golden file storage."""
        return {
            "name": self.name,
            "features": self.features,
            "rules": [
                {
                    "id": r.id,
                    "field": r.field,
                    "op": r.op,
                    "value": r.value,
                    "action": r.action,
                    "score": r.score,
                    "severity": r.severity,
                    "reason": r.reason,
                }
                for r in self.ruleset.rules
            ],
            "current_score": self.current_score,
            "expected_score": self.expected_score,
            "expected_matched_rules": self.expected_matched_rules,
            "expected_rejected": self.expected_rejected,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleTestCase":
        """Create test case from dictionary."""
        rules = [Rule(**rule_dict) for rule_dict in data["rules"]]
        return cls(
            name=data["name"],
            features=data["features"],
            rules=rules,
            current_score=data["current_score"],
            expected_score=data.get("expected_score"),
            expected_matched_rules=data.get("expected_matched_rules"),
            expected_rejected=data.get("expected_rejected"),
        )

    def run(self) -> dict[str, Any]:
        """Run the test case and return results."""
        result = evaluate_rules(self.features, self.current_score, self.ruleset)
        return {
            "final_score": result.final_score,
            "matched_rules": result.matched_rules,
            "rejected": result.rejected,
            "explanations": result.explanations,
        }

    def assert_result(self, result: dict[str, Any]) -> None:
        """Assert that result matches expected values."""
        assert result["final_score"] == self.expected_score, (
            f"Score mismatch: expected {self.expected_score}, "
            f"got {result['final_score']}"
        )
        assert set(result["matched_rules"]) == set(self.expected_matched_rules), (
            f"Matched rules mismatch: expected {self.expected_matched_rules}, "
            f"got {result['matched_rules']}"
        )
        assert result["rejected"] == self.expected_rejected, (
            f"Rejected flag mismatch: expected {self.expected_rejected}, "
            f"got {result['rejected']}"
        )


class RuleTestHarness:
    """Test harness for rule evaluation with golden file support."""

    def __init__(self, golden_dir: Path | None = None):
        """Initialize test harness.

        Args:
            golden_dir: Directory for golden files. If None, uses tests/fixtures/golden.
        """
        if golden_dir is None:
            golden_dir = Path(__file__).parent / "fixtures" / "golden"
        self.golden_dir = Path(golden_dir)
        self.golden_dir.mkdir(parents=True, exist_ok=True)

    def load_golden(self, test_name: str) -> dict[str, Any] | None:
        """Load golden file for a test.

        Args:
            test_name: Name of the test.

        Returns:
            Golden file contents or None if not found.
        """
        golden_file = self.golden_dir / f"{test_name}.json"
        if not golden_file.exists():
            return None
        with open(golden_file) as f:
            return json.load(f)

    def save_golden(self, test_name: str, data: dict[str, Any]) -> None:
        """Save golden file for a test.

        Args:
            test_name: Name of the test.
            data: Data to save.
        """
        golden_file = self.golden_dir / f"{test_name}.json"
        with open(golden_file, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def run_test_case(
        self, test_case: RuleTestCase, update_golden: bool = False
    ) -> dict[str, Any]:
        """Run a test case with optional golden file comparison.

        Args:
            test_case: Test case to run.
            update_golden: If True, update golden file instead of comparing.

        Returns:
            Test results.
        """
        result = test_case.run()
        golden_data = self.load_golden(test_case.name)

        if update_golden or golden_data is None:
            # Save or update golden file
            golden_data = {
                "test_case": test_case.to_dict(),
                "result": result,
            }
            self.save_golden(test_case.name, golden_data)
            return result

        # Compare with golden file
        expected_result = golden_data["result"]
        assert result["final_score"] == expected_result["final_score"], (
            f"Score mismatch: expected {expected_result['final_score']}, "
            f"got {result['final_score']}"
        )
        assert set(result["matched_rules"]) == set(expected_result["matched_rules"]), (
            f"Matched rules mismatch: expected {expected_result['matched_rules']}, "
            f"got {result['matched_rules']}"
        )
        assert result["rejected"] == expected_result["rejected"], (
            f"Rejected flag mismatch: expected {expected_result['rejected']}, "
            f"got {result['rejected']}"
        )
        return result


class TestRuleHarness:
    """Tests for the rule test harness itself."""

    def test_harness_basic_usage(self, tmp_path: Path):
        """Test basic harness usage."""
        harness = RuleTestHarness(golden_dir=tmp_path / "golden")
        test_case = RuleTestCase(
            name="basic_test",
            features={"velocity_24h": 10},
            rules=[
                Rule(
                    id="test_rule",
                    field="velocity_24h",
                    op=">",
                    value=5,
                    action="clamp_min",
                    score=80,
                )
            ],
            current_score=50,
            expected_score=80,
            expected_matched_rules=["test_rule"],
        )

        # First run - creates golden file
        result = harness.run_test_case(test_case, update_golden=True)
        assert result["final_score"] == 80

        # Second run - compares with golden file
        result2 = harness.run_test_case(test_case, update_golden=False)
        assert result2["final_score"] == 80

    def test_test_case_serialization(self):
        """Test that test cases can be serialized and deserialized."""
        test_case = RuleTestCase(
            name="serialization_test",
            features={"velocity_24h": 10},
            rules=[
                Rule(
                    id="test_rule",
                    field="velocity_24h",
                    op=">",
                    value=5,
                    action="clamp_min",
                    score=80,
                )
            ],
            current_score=50,
        )

        data = test_case.to_dict()
        restored = RuleTestCase.from_dict(data)

        assert restored.name == test_case.name
        assert restored.features == test_case.features
        assert restored.current_score == test_case.current_score
        assert len(restored.ruleset.rules) == 1
        assert restored.ruleset.rules[0].id == "test_rule"


class TestDeterministicRuleEvaluation:
    """Deterministic tests for rule evaluation using test harness."""

    @pytest.fixture
    def harness(self, tmp_path: Path) -> RuleTestHarness:
        """Create test harness."""
        return RuleTestHarness(golden_dir=tmp_path / "golden")

    @pytest.mark.parametrize(
        "operator,value,feature_value,should_match",
        [
            (">", 5, 10, True),
            (">", 5, 5, False),
            (">", 5, 3, False),
            (">=", 5, 5, True),
            (">=", 5, 10, True),
            (">=", 5, 3, False),
            ("<", 5, 3, True),
            ("<", 5, 5, False),
            ("<", 5, 10, False),
            ("<=", 5, 5, True),
            ("<=", 5, 3, True),
            ("<=", 5, 10, False),
            ("==", 5, 5, True),
            ("==", 5, 10, False),
        ],
    )
    def test_comparison_operators(
        self,
        harness: RuleTestHarness,
        operator: str,
        value: int,
        feature_value: int,
        should_match: bool,
    ):
        """Test all comparison operators with boundary conditions."""
        test_case = RuleTestCase(
            name=f"operator_{operator}_value_{value}_feature_{feature_value}",
            features={"velocity_24h": feature_value},
            rules=[
                Rule(
                    id="test_rule",
                    field="velocity_24h",
                    op=operator,
                    value=value,
                    action="clamp_min",
                    score=80,
                )
            ],
            current_score=50,
            expected_score=80 if should_match else 50,
            expected_matched_rules=["test_rule"] if should_match else [],
        )

        result = harness.run_test_case(test_case, update_golden=True)
        test_case.assert_result(result)

    @pytest.mark.parametrize(
        "action,current_score,rule_score,expected_score",
        [
            ("clamp_min", 50, 80, 80),
            ("clamp_min", 90, 80, 90),  # Current higher, no change
            ("clamp_max", 50, 20, 20),
            ("clamp_max", 10, 20, 10),  # Current lower, no change
            ("override_score", 50, 85, 85),
            ("reject", 50, None, 99),
        ],
    )
    def test_all_actions(
        self,
        harness: RuleTestHarness,
        action: str,
        current_score: int,
        rule_score: int | None,
        expected_score: int,
    ):
        """Test all action types."""
        rule_kwargs = {
            "id": "test_rule",
            "field": "velocity_24h",
            "op": ">",
            "value": 5,
            "action": action,
        }
        if rule_score is not None:
            rule_kwargs["score"] = rule_score

        test_case = RuleTestCase(
            name=f"action_{action}_current_{current_score}_rule_{rule_score}",
            features={"velocity_24h": 10},
            rules=[Rule(**rule_kwargs)],
            current_score=current_score,
            expected_score=expected_score,
            expected_matched_rules=["test_rule"],
            expected_rejected=(action == "reject"),
        )

        result = harness.run_test_case(test_case, update_golden=True)
        test_case.assert_result(result)

    def test_precedence_reject_over_override(self, harness: RuleTestHarness):
        """Test that reject takes precedence over override."""
        test_case = RuleTestCase(
            name="precedence_reject_over_override",
            features={"velocity_24h": 10, "amount_to_avg_ratio_30d": 15.0},
            rules=[
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
            ],
            current_score=50,
            expected_score=99,
            expected_matched_rules=["override", "reject"],
            expected_rejected=True,
        )

        result = harness.run_test_case(test_case, update_golden=True)
        test_case.assert_result(result)

    def test_precedence_override_over_clamp(self, harness: RuleTestHarness):
        """Test that override takes precedence over clamp."""
        test_case = RuleTestCase(
            name="precedence_override_over_clamp",
            features={"velocity_24h": 10, "amount_to_avg_ratio_30d": 5.0},
            rules=[
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
            ],
            current_score=50,
            expected_score=85,
            expected_matched_rules=["clamp", "override"],
        )

        result = harness.run_test_case(test_case, update_golden=True)
        test_case.assert_result(result)

    def test_list_operators(self, harness: RuleTestHarness):
        """Test 'in' and 'not_in' operators."""
        test_cases = [
            RuleTestCase(
                name="operator_in_match",
                features={"velocity_24h": 10},
                rules=[
                    Rule(
                        id="in_rule",
                        field="velocity_24h",
                        op="in",
                        value=[5, 10, 15],
                        action="clamp_min",
                        score=80,
                    )
                ],
                current_score=50,
                expected_score=80,
                expected_matched_rules=["in_rule"],
            ),
            RuleTestCase(
                name="operator_in_no_match",
                features={"velocity_24h": 7},
                rules=[
                    Rule(
                        id="in_rule",
                        field="velocity_24h",
                        op="in",
                        value=[5, 10, 15],
                        action="clamp_min",
                        score=80,
                    )
                ],
                current_score=50,
                expected_score=50,
                expected_matched_rules=[],
            ),
            RuleTestCase(
                name="operator_not_in_match",
                features={"velocity_24h": 7},
                rules=[
                    Rule(
                        id="not_in_rule",
                        field="velocity_24h",
                        op="not_in",
                        value=[5, 10, 15],
                        action="clamp_min",
                        score=80,
                    )
                ],
                current_score=50,
                expected_score=80,
                expected_matched_rules=["not_in_rule"],
            ),
        ]

        for test_case in test_cases:
            result = harness.run_test_case(test_case, update_golden=True)
            test_case.assert_result(result)
