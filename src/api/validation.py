"""Rule validation and conflict detection."""

from dataclasses import dataclass
from typing import Any

from api.rules import Rule, RuleSet


@dataclass
class Conflict:
    """Represents a conflict between two rules."""

    rule1_id: str
    rule2_id: str
    conflict_type: str
    description: str


@dataclass
class Redundancy:
    """Represents a redundant rule."""

    rule_id: str
    redundant_with: str
    redundancy_type: str
    description: str


def _ranges_overlap(op1: str, value1: Any, op2: str, value2: Any) -> bool:
    """Check if two numeric ranges overlap.

    Args:
        op1: First operator.
        value1: First value.
        op2: Second operator.
        value2: Second value.

    Returns:
        True if ranges overlap, False otherwise.
    """

    # Convert to comparable ranges
    def get_range(op: str, val: Any) -> tuple[float, float]:
        """Get numeric range for operator and value."""
        if op == ">":
            return (float(val), float("inf"))
        elif op == ">=":
            return (float(val), float("inf"))
        elif op == "<":
            return (float("-inf"), float(val))
        elif op == "<=":
            return (float("-inf"), float(val))
        elif op == "==":
            return (float(val), float(val))
        else:
            # in, not_in - can't determine range easily
            return None

    range1 = get_range(op1, value1)
    range2 = get_range(op2, value2)

    if range1 is None or range2 is None:
        return False  # Can't determine overlap for list operators

    # Check if ranges overlap
    return not (range1[1] < range2[0] or range2[1] < range1[0])


def _conditions_overlap(rule1: Rule, rule2: Rule) -> bool:
    """Check if two rules have overlapping conditions.

    Args:
        rule1: First rule.
        rule2: Second rule.

    Returns:
        True if conditions overlap, False otherwise.
    """
    # Must be same field
    if rule1.field != rule2.field:
        return False

    # For list operators, check if sets overlap
    if rule1.op in ["in", "not_in"] and rule2.op in ["in", "not_in"]:
        if not isinstance(rule1.value, list) or not isinstance(rule2.value, list):
            return False

        set1 = set(rule1.value)
        set2 = set(rule2.value)

        if rule1.op == "in" and rule2.op == "in":
            # Both "in" - overlap if sets intersect
            return bool(set1 & set2)
        elif rule1.op == "not_in" and rule2.op == "not_in":
            # Both "not_in" - always overlap (both exclude sets)
            return True
        else:
            # One "in", one "not_in" - overlap if value in "in" is not in "not_in"
            if rule1.op == "in":
                return bool(set1 - set2)
            else:
                return bool(set2 - set1)

    # For comparison operators, check numeric range overlap
    if rule1.op in [">", ">=", "<", "<=", "=="] and rule2.op in [
        ">",
        ">=",
        "<",
        "<=",
        "==",
    ]:
        return _ranges_overlap(rule1.op, rule1.value, rule2.op, rule2.value)

    # Mixed operators - assume they might overlap
    if rule1.op == "==":
        # Equality with other operators
        if rule2.op in [">", ">=", "<", "<="]:
            return _ranges_overlap("==", rule1.value, rule2.op, rule2.value)
    elif rule2.op == "==":
        return _ranges_overlap(rule1.op, rule1.value, "==", rule2.value)

    return False


def _actions_conflict(action1: str, action2: str) -> bool:
    """Check if two actions conflict.

    Args:
        action1: First action.
        action2: Second action.

    Returns:
        True if actions conflict, False otherwise.
    """
    # Reject conflicts with everything except itself
    if action1 == "reject" or action2 == "reject":
        return action1 != action2

    # Override conflicts with override (only one can apply)
    if action1 == "override_score" and action2 == "override_score":
        return True

    # Clamp actions don't conflict with each other (they're composable)
    return False


def detect_conflicts(ruleset: RuleSet) -> list[Conflict]:
    """Detect conflicts between rules in a ruleset.

    Args:
        ruleset: RuleSet to analyze.

    Returns:
        List of conflicts found.
    """
    conflicts = []

    for i, rule1 in enumerate(ruleset.rules):
        for rule2 in ruleset.rules[i + 1 :]:
            # Check if conditions overlap
            if not _conditions_overlap(rule1, rule2):
                continue

            # Check if actions conflict
            if _actions_conflict(rule1.action, rule2.action):
                conflict_type = "action_conflict"
                description = (
                    f"Rules '{rule1.id}' and '{rule2.id}' have overlapping conditions "
                    f"on field '{rule1.field}' but conflicting actions: "
                    f"{rule1.action} vs {rule2.action}"
                )
                conflicts.append(
                    Conflict(
                        rule1_id=rule1.id,
                        rule2_id=rule2.id,
                        conflict_type=conflict_type,
                        description=description,
                    )
                )

    return conflicts


def detect_redundancies(ruleset: RuleSet) -> list[Redundancy]:
    """Detect redundant rules in a ruleset.

    A rule is redundant if another rule has a subset of its conditions
    and the same action, or if it's a strict subset of another rule.

    Args:
        ruleset: RuleSet to analyze.

    Returns:
        List of redundancies found.
    """
    redundancies = []

    for i, rule1 in enumerate(ruleset.rules):
        for rule2 in ruleset.rules[i + 1 :]:
            # Must be same field and action to be redundant
            if rule1.field != rule2.field or rule1.action != rule2.action:
                continue

            # Check if one is a subset of the other
            if _is_subset_condition(rule1, rule2):
                # rule1 is subset of rule2, so rule1 is redundant
                redundancies.append(
                    Redundancy(
                        rule_id=rule1.id,
                        redundant_with=rule2.id,
                        redundancy_type="subset",
                        description=(
                            f"Rule '{rule1.id}' is redundant with '{rule2.id}': "
                            f"its condition on '{rule1.field}' is a subset"
                        ),
                    )
                )
            elif _is_subset_condition(rule2, rule1):
                # rule2 is subset of rule1, so rule2 is redundant
                redundancies.append(
                    Redundancy(
                        rule_id=rule2.id,
                        redundant_with=rule1.id,
                        redundancy_type="subset",
                        description=(
                            f"Rule '{rule2.id}' is redundant with '{rule1.id}': "
                            f"its condition on '{rule2.field}' is a subset"
                        ),
                    )
                )

    return redundancies


def _is_subset_condition(rule1: Rule, rule2: Rule) -> bool:
    """Check if rule1's condition is a subset of rule2's condition.

    Args:
        rule1: First rule (potential subset).
        rule2: Second rule (potential superset).

    Returns:
        True if rule1 is a subset of rule2.
    """
    if rule1.field != rule2.field:
        return False

    # For list operators
    if rule1.op in ["in", "not_in"] and rule2.op in ["in", "not_in"]:
        if not isinstance(rule1.value, list) or not isinstance(rule2.value, list):
            return False

        set1 = set(rule1.value)
        set2 = set(rule2.value)

        if rule1.op == "in" and rule2.op == "in":
            # rule1 is subset if its values are all in rule2
            return set1.issubset(set2)
        elif rule1.op == "not_in" and rule2.op == "not_in":
            # rule1 is subset if rule2 excludes more values
            return set2.issubset(set1)
        else:
            return False

    # For comparison operators, check if range1 is subset of range2
    def get_range(op: str, val: Any) -> tuple[float, float] | None:
        """Get numeric range for operator and value."""
        try:
            val_float = float(val)
        except (TypeError, ValueError):
            return None

        if op == ">":
            return (val_float, float("inf"))
        elif op == ">=":
            return (val_float, float("inf"))
        elif op == "<":
            return (float("-inf"), val_float)
        elif op == "<=":
            return (float("-inf"), val_float)
        elif op == "==":
            return (val_float, val_float)
        else:
            return None

    range1 = get_range(rule1.op, rule1.value)
    range2 = get_range(rule2.op, rule2.value)

    if range1 is None or range2 is None:
        return False

    # Check if range1 is subset of range2
    return range1[0] >= range2[0] and range1[1] <= range2[1]


def validate_ruleset(
    ruleset: RuleSet, strict: bool = False
) -> tuple[list[Conflict], list[Redundancy]]:
    """Validate a ruleset and return conflicts and redundancies.

    Args:
        ruleset: RuleSet to validate.
        strict: If True, raise ValueError on conflicts. If False, return them.

    Returns:
        Tuple of (conflicts, redundancies).

    Raises:
        ValueError: If strict=True and conflicts are found.
    """
    conflicts = detect_conflicts(ruleset)
    redundancies = detect_redundancies(ruleset)

    if strict and conflicts:
        conflict_msgs = [c.description for c in conflicts]
        raise ValueError(
            "Rule conflicts detected:\n"
            + "\n".join(f"  - {msg}" for msg in conflict_msgs)
        )

    return conflicts, redundancies
