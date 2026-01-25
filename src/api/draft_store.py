"""Draft rule storage and management.

Manages draft rules separately from production ruleset, with
persistent storage and status enforcement.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from api.rules import Rule, RuleStatus

logger = logging.getLogger(__name__)


class DraftRuleStore:
    """Store and manage draft rules with persistent storage."""

    def __init__(self, storage_path: Path | str | None = None):
        """Initialize draft rule store.

        Args:
            storage_path: Path for persistent storage. If None, in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._rules: dict[str, Rule] = {}  # rule_id -> Rule

        # Load existing rules if file exists
        if self.storage_path and self.storage_path.exists():
            self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from storage file."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                self._rules = {}
                for rule_id, rule_dict in data.items():
                    try:
                        rule = Rule(**rule_dict)
                        # Enforce draft-only: only load rules with draft status
                        if rule.status == RuleStatus.DRAFT.value:
                            self._rules[rule_id] = rule
                        else:
                            logger.warning(
                                f"Skipping non-draft rule {rule_id} "
                                f"with status {rule.status}"
                            )
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to load rule {rule_id}: {e}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                f"Failed to load draft rules from {self.storage_path}: {e}"
            )
            self._rules = {}

    def _save_rules(self) -> None:
        """Save rules to storage file."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {rule_id: asdict(rule) for rule_id, rule in self._rules.items()}

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save draft rules to {self.storage_path}: {e}")

    def save(self, rule: Rule) -> None:
        """Save a draft rule or update an existing rule's status.

        Args:
            rule: Rule to save. Must have status="draft" for new rules,
                or can be pending_review if updating existing draft.

        Raises:
            ValueError: If rule status is invalid for this operation.
        """
        # Allow saving draft rules
        if rule.status == RuleStatus.DRAFT.value:
            self._rules[rule.id] = rule
            self._save_rules()
            logger.debug(f"Saved draft rule {rule.id}")
        # Allow updating existing draft to pending_review
        elif rule.status == RuleStatus.PENDING_REVIEW.value and rule.id in self._rules:
            existing = self._rules[rule.id]
            if existing.status == RuleStatus.DRAFT.value:
                self._rules[rule.id] = rule
                self._save_rules()
                logger.debug(f"Updated rule {rule.id} to pending_review")
            else:
                raise ValueError(
                    f"Cannot update rule {rule.id} from {existing.status} "
                    "to pending_review"
                )
        else:
            raise ValueError(
                f"Cannot save rule {rule.id} with status {rule.status}. "
                "Only draft rules can be created, or existing drafts can be "
                "updated to pending_review."
            )

    def get(self, rule_id: str) -> Rule | None:
        """Get a draft rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            Rule if found, None otherwise.
        """
        return self._rules.get(rule_id)

    def list_rules(
        self,
        status: str | None = None,
        include_archived: bool = False,
    ) -> list[Rule]:
        """List draft rules with optional filters.

        Args:
            status: Filter by status. If None, returns all draft rules.
            include_archived: If True, includes archived rules. Default False.

        Returns:
            List of matching Rules, ordered by rule ID.
        """
        rules = list(self._rules.values())

        # Filter by status
        if status is not None:
            rules = [r for r in rules if r.status == status]
        elif not include_archived:
            # Exclude archived by default
            rules = [r for r in rules if r.status != RuleStatus.ARCHIVED.value]

        # Sort by rule ID
        rules.sort(key=lambda r: r.id)

        return rules

    def delete(self, rule_id: str) -> bool:
        """Delete a draft rule (archive it).

        Args:
            rule_id: Rule identifier.

        Returns:
            True if rule was found and archived, False otherwise.
        """
        rule = self._rules.get(rule_id)
        if rule is None:
            return False

        if rule.status != RuleStatus.DRAFT.value:
            logger.warning(
                f"Cannot delete rule {rule_id} with status {rule.status}. "
                "Only draft rules can be deleted."
            )
            return False

        # Archive the rule
        rule_dict = asdict(rule)
        rule_dict["status"] = RuleStatus.ARCHIVED.value
        archived_rule = Rule(**rule_dict)

        self._rules[rule_id] = archived_rule
        self._save_rules()
        logger.debug(f"Archived draft rule {rule_id}")

        return True

    def exists(self, rule_id: str) -> bool:
        """Check if a rule exists.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if rule exists, False otherwise.
        """
        return rule_id in self._rules


# Global draft rule store instance
_global_draft_store: DraftRuleStore | None = None


def get_draft_store() -> DraftRuleStore:
    """Get the global draft rule store instance.

    Returns:
        Global DraftRuleStore instance.
    """
    global _global_draft_store
    if _global_draft_store is None:
        _global_draft_store = DraftRuleStore()
    return _global_draft_store


def set_draft_store(store: DraftRuleStore) -> None:
    """Set the global draft rule store instance (for testing).

    Args:
        store: DraftRuleStore instance to use.
    """
    global _global_draft_store
    _global_draft_store = store
