"""Rule store abstraction for pluggable backends.

This module provides a protocol (interface) for rule storage backends,
allowing the system to use different storage implementations (in-memory,
file-based, PostgreSQL) without changing the API layer.

The default backend is determined by the RULE_STORE_BACKEND environment variable:
- "inmemory" (default): In-memory storage with optional file persistence
- "postgres": PostgreSQL-backed storage (requires database configuration)
"""

import os
from typing import Protocol

from api.rules import Rule


class RuleStore(Protocol):
    """Protocol defining the rule store interface.

    Any rule store backend must implement these methods to be compatible
    with the API layer. This allows swapping storage backends without
    changing endpoint logic.
    """

    def save(self, rule: Rule) -> None:
        """Save a rule to the store.

        For new rules, status must be "draft".
        For existing rules, validates state transitions.

        Args:
            rule: Rule to save.

        Raises:
            ValueError: If the status transition is invalid.
        """
        ...

    def get(self, rule_id: str) -> Rule | None:
        """Get a rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            Rule if found, None otherwise.
        """
        ...

    def list_rules(
        self,
        status: str | None = None,
        include_archived: bool = False,
    ) -> list[Rule]:
        """List rules with optional filters.

        Args:
            status: Filter by status. If None, returns all non-archived rules.
            include_archived: If True, includes archived rules. Default False.

        Returns:
            List of matching Rules, ordered by rule ID.
        """
        ...

    def delete(self, rule_id: str) -> bool:
        """Delete (archive) a rule.

        Only draft rules can be deleted. Deleting a rule archives it
        rather than removing it completely.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if rule was found and archived, False otherwise.
        """
        ...

    def exists(self, rule_id: str) -> bool:
        """Check if a rule exists.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if rule exists, False otherwise.
        """
        ...


def get_rule_store_backend() -> str:
    """Get the configured rule store backend.

    Returns:
        The backend name from RULE_STORE_BACKEND env var, defaulting to "inmemory".
    """
    return os.getenv("RULE_STORE_BACKEND", "inmemory")


def is_postgres_backend_enabled() -> bool:
    """Check if PostgreSQL backend is enabled.

    Returns:
        True if RULE_STORE_BACKEND is set to "postgres".
    """
    return get_rule_store_backend() == "postgres"
