"""Integration tests for PostgresRuleStore.

These tests require a PostgreSQL database and are skipped unless:
1. RULE_STORE_BACKEND=postgres
2. Database is accessible

Run with:
  RULE_STORE_BACKEND=postgres pytest tests/integration/test_postgres_rule_store.py
"""

import pytest

from api.rule_store import is_postgres_backend_enabled
from api.rules import Rule, RuleStatus

# Skip entire module unless postgres backend is enabled
pytestmark = pytest.mark.skipif(
    not is_postgres_backend_enabled(),
    reason="PostgreSQL backend not enabled (set RULE_STORE_BACKEND=postgres)",
)


@pytest.fixture(scope="module")
def check_database():
    """Check that database is accessible before running tests."""
    try:
        from api.postgres_rule_store import PostgresRuleStore

        store = PostgresRuleStore()
        # Try a simple operation to verify connectivity
        store.list_rules()
    except Exception as e:
        pytest.skip(f"Database not accessible: {e}")


@pytest.fixture
def store(check_database):
    """Create a PostgresRuleStore for testing."""
    from api.postgres_rule_store import PostgresRuleStore

    store = PostgresRuleStore()
    yield store
    # Cleanup: remove test rules
    _cleanup_test_rules(store)


def _cleanup_test_rules(store):
    """Remove rules created during testing."""
    from api.postgres_rule_store import RuleDB

    with store._get_session() as session:
        # Delete rules with test prefix
        session.query(RuleDB).filter(RuleDB.rule_id.like("test_%")).delete(
            synchronize_session=False
        )


class TestPostgresRuleStoreBasicOperations:
    """Basic CRUD operations for PostgresRuleStore."""

    def test_save_and_get_rule(self, store):
        """Test saving and retrieving a rule."""
        rule = Rule(
            id="test_pg_001",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            severity="medium",
            status=RuleStatus.DRAFT.value,
        )

        store.save(rule)
        retrieved = store.get("test_pg_001")

        assert retrieved is not None
        assert retrieved.id == "test_pg_001"
        assert retrieved.field == "velocity_24h"
        assert retrieved.value == 5
        assert retrieved.status == RuleStatus.DRAFT.value

    def test_list_rules(self, store):
        """Test listing rules."""
        # Create test rules
        for i in range(3):
            rule = Rule(
                id=f"test_list_{i}",
                field="velocity_24h",
                op=">",
                value=5 + i,
                action="clamp_min",
                score=70,
                status=RuleStatus.DRAFT.value,
            )
            store.save(rule)

        rules = store.list_rules()
        test_rules = [r for r in rules if r.id.startswith("test_list_")]

        assert len(test_rules) >= 3

    def test_list_rules_by_status(self, store):
        """Test listing rules filtered by status."""
        rule = Rule(
            id="test_status_filter",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        draft_rules = store.list_rules(status=RuleStatus.DRAFT.value)
        test_drafts = [r for r in draft_rules if r.id.startswith("test_")]

        assert len(test_drafts) >= 1
        assert all(r.status == RuleStatus.DRAFT.value for r in test_drafts)

    def test_delete_archives_rule(self, store):
        """Test that delete archives the rule."""
        rule = Rule(
            id="test_delete_pg",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        result = store.delete("test_delete_pg")
        assert result is True

        archived = store.get("test_delete_pg")
        assert archived is not None
        assert archived.status == RuleStatus.ARCHIVED.value

    def test_exists(self, store):
        """Test checking if a rule exists."""
        assert store.exists("nonexistent_rule") is False

        rule = Rule(
            id="test_exists_pg",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        assert store.exists("test_exists_pg") is True


class TestPostgresRuleStoreStatusTransitions:
    """Test rule status transitions in PostgresRuleStore."""

    def test_valid_status_transition(self, store):
        """Test valid status transition from draft to pending_review."""
        rule = Rule(
            id="test_transition",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        # Update to pending_review
        rule.status = RuleStatus.PENDING_REVIEW.value
        store.save(rule)

        retrieved = store.get("test_transition")
        assert retrieved.status == RuleStatus.PENDING_REVIEW.value

    def test_invalid_status_transition_raises(self, store):
        """Test that invalid status transition raises ValueError."""
        rule = Rule(
            id="test_invalid_transition",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        # Try to skip to active (invalid)
        rule.status = RuleStatus.ACTIVE.value
        with pytest.raises(ValueError, match="Cannot update rule"):
            store.save(rule)

    def test_cannot_save_non_draft_new_rule(self, store):
        """Test that new rules must have draft status."""
        rule = Rule(
            id="test_non_draft_new",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.ACTIVE.value,
        )

        with pytest.raises(ValueError, match="Only draft rules can be created"):
            store.save(rule)


class TestPostgresRuleStoreComplexValues:
    """Test handling of complex values in PostgresRuleStore."""

    def test_list_value(self, store):
        """Test rule with list value (for 'in' operator)."""
        rule = Rule(
            id="test_list_value",
            field="currency",
            op="in",
            value=["USD", "EUR", "GBP"],
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        retrieved = store.get("test_list_value")
        assert retrieved.value == ["USD", "EUR", "GBP"]

    def test_float_value(self, store):
        """Test rule with float value."""
        rule = Rule(
            id="test_float_value",
            field="amount_ratio",
            op=">",
            value=3.14159,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        retrieved = store.get("test_float_value")
        assert abs(retrieved.value - 3.14159) < 0.0001
