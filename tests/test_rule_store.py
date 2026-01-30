"""Tests for rule store abstraction and backends."""

import pytest

from api.draft_store import DraftRuleStore
from api.rule_store import get_rule_store_backend, is_postgres_backend_enabled
from api.rules import Rule, RuleStatus


class TestRuleStoreBackendConfig:
    """Tests for rule store backend configuration."""

    def test_default_backend_is_inmemory(self, monkeypatch):
        """Test that default backend is inmemory when env var not set."""
        monkeypatch.delenv("RULE_STORE_BACKEND", raising=False)
        assert get_rule_store_backend() == "inmemory"

    def test_backend_from_env_var(self, monkeypatch):
        """Test that backend is read from RULE_STORE_BACKEND env var."""
        monkeypatch.setenv("RULE_STORE_BACKEND", "postgres")
        assert get_rule_store_backend() == "postgres"

    def test_postgres_backend_disabled_by_default(self, monkeypatch):
        """Test that postgres backend is disabled by default."""
        monkeypatch.delenv("RULE_STORE_BACKEND", raising=False)
        assert is_postgres_backend_enabled() is False

    def test_postgres_backend_enabled_when_configured(self, monkeypatch):
        """Test that postgres backend is enabled when configured."""
        monkeypatch.setenv("RULE_STORE_BACKEND", "postgres")
        assert is_postgres_backend_enabled() is True


class TestDraftRuleStoreImplementsProtocol:
    """Tests verifying DraftRuleStore implements RuleStore protocol."""

    @pytest.fixture
    def store(self):
        """Create an in-memory store for testing."""
        return DraftRuleStore()

    def test_draft_store_has_save_method(self, store):
        """Test that DraftRuleStore has save method."""
        assert hasattr(store, "save")
        assert callable(store.save)

    def test_draft_store_has_get_method(self, store):
        """Test that DraftRuleStore has get method."""
        assert hasattr(store, "get")
        assert callable(store.get)

    def test_draft_store_has_list_rules_method(self, store):
        """Test that DraftRuleStore has list_rules method."""
        assert hasattr(store, "list_rules")
        assert callable(store.list_rules)

    def test_draft_store_has_delete_method(self, store):
        """Test that DraftRuleStore has delete method."""
        assert hasattr(store, "delete")
        assert callable(store.delete)

    def test_draft_store_has_exists_method(self, store):
        """Test that DraftRuleStore has exists method."""
        assert hasattr(store, "exists")
        assert callable(store.exists)

    def test_save_and_get_round_trip(self, store):
        """Test that save and get work correctly."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            severity="medium",
            status=RuleStatus.DRAFT.value,
        )

        store.save(rule)
        retrieved = store.get("test_rule")

        assert retrieved is not None
        assert retrieved.id == rule.id
        assert retrieved.field == rule.field
        assert retrieved.status == RuleStatus.DRAFT.value

    def test_list_rules_returns_list(self, store):
        """Test that list_rules returns a list."""
        rules = store.list_rules()
        assert isinstance(rules, list)

    def test_delete_returns_bool(self, store):
        """Test that delete returns a boolean."""
        # Create a draft rule first
        rule = Rule(
            id="delete_test",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        result = store.delete("delete_test")
        assert isinstance(result, bool)
        assert result is True

    def test_exists_returns_bool(self, store):
        """Test that exists returns a boolean."""
        result = store.exists("nonexistent")
        assert isinstance(result, bool)
        assert result is False

        # Create a rule and check again
        rule = Rule(
            id="exists_test",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        result = store.exists("exists_test")
        assert result is True


class TestInMemoryBehaviorPreserved:
    """Tests ensuring in-memory behavior is preserved."""

    @pytest.fixture
    def store(self):
        """Create an in-memory store (no storage path)."""
        return DraftRuleStore(storage_path=None)

    def test_inmemory_store_works_without_file(self, store):
        """Test that in-memory store works without any file path."""
        rule = Rule(
            id="memory_test",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )

        store.save(rule)
        retrieved = store.get("memory_test")

        assert retrieved is not None
        assert retrieved.id == "memory_test"

    def test_inmemory_store_list_rules_works(self, store):
        """Test that list_rules works for in-memory store."""
        # Save multiple rules
        for i in range(3):
            rule = Rule(
                id=f"list_test_{i}",
                field="velocity_24h",
                op=">",
                value=5 + i,
                action="clamp_min",
                score=70,
                status=RuleStatus.DRAFT.value,
            )
            store.save(rule)

        rules = store.list_rules()
        assert len(rules) == 3
        assert all(r.status == RuleStatus.DRAFT.value for r in rules)

    def test_inmemory_store_status_filter_works(self, store):
        """Test that status filter works for in-memory store."""
        # Save a draft rule
        rule = Rule(
            id="filter_test",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        store.save(rule)

        # Filter by draft status
        draft_rules = store.list_rules(status=RuleStatus.DRAFT.value)
        assert len(draft_rules) >= 1
        assert all(r.status == RuleStatus.DRAFT.value for r in draft_rules)

        # Filter by active status (should be empty for fresh store)
        active_rules = store.list_rules(status=RuleStatus.ACTIVE.value)
        assert len(active_rules) == 0
