"""Tests for rule versioning and rollback."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from api.audit import AuditLogger, set_audit_logger
from api.rules import Rule
from api.versioning import (
    RuleVersion,
    RuleVersionStore,
    get_version_store,
    set_version_store,
)


class TestRuleVersion:
    """Tests for RuleVersion dataclass."""

    def test_rule_version_creation(self):
        """Test creating a rule version."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )

        version = RuleVersion(
            rule_id="test_rule",
            version_id="test_rule_v1",
            rule=rule,
            timestamp=datetime.now(timezone.utc),
            created_by="user123",
            reason="Initial version",
        )

        assert version.rule_id == "test_rule"
        assert version.version_id == "test_rule_v1"
        assert version.rule.id == "test_rule"
        assert version.created_by == "user123"

    def test_rule_version_serialization(self):
        """Test serialization and deserialization."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )

        original = RuleVersion(
            rule_id="test_rule",
            version_id="test_rule_v1",
            rule=rule,
            timestamp=datetime.now(timezone.utc),
            created_by="user123",
        )

        # Serialize
        data = original.to_dict()
        assert isinstance(data["timestamp"], str)

        # Deserialize
        restored = RuleVersion.from_dict(data)
        assert restored.rule_id == original.rule_id
        assert restored.version_id == original.version_id
        assert restored.rule.id == original.rule.id


class TestRuleVersionStore:
    """Tests for RuleVersionStore class."""

    @pytest.fixture
    def version_store(self):
        """Create a version store for testing."""
        test_logger = AuditLogger()
        set_audit_logger(test_logger)
        return RuleVersionStore()

    @pytest.fixture
    def sample_rule(self):
        """Create a sample rule."""
        return Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="draft",
        )

    def test_save_version(self, version_store, sample_rule):
        """Test saving a rule version."""
        version = version_store.save(
            sample_rule, created_by="user123", reason="Initial version"
        )

        assert version.rule_id == "test_rule"
        assert version.created_by == "user123"
        assert version.reason == "Initial version"

    def test_get_version(self, version_store, sample_rule):
        """Test retrieving a specific version."""
        version = version_store.save(sample_rule, created_by="user123")
        retrieved = version_store.get_version("test_rule", version.version_id)

        assert retrieved is not None
        assert retrieved.version_id == version.version_id
        assert retrieved.rule.id == version.rule.id

    def test_get_nonexistent_version(self, version_store):
        """Test retrieving a nonexistent version."""
        retrieved = version_store.get_version("nonexistent", "v1")
        assert retrieved is None

    def test_list_versions(self, version_store, sample_rule):
        """Test listing all versions of a rule."""
        version1 = version_store.save(sample_rule, created_by="user1")

        # Update rule and save again
        updated_rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=10,  # Changed value
            action="clamp_min",
            score=80,
            status="active",  # Changed status
        )
        version2 = version_store.save(updated_rule, created_by="user2")

        versions = version_store.list_versions("test_rule")
        assert len(versions) == 2
        assert versions[0].version_id == version1.version_id
        assert versions[1].version_id == version2.version_id

    def test_get_latest_version(self, version_store, sample_rule):
        """Test getting the latest version."""
        _ = version_store.save(sample_rule, created_by="user1")

        updated_rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=10,
            action="clamp_min",
            score=80,
        )
        version2 = version_store.save(updated_rule, created_by="user2")

        latest = version_store.get_latest_version("test_rule")
        assert latest is not None
        assert latest.version_id == version2.version_id

    def test_rollback(self, version_store, sample_rule):
        """Test rolling back to a previous version."""
        # Save initial version
        version1 = version_store.save(sample_rule, created_by="user1")

        # Update and save
        updated_rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=10,
            action="clamp_min",
            score=80,
        )
        version2 = version_store.save(updated_rule, created_by="user2")

        # Rollback to version1
        rollback_version = version_store.rollback(
            "test_rule", version1.version_id, rolled_back_by="user3", reason="Bug fix"
        )

        # Should create a new version
        assert rollback_version.version_id != version1.version_id
        assert rollback_version.version_id != version2.version_id

        # Rule should match version1
        assert rollback_version.rule.value == version1.rule.value

        # Should have 3 versions now
        versions = version_store.list_versions("test_rule")
        assert len(versions) == 3

    def test_rollback_nonexistent_version_raises(self, version_store, sample_rule):
        """Test that rolling back to nonexistent version raises error."""
        version_store.save(sample_rule, created_by="user1")

        with pytest.raises(ValueError, match="not found"):
            version_store.rollback(
                "test_rule", "nonexistent_version", rolled_back_by="user1"
            )

    def test_get_ruleset_at_timestamp(self, version_store):
        """Test reconstructing ruleset at a point in time."""
        base_time = datetime.now(timezone.utc)

        # Create rule1 at time 0
        rule1_v1 = Rule(
            id="rule1",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        version1 = version_store.save(rule1_v1, created_by="user1")
        version1.timestamp = base_time

        # Update rule1 at time +1 day
        rule1_v2 = Rule(
            id="rule1",
            field="velocity_24h",
            op=">",
            value=10,
            action="clamp_min",
            score=80,
        )
        version2 = version_store.save(rule1_v2, created_by="user1")
        version2.timestamp = base_time + timedelta(days=1)

        # Create rule2 at time +2 days
        rule2 = Rule(
            id="rule2",
            field="amount_to_avg_ratio_30d",
            op=">",
            value=3.0,
            action="clamp_min",
            score=75,
        )
        version3 = version_store.save(rule2, created_by="user1")
        version3.timestamp = base_time + timedelta(days=2)

        # Reconstruct at time +0.5 days (should have rule1_v1, no rule2)
        ruleset = version_store.get_ruleset_at(base_time + timedelta(hours=12))
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].id == "rule1"
        assert ruleset.rules[0].value == 5  # Should be v1

        # Reconstruct at time +1.5 days (should have rule1_v2, no rule2)
        ruleset = version_store.get_ruleset_at(base_time + timedelta(days=1, hours=12))
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].id == "rule1"
        assert ruleset.rules[0].value == 10  # Should be v2

        # Reconstruct at time +3 days (should have rule1_v2 and rule2)
        ruleset = version_store.get_ruleset_at(base_time + timedelta(days=3))
        assert len(ruleset.rules) == 2
        rule_ids = {r.id for r in ruleset.rules}
        assert "rule1" in rule_ids
        assert "rule2" in rule_ids

    def test_get_ruleset_at_with_specific_rule_ids(self, version_store):
        """Test reconstructing ruleset with specific rule IDs."""
        rule1 = Rule(
            id="rule1", field="v", op=">", value=5, action="clamp_min", score=80
        )
        rule2 = Rule(
            id="rule2", field="a", op=">", value=3, action="clamp_min", score=75
        )
        rule3 = Rule(
            id="rule3", field="b", op=">", value=2, action="clamp_min", score=70
        )

        version_store.save(rule1, created_by="user1")
        version_store.save(rule2, created_by="user1")
        version_store.save(rule3, created_by="user1")

        # Reconstruct with only rule1 and rule2
        ruleset = version_store.get_ruleset_at(
            datetime.now(timezone.utc), rule_ids=["rule1", "rule2"]
        )
        assert len(ruleset.rules) == 2
        rule_ids = {r.id for r in ruleset.rules}
        assert "rule1" in rule_ids
        assert "rule2" in rule_ids
        assert "rule3" not in rule_ids


class TestRuleVersionStorePersistence:
    """Tests for persistent storage of versions."""

    def test_save_and_load_versions(self):
        """Test saving and loading versions from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "versions.json"

            # Create store and save versions
            store1 = RuleVersionStore(storage_path=storage_path)
            rule1 = Rule(
                id="rule1", field="v", op=">", value=5, action="clamp_min", score=80
            )
            store1.save(rule1, created_by="user1")

            # Create new store and load versions
            store2 = RuleVersionStore(storage_path=storage_path)
            versions = store2.list_versions("rule1")
            assert len(versions) == 1
            assert versions[0].rule.id == "rule1"

    def test_load_malformed_file_handles_gracefully(self):
        """Test that malformed JSON files are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "versions.json"

            # Write malformed JSON
            with open(storage_path, "w") as f:
                f.write("invalid json {")

            # Should not raise, just start with empty versions
            store = RuleVersionStore(storage_path=storage_path)
            assert len(store._versions) == 0


class TestGlobalVersionStore:
    """Tests for global version store."""

    def test_get_version_store_returns_singleton(self):
        """Test that get_version_store returns a singleton."""
        store1 = get_version_store()
        store2 = get_version_store()

        assert store1 is store2

    def test_set_version_store_for_testing(self):
        """Test that set_version_store allows replacing for testing."""
        original = get_version_store()
        test_store = RuleVersionStore()

        set_version_store(test_store)
        assert get_version_store() is test_store

        # Restore original
        set_version_store(original)
        assert get_version_store() is original
