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
    diff_rule_versions,
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


class TestDiffRuleVersions:
    """Tests for diff_rule_versions function."""

    @pytest.fixture
    def base_rule(self):
        """Create a base rule for testing."""
        return Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            severity="high",
            reason="High velocity",
            status="active",
        )

    @pytest.fixture
    def version_a(self, base_rule):
        """Create version A (newer)."""
        return RuleVersion(
            rule_id="test_rule",
            version_id="test_rule_v2",
            rule=base_rule,
            timestamp=datetime.now(timezone.utc),
            created_by="user_a",
            reason="Version A",
        )

    @pytest.fixture
    def version_b(self, base_rule):
        """Create version B (older) with same rule."""
        return RuleVersion(
            rule_id="test_rule",
            version_id="test_rule_v1",
            rule=base_rule,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            created_by="user_b",
            reason="Version B",
        )

    def test_diff_identical_versions(self, version_a, version_b):
        """Test that identical rules show all fields unchanged."""
        result = diff_rule_versions(version_a, version_b)

        assert result.rule_id == "test_rule"
        assert result.version_a_id == "test_rule_v2"
        assert result.version_b_id == "test_rule_v1"
        assert result.is_breaking is False

        # All fields should be unchanged
        for change in result.changes:
            assert change.change_type == "unchanged"

    def test_diff_field_change_is_breaking(self, version_a, version_b):
        """Test that changing field is a breaking change."""
        # Modify version_a's rule field
        version_a.rule = Rule(
            id="test_rule",
            field="amount_to_avg_ratio_30d",  # Changed
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            severity="high",
            reason="High velocity",
            status="active",
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is True
        field_change = next(c for c in result.changes if c.field_name == "field")
        assert field_change.change_type == "modified"
        assert field_change.old_value == "velocity_24h"
        assert field_change.new_value == "amount_to_avg_ratio_30d"

    def test_diff_op_change_is_breaking(self, version_a, version_b):
        """Test that changing operator is a breaking change."""
        version_a.rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">=",  # Changed from >
            value=5,
            action="clamp_min",
            score=80,
            severity="high",
            reason="High velocity",
            status="active",
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is True
        op_change = next(c for c in result.changes if c.field_name == "op")
        assert op_change.change_type == "modified"
        assert op_change.old_value == ">"
        assert op_change.new_value == ">="

    def test_diff_action_change_is_breaking(self, version_a, version_b):
        """Test that changing action is a breaking change."""
        version_a.rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="reject",  # Changed from clamp_min
            severity="high",
            reason="High velocity",
            status="active",
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is True
        action_change = next(c for c in result.changes if c.field_name == "action")
        assert action_change.change_type == "modified"
        assert action_change.old_value == "clamp_min"
        assert action_change.new_value == "reject"

    def test_diff_value_change_is_not_breaking(self, version_a, version_b):
        """Test that changing value is NOT a breaking change (tuning)."""
        version_a.rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=10,  # Changed from 5 (tuning)
            action="clamp_min",
            score=80,
            severity="high",
            reason="High velocity",
            status="active",
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is False  # Value change is NOT breaking
        value_change = next(c for c in result.changes if c.field_name == "value")
        assert value_change.change_type == "modified"
        assert value_change.old_value == 5
        assert value_change.new_value == 10

    def test_diff_score_change_is_not_breaking(self, version_a, version_b):
        """Test that changing score is NOT a breaking change."""
        version_a.rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=90,  # Changed from 80
            severity="high",
            reason="High velocity",
            status="active",
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is False
        score_change = next(c for c in result.changes if c.field_name == "score")
        assert score_change.change_type == "modified"
        assert score_change.old_value == 80
        assert score_change.new_value == 90

    def test_diff_severity_change_is_not_breaking(self, version_a, version_b):
        """Test that changing severity is NOT a breaking change."""
        version_a.rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            severity="medium",  # Changed from high
            reason="High velocity",
            status="active",
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is False
        severity_change = next(c for c in result.changes if c.field_name == "severity")
        assert severity_change.change_type == "modified"
        assert severity_change.old_value == "high"
        assert severity_change.new_value == "medium"

    def test_diff_reason_change_is_not_breaking(self, version_a, version_b):
        """Test that changing reason is NOT a breaking change."""
        version_a.rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            severity="high",
            reason="Updated reason",  # Changed
            status="active",
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is False
        reason_change = next(c for c in result.changes if c.field_name == "reason")
        assert reason_change.change_type == "modified"
        assert reason_change.old_value == "High velocity"
        assert reason_change.new_value == "Updated reason"

    def test_diff_status_change_is_not_breaking(self, version_a, version_b):
        """Test that changing status is NOT a breaking change."""
        version_a.rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            severity="high",
            reason="High velocity",
            status="shadow",  # Changed from active
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is False
        status_change = next(c for c in result.changes if c.field_name == "status")
        assert status_change.change_type == "modified"
        assert status_change.old_value == "active"
        assert status_change.new_value == "shadow"

    def test_diff_multiple_changes(self, version_a, version_b):
        """Test diff with multiple field changes."""
        version_a.rule = Rule(
            id="test_rule",
            field="amount_to_avg_ratio_30d",  # Changed (breaking)
            op=">=",  # Changed (breaking)
            value=10,  # Changed (not breaking)
            action="clamp_min",  # Same as original
            score=80,  # Same as original
            severity="medium",  # Changed
            reason="Updated",  # Changed
            status="shadow",  # Changed
        )

        result = diff_rule_versions(version_a, version_b)

        assert result.is_breaking is True  # field and op changed

        # Count modified fields
        modified_count = sum(1 for c in result.changes if c.change_type == "modified")
        assert modified_count == 6  # field, op, value, severity, reason, status changed

        # Verify score and action are unchanged
        score_change = next(c for c in result.changes if c.field_name == "score")
        assert score_change.change_type == "unchanged"
        action_change = next(c for c in result.changes if c.field_name == "action")
        assert action_change.change_type == "unchanged"

    def test_diff_includes_metadata(self, version_a, version_b):
        """Test that diff result includes version metadata."""
        result = diff_rule_versions(version_a, version_b)

        assert result.version_a_created_by == "user_a"
        assert result.version_b_created_by == "user_b"
        assert result.version_a_timestamp is not None
        assert result.version_b_timestamp is not None
        assert result.version_a_timestamp > result.version_b_timestamp
