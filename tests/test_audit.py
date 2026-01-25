"""Tests for audit trail infrastructure."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from api.audit import AuditLogger, AuditRecord, get_audit_logger, set_audit_logger


class TestAuditRecord:
    """Tests for AuditRecord dataclass."""

    def test_audit_record_creation(self):
        """Test creating an audit record."""
        record = AuditRecord(
            rule_id="test_rule",
            action="create",
            actor="user123",
            timestamp=datetime.now(timezone.utc),
            after_state={"id": "test_rule", "status": "active"},
            reason="Initial creation",
        )

        assert record.rule_id == "test_rule"
        assert record.action == "create"
        assert record.actor == "user123"
        assert record.reason == "Initial creation"

    def test_audit_record_serialization(self):
        """Test serialization and deserialization."""
        original = AuditRecord(
            rule_id="test_rule",
            action="update",
            actor="user123",
            timestamp=datetime.now(timezone.utc),
            before_state={"status": "draft"},
            after_state={"status": "active"},
        )

        # Serialize
        data = original.to_dict()
        assert isinstance(data["timestamp"], str)

        # Deserialize
        restored = AuditRecord.from_dict(data)
        assert restored.rule_id == original.rule_id
        assert restored.action == original.action
        assert restored.timestamp == original.timestamp


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_log_create_action(self):
        """Test logging a create action."""
        logger = AuditLogger()
        record = logger.log(
            rule_id="test_rule",
            action="create",
            actor="user123",
            after_state={"id": "test_rule", "status": "draft"},
            reason="New rule created",
        )

        assert record.rule_id == "test_rule"
        assert record.action == "create"
        assert record.actor == "user123"
        assert len(logger._records) == 1

    def test_log_state_change(self):
        """Test logging a state change."""
        logger = AuditLogger()
        record = logger.log(
            rule_id="test_rule",
            action="state_change",
            actor="user123",
            before_state={"status": "draft"},
            after_state={"status": "active"},
            reason="Approved for production",
        )

        assert record.action == "state_change"
        assert record.before_state["status"] == "draft"
        assert record.after_state["status"] == "active"

    def test_query_by_rule_id(self):
        """Test querying by rule ID."""
        logger = AuditLogger()
        logger.log("rule1", "create", "user1", after_state={"id": "rule1"})
        logger.log("rule2", "create", "user2", after_state={"id": "rule2"})
        logger.log("rule1", "update", "user1", after_state={"id": "rule1", "status": "active"})

        results = logger.query(rule_id="rule1")
        assert len(results) == 2
        assert all(r.rule_id == "rule1" for r in results)

    def test_query_by_actor(self):
        """Test querying by actor."""
        logger = AuditLogger()
        logger.log("rule1", "create", "user1")
        logger.log("rule2", "create", "user2")
        logger.log("rule3", "update", "user1")

        results = logger.query(actor="user1")
        assert len(results) == 2
        assert all(r.actor == "user1" for r in results)

    def test_query_by_action(self):
        """Test querying by action type."""
        logger = AuditLogger()
        logger.log("rule1", "create", "user1")
        logger.log("rule2", "update", "user1")
        logger.log("rule3", "create", "user2")

        results = logger.query(action="create")
        assert len(results) == 2
        assert all(r.action == "create" for r in results)

    def test_query_by_date_range(self):
        """Test querying by date range."""
        logger = AuditLogger()
        base_time = datetime.now(timezone.utc)

        # Create records at different times
        record1 = logger.log("rule1", "create", "user1")
        record1.timestamp = base_time - timedelta(days=2)

        record2 = logger.log("rule2", "create", "user1")
        record2.timestamp = base_time - timedelta(days=1)

        record3 = logger.log("rule3", "create", "user1")
        record3.timestamp = base_time

        # Query for last 1.5 days
        results = logger.query(
            start_date=base_time - timedelta(days=1, hours=12),
            end_date=base_time,
        )
        assert len(results) == 2
        assert results[0].rule_id == "rule2"
        assert results[1].rule_id == "rule3"

    def test_get_rule_history(self):
        """Test getting complete history for a rule."""
        logger = AuditLogger()
        logger.log("rule1", "create", "user1", after_state={"status": "draft"})
        logger.log("rule2", "create", "user2", after_state={"status": "draft"})
        logger.log("rule1", "state_change", "user1", before_state={"status": "draft"}, after_state={"status": "active"})
        logger.log("rule1", "update", "user1", after_state={"status": "active", "score": 80})

        history = logger.get_rule_history("rule1")
        assert len(history) == 3
        assert history[0].action == "create"
        assert history[1].action == "state_change"
        assert history[2].action == "update"

    def test_append_only_semantics(self):
        """Test that records are append-only (no deletion)."""
        logger = AuditLogger()
        record1 = logger.log("rule1", "create", "user1")
        record2 = logger.log("rule1", "update", "user1")

        # Records should be in order
        all_records = logger.get_all_records()
        assert len(all_records) == 2
        assert all_records[0] == record1
        assert all_records[1] == record2

        # No way to delete records
        assert len(logger._records) == 2


class TestAuditLoggerPersistence:
    """Tests for persistent storage of audit logs."""

    def test_save_and_load_records(self):
        """Test saving and loading records from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "audit.json"

            # Create logger and log some records
            logger1 = AuditLogger(storage_path=storage_path)
            logger1.log("rule1", "create", "user1", after_state={"id": "rule1"})
            logger1.log("rule2", "create", "user2", after_state={"id": "rule2"})

            # Create new logger and load records
            logger2 = AuditLogger(storage_path=storage_path)
            assert len(logger2._records) == 2
            assert logger2._records[0].rule_id == "rule1"
            assert logger2._records[1].rule_id == "rule2"

    def test_load_malformed_file_handles_gracefully(self):
        """Test that malformed JSON files are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "audit.json"

            # Write malformed JSON
            with open(storage_path, "w") as f:
                f.write("invalid json {")

            # Should not raise, just start with empty records
            logger = AuditLogger(storage_path=storage_path)
            assert len(logger._records) == 0

    def test_in_memory_only_mode(self):
        """Test that in-memory mode works without file."""
        logger = AuditLogger(storage_path=None)
        logger.log("rule1", "create", "user1")

        assert len(logger._records) == 1
        assert logger.storage_path is None


class TestGlobalAuditLogger:
    """Tests for global audit logger."""

    def test_get_audit_logger_returns_singleton(self):
        """Test that get_audit_logger returns a singleton."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()

        assert logger1 is logger2

    def test_set_audit_logger_for_testing(self):
        """Test that set_audit_logger allows replacing for testing."""
        original = get_audit_logger()
        test_logger = AuditLogger()

        set_audit_logger(test_logger)
        assert get_audit_logger() is test_logger

        # Restore original
        set_audit_logger(original)
        assert get_audit_logger() is original
