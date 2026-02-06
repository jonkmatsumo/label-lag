"""Audit trail for rule changes."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """A single audit log record."""

    rule_id: str
    action: str  # create, update, state_change, delete
    actor: str
    timestamp: datetime
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format string
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditRecord":
        """Create from dictionary."""
        # Convert ISO format string back to datetime
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class AuditLogger:
    """Append-only audit logger for rule changes."""

    def __init__(self, storage_path: Path | str | None = None):
        """Initialize audit logger.

        Args:
            storage_path: Path for persistent storage. If None, in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._records: list[AuditRecord] = []
        self._lock = Lock()  # Thread safety

        # Load existing records if file exists
        if self.storage_path and self.storage_path.exists():
            self._load_records()

    def _load_records(self) -> None:
        """Load records from storage file."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                self._records = [AuditRecord.from_dict(record) for record in data]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                f"Failed to load audit records from {self.storage_path}: {e}"
            )
            self._records = []

    def _save_records(self) -> None:
        """Save records to storage file with atomic write."""
        if not self.storage_path:
            return

        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: write to temp file, then rename
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump([r.to_dict() for r in self._records], f, indent=2)
            temp_path.replace(self.storage_path)
        except OSError as e:
            logger.error(f"Failed to save audit records to {self.storage_path}: {e}")

    def log(
        self,
        rule_id: str,
        action: str,
        actor: str,
        before_state: dict[str, Any] | None = None,
        after_state: dict[str, Any] | None = None,
        reason: str = "",
    ) -> AuditRecord:
        """Log a rule change.

        Args:
            rule_id: ID of the rule that changed.
            action: Type of action (create, update, state_change, delete).
            actor: Who made the change (user ID, system, etc.).
            before_state: State before the change (for updates/state_change).
            after_state: State after the change.
            reason: Optional reason for the change.

        Returns:
            Created audit record.
        """
        with self._lock:
            record = AuditRecord(
                rule_id=rule_id,
                action=action,
                actor=actor,
                timestamp=datetime.now(timezone.utc),
                before_state=before_state,
                after_state=after_state,
                reason=reason,
            )

            self._records.append(record)
            self._save_records()

            logger.info(
                f"Audit log: {action} on rule {rule_id} by {actor} "
                f"at {record.timestamp}"
            )

            return record

    def query(
        self,
        rule_id: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditRecord]:
        """Query audit records.

        Args:
            rule_id: Filter by rule ID.
            actor: Filter by actor.
            action: Filter by action type.
            start_date: Filter records after this date (inclusive).
            end_date: Filter records before this date (inclusive).

        Returns:
            List of matching audit records, ordered by timestamp (oldest first).
        """
        with self._lock:
            results = self._records.copy()

            # Apply filters
            if rule_id is not None:
                results = [r for r in results if r.rule_id == rule_id]

            if actor is not None:
                results = [r for r in results if r.actor == actor]

            if action is not None:
                results = [r for r in results if r.action == action]

            if start_date is not None:
                results = [r for r in results if r.timestamp >= start_date]

            if end_date is not None:
                results = [r for r in results if r.timestamp <= end_date]

            # Sort by timestamp (oldest first)
            results.sort(key=lambda r: r.timestamp)

            return results

    def get_rule_history(self, rule_id: str) -> list[AuditRecord]:
        """Get complete history for a rule.

        Args:
            rule_id: ID of the rule.

        Returns:
            List of audit records for the rule, ordered by timestamp.
        """
        return self.query(rule_id=rule_id)

    def get_all_records(self) -> list[AuditRecord]:
        """Get all audit records.

        Returns:
            List of all records, ordered by timestamp.
        """
        with self._lock:
            return sorted(self._records, key=lambda r: r.timestamp)


# Global audit logger instance (can be replaced for testing)
_global_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance.

    Returns:
        Global AuditLogger instance.
    """
    global _global_audit_logger
    if _global_audit_logger is None:
        storage_path = os.getenv("AUDIT_STORAGE_PATH")
        _global_audit_logger = AuditLogger(storage_path=storage_path)
    return _global_audit_logger


def set_audit_logger(logger: AuditLogger) -> None:
    """Set the global audit logger instance (for testing).

    Args:
        logger: AuditLogger instance to use.
    """
    global _global_audit_logger
    _global_audit_logger = logger
