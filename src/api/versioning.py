"""Rule versioning and rollback functionality."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from api.audit import get_audit_logger
from api.rules import Rule, RuleSet

logger = logging.getLogger(__name__)


@dataclass
class RuleVersion:
    """A versioned snapshot of a rule."""

    rule_id: str
    version_id: str
    rule: Rule
    timestamp: datetime
    created_by: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "version_id": self.version_id,
            "rule": asdict(self.rule),
            "timestamp": self.timestamp.isoformat(),
            "created_by": self.created_by,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleVersion":
        """Create from dictionary."""
        rule_data = data["rule"]
        rule = Rule(**rule_data)

        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            rule_id=data["rule_id"],
            version_id=data["version_id"],
            rule=rule,
            timestamp=timestamp,
            created_by=data["created_by"],
            reason=data.get("reason", ""),
        )


class RuleVersionStore:
    """Store and manage rule versions."""

    def __init__(self, storage_path: Path | str | None = None):
        """Initialize version store.

        Args:
            storage_path: Path for persistent storage. If None, in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._versions: dict[str, list[RuleVersion]] = {}  # rule_id -> list of versions
        self._audit_logger = get_audit_logger()
        self._lock = Lock()  # Thread safety

        # Load existing versions if file exists
        if self.storage_path and self.storage_path.exists():
            self._load_versions()

    def _load_versions(self) -> None:
        """Load versions from storage file."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                self._versions = {}
                for rule_id, version_list in data.items():
                    self._versions[rule_id] = [
                        RuleVersion.from_dict(v) for v in version_list
                    ]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                f"Failed to load rule versions from {self.storage_path}: {e}"
            )
            self._versions = {}

    def _save_versions(self) -> None:
        """Save versions to storage file with atomic write."""
        if not self.storage_path:
            return

        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {}
            for rule_id, versions in self._versions.items():
                data[rule_id] = [v.to_dict() for v in versions]

            # Atomic write: write to temp file, then rename
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.storage_path)
        except OSError as e:
            logger.error(f"Failed to save rule versions to {self.storage_path}: {e}")

    def _generate_version_id(self, rule_id: str) -> str:
        """Generate a unique version ID.

        Args:
            rule_id: ID of the rule.

        Returns:
            Version ID in format: rule_id_timestamp.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"{rule_id}_{timestamp}"

    def save(self, rule: Rule, created_by: str, reason: str = "") -> RuleVersion:
        """Save a new version of a rule.

        Args:
            rule: Rule to save.
            created_by: Who created this version.
            reason: Optional reason for this version.

        Returns:
            Created RuleVersion.
        """
        with self._lock:
            version_id = self._generate_version_id(rule.id)
            version = RuleVersion(
                rule_id=rule.id,
                version_id=version_id,
                rule=rule,
                timestamp=datetime.now(timezone.utc),
                created_by=created_by,
                reason=reason,
            )

            # Add to versions list
            if rule.id not in self._versions:
                self._versions[rule.id] = []
            self._versions[rule.id].append(version)

            # Sort by timestamp (oldest first)
            self._versions[rule.id].sort(key=lambda v: v.timestamp)

            self._save_versions()

            logger.info(f"Saved version {version_id} of rule {rule.id}")

            return version

    def get_version(self, rule_id: str, version_id: str) -> RuleVersion | None:
        """Get a specific version of a rule.

        Args:
            rule_id: ID of the rule.
            version_id: Version ID.

        Returns:
            RuleVersion if found, None otherwise.
        """
        with self._lock:
            if rule_id not in self._versions:
                return None

            for version in self._versions[rule_id]:
                if version.version_id == version_id:
                    return version

            return None

    def list_versions(self, rule_id: str) -> list[RuleVersion]:
        """List all versions of a rule.

        Args:
            rule_id: ID of the rule.

        Returns:
            List of versions, ordered by timestamp (oldest first).
        """
        return self._versions.get(rule_id, []).copy()

    def get_latest_version(self, rule_id: str) -> RuleVersion | None:
        """Get the latest version of a rule.

        Args:
            rule_id: ID of the rule.

        Returns:
            Latest RuleVersion if found, None otherwise.
        """
        with self._lock:
            versions = self._versions.get(rule_id, [])
            if not versions:
                return None
            return versions[-1]  # Last one is latest (sorted by timestamp)

    def rollback(
        self, rule_id: str, version_id: str, rolled_back_by: str, reason: str = ""
    ) -> RuleVersion:
        """Rollback a rule to a previous version.

        This creates a new version (does not delete history).

        Args:
            rule_id: ID of the rule.
            version_id: Version to rollback to.
            rolled_back_by: Who is performing the rollback.
            reason: Optional reason for rollback.

        Returns:
            New RuleVersion created from the rollback.

        Raises:
            ValueError: If version not found.
        """
        with self._lock:
            # Get the version to rollback to (inline to avoid nested lock)
            if rule_id not in self._versions:
                raise ValueError(f"Version {version_id} not found for rule {rule_id}")

            target_version = None
            for version in self._versions[rule_id]:
                if version.version_id == version_id:
                    target_version = version
                    break

            if target_version is None:
                raise ValueError(f"Version {version_id} not found for rule {rule_id}")

            # Get latest version before rollback (inline)
            latest_before = None
            if rule_id in self._versions and self._versions[rule_id]:
                latest_before = self._versions[rule_id][-1]

            # Create new version from the target version's rule
            rollback_reason = reason or f"Rollback to version {version_id}"
            version_id_new = self._generate_version_id(rule_id)
            new_version = RuleVersion(
                rule_id=rule_id,
                version_id=version_id_new,
                rule=target_version.rule,
                timestamp=datetime.now(timezone.utc),
                created_by=rolled_back_by,
                reason=rollback_reason,
            )

            # Add to versions list
            if rule_id not in self._versions:
                self._versions[rule_id] = []
            self._versions[rule_id].append(new_version)
            self._versions[rule_id].sort(key=lambda v: v.timestamp)

            self._save_versions()

            # Log rollback in audit trail (outside lock to avoid deadlock)
            before_version_id = latest_before.version_id if latest_before else None

        self._audit_logger.log(
            rule_id=rule_id,
            action="rollback",
            actor=rolled_back_by,
            before_state={"version_id": before_version_id},
            after_state={
                "version_id": new_version.version_id,
                "rolled_back_to": version_id,
            },
            reason=rollback_reason,
        )

        logger.info(
            "Rolled back rule %s to version %s, new version %s",
            rule_id,
            version_id,
            new_version.version_id,
        )

        return new_version

    def get_ruleset_at(
        self, timestamp: datetime, rule_ids: list[str] | None = None
    ) -> RuleSet:
        """Reconstruct a ruleset at a specific point in time.

        Args:
            timestamp: Point in time to reconstruct.
            rule_ids: Optional list of rule IDs to include. If None, includes all rules.

        Returns:
            RuleSet with rules as they were at the specified time.
        """
        rules = []

        # Determine which rules to include
        if rule_ids is None:
            rule_ids = list(self._versions.keys())

        for rule_id in rule_ids:
            if rule_id not in self._versions:
                continue

            # Find the version that was active at the timestamp
            # (the latest version created before or at the timestamp)
            active_version = None
            for version in self._versions[rule_id]:
                if version.timestamp <= timestamp:
                    active_version = version
                else:
                    break  # Versions are sorted, so we can stop here

            if active_version:
                rules.append(active_version.rule)

        return RuleSet(version=f"reconstructed_at_{timestamp.isoformat()}", rules=rules)


# Global version store instance (can be replaced for testing)
_global_version_store: RuleVersionStore | None = None


def get_version_store() -> RuleVersionStore:
    """Get the global version store instance.

    Returns:
        Global RuleVersionStore instance.
    """
    global _global_version_store
    if _global_version_store is None:
        storage_path = os.getenv("VERSION_STORAGE_PATH")
        _global_version_store = RuleVersionStore(storage_path=storage_path)
    return _global_version_store


def set_version_store(store: RuleVersionStore) -> None:
    """Set the global version store instance (for testing).

    Args:
        store: RuleVersionStore instance to use.
    """
    global _global_version_store
    _global_version_store = store
