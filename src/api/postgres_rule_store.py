"""PostgreSQL-backed rule store implementation.

This module provides a PostgreSQL-backed implementation of the RuleStore
protocol. It is only loaded when RULE_STORE_BACKEND=postgres to avoid
database dependencies in unit tests.

Usage:
    # Set environment variable
    RULE_STORE_BACKEND=postgres

    # The store is then available through get_draft_store()
    from api.draft_store import get_draft_store
    store = get_draft_store()
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from threading import Lock

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, sessionmaker

from api.rules import Rule, RuleStatus

logger = logging.getLogger(__name__)

# Use a separate base for rule-specific tables to avoid coupling with
# the synthetic_pipeline database schema
RuleBase = declarative_base()


class RuleDB(RuleBase):
    """SQLAlchemy model for rule storage."""

    __tablename__ = "rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    field: Mapped[str] = mapped_column(String(100), nullable=False)
    op: Mapped[str] = mapped_column(String(20), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-encoded
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=RuleStatus.DRAFT.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_rule(self) -> Rule:
        """Convert database record to Rule object."""
        import json

        value = json.loads(self.value)
        return Rule(
            id=self.rule_id,
            field=self.field,
            op=self.op,
            value=value,
            action=self.action,
            score=self.score,
            severity=self.severity,
            reason=self.reason,
            status=self.status,
        )

    @classmethod
    def from_rule(cls, rule: Rule) -> "RuleDB":
        """Create database record from Rule object."""
        import json

        return cls(
            rule_id=rule.id,
            field=rule.field,
            op=rule.op,
            value=json.dumps(rule.value),
            action=rule.action,
            score=rule.score,
            severity=rule.severity,
            reason=rule.reason,
            status=rule.status,
        )


def get_rule_database_url() -> str:
    """Get the database URL for rule storage.

    Uses RULE_DATABASE_URL if set, otherwise falls back to DATABASE_URL,
    then constructs from individual POSTGRES_* variables.
    """
    # First check for rule-specific URL
    url = os.getenv("RULE_DATABASE_URL")
    if url:
        return url

    # Fall back to general DATABASE_URL
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    # Construct from individual variables
    user = os.getenv("POSTGRES_USER", "synthetic")
    password = os.getenv("POSTGRES_PASSWORD", "synthetic_dev_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "synthetic_data")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


class PostgresRuleStore:
    """PostgreSQL-backed rule store.

    Implements the RuleStore protocol with PostgreSQL persistence.
    Thread-safe through connection pooling and proper transaction handling.
    """

    def __init__(self, database_url: str | None = None):
        """Initialize PostgreSQL rule store.

        Args:
            database_url: Database connection URL. If None, reads from environment.
        """
        self.database_url = database_url or get_rule_database_url()
        self._engine = create_engine(
            self.database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )
        self._lock = Lock()
        self._tables_created = False

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        if not self._tables_created:
            with self._lock:
                if not self._tables_created:
                    RuleBase.metadata.create_all(bind=self._engine)
                    self._tables_created = True
                    logger.info("Rule database tables created")

    @contextmanager
    def _get_session(self):
        """Get a database session with automatic commit/rollback."""
        self._ensure_tables()
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save(self, rule: Rule) -> None:
        """Save a rule to the database.

        Args:
            rule: Rule to save.

        Raises:
            ValueError: If the status transition is invalid.
        """
        import json

        with self._get_session() as session:
            existing = session.query(RuleDB).filter(RuleDB.rule_id == rule.id).first()

            if existing is None:
                # New rule - must be draft
                if rule.status != RuleStatus.DRAFT.value:
                    raise ValueError(
                        f"Cannot save rule {rule.id} with status {rule.status}. "
                        "Only draft rules can be created."
                    )
                db_rule = RuleDB.from_rule(rule)
                session.add(db_rule)
                logger.debug(f"Saved new draft rule {rule.id}")
            else:
                # Existing rule - validate transition
                valid_transitions = {
                    RuleStatus.DRAFT.value: [RuleStatus.PENDING_REVIEW.value],
                    RuleStatus.PENDING_REVIEW.value: [
                        RuleStatus.APPROVED.value,
                        RuleStatus.DRAFT.value,
                    ],
                    RuleStatus.APPROVED.value: [
                        RuleStatus.ACTIVE.value,
                        RuleStatus.DRAFT.value,
                    ],
                }

                if existing.status in valid_transitions:
                    if rule.status not in valid_transitions[existing.status]:
                        raise ValueError(
                            f"Cannot update rule {rule.id} from {existing.status} "
                            f"to {rule.status}"
                        )

                # Update fields
                existing.field = rule.field
                existing.op = rule.op
                existing.value = json.dumps(rule.value)
                existing.action = rule.action
                existing.score = rule.score
                existing.severity = rule.severity
                existing.reason = rule.reason
                existing.status = rule.status
                existing.updated_at = datetime.utcnow()

                logger.debug(f"Updated rule {rule.id} to status {rule.status}")

    def get(self, rule_id: str) -> Rule | None:
        """Get a rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            Rule if found, None otherwise.
        """
        with self._get_session() as session:
            db_rule = session.query(RuleDB).filter(RuleDB.rule_id == rule_id).first()
            if db_rule is None:
                return None
            return db_rule.to_rule()

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
        with self._get_session() as session:
            query = session.query(RuleDB)

            if status is not None:
                query = query.filter(RuleDB.status == status)
            elif not include_archived:
                query = query.filter(RuleDB.status != RuleStatus.ARCHIVED.value)

            query = query.order_by(RuleDB.rule_id)
            return [db_rule.to_rule() for db_rule in query.all()]

    def delete(self, rule_id: str) -> bool:
        """Archive a rule (soft delete).

        Args:
            rule_id: Rule identifier.

        Returns:
            True if rule was found and archived, False otherwise.
        """
        with self._get_session() as session:
            db_rule = session.query(RuleDB).filter(RuleDB.rule_id == rule_id).first()

            if db_rule is None:
                return False

            if db_rule.status != RuleStatus.DRAFT.value:
                logger.warning(
                    f"Cannot delete rule {rule_id} with status {db_rule.status}. "
                    "Only draft rules can be deleted."
                )
                return False

            db_rule.status = RuleStatus.ARCHIVED.value
            db_rule.updated_at = datetime.utcnow()
            logger.debug(f"Archived rule {rule_id}")
            return True

    def exists(self, rule_id: str) -> bool:
        """Check if a rule exists.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if rule exists, False otherwise.
        """
        with self._get_session() as session:
            count = session.query(RuleDB).filter(RuleDB.rule_id == rule_id).count()
            return count > 0
