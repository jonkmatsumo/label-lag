"""PostgreSQL-backed inference event sink.

This module provides a PostgreSQL-backed implementation of the inference
event sink. It is only loaded when INFERENCE_EVENT_SINK=postgres to avoid
database dependencies in unit tests.

Best-effort insert: if database is unavailable, events are logged to
warning and discarded (no retries, bounded, non-blocking).
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from threading import Lock

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, sessionmaker

from api.inference_log import InferenceEvent

logger = logging.getLogger(__name__)

# Separate base for inference event tables
InferenceEventBase = declarative_base()


class InferenceEventDB(InferenceEventBase):
    """SQLAlchemy model for inference event storage."""

    __tablename__ = "inference_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    rules_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_score: Mapped[int] = mapped_column(Integer, nullable=False)
    final_score: Mapped[int] = mapped_column(Integer, nullable=False)
    rule_impacts_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )


def get_inference_database_url() -> str:
    """Get the database URL for inference event storage.

    Uses INFERENCE_DATABASE_URL if set, otherwise falls back to DATABASE_URL,
    then constructs from individual POSTGRES_* variables.
    """
    # First check for inference-specific URL
    url = os.getenv("INFERENCE_DATABASE_URL")
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


class PostgresInferenceSink:
    """PostgreSQL-backed inference event sink.

    Implements best-effort insert: if database is unavailable, events are
    logged to warning and discarded. This ensures inference requests never
    fail due to logging issues.
    """

    def __init__(self, database_url: str | None = None):
        """Initialize PostgreSQL inference event sink.

        Args:
            database_url: Database connection URL. If None, reads from env.
        """
        self.database_url = database_url or get_inference_database_url()
        self._engine = create_engine(
            self.database_url,
            pool_size=3,  # Smaller pool for logging
            max_overflow=5,
            pool_pre_ping=True,
            pool_timeout=5,  # Short timeout to avoid blocking
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )
        self._lock = Lock()
        self._tables_created = False
        self._enabled = True  # Can be disabled on persistent failures

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        if not self._tables_created:
            with self._lock:
                if not self._tables_created:
                    try:
                        InferenceEventBase.metadata.create_all(bind=self._engine)
                        self._tables_created = True
                        logger.info("Inference event database tables created")
                    except Exception as e:
                        logger.warning(f"Failed to create inference tables: {e}")

    @contextmanager
    def _get_session(self):
        """Get a database session with automatic commit/rollback."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def log_event(self, event: InferenceEvent) -> None:
        """Log an inference event to PostgreSQL.

        Best-effort: if insert fails, log warning and continue.
        Never raises exceptions.

        Args:
            event: The inference event to log.
        """
        if not self._enabled:
            return

        try:
            self._ensure_tables()

            # Serialize rule impacts to JSON
            rule_impacts = [
                {
                    "rule_id": ri.rule_id,
                    "is_shadow": ri.is_shadow,
                    "score_delta": ri.score_delta,
                    "details": ri.details,
                }
                for ri in event.rule_impacts
            ]

            db_event = InferenceEventDB(
                request_id=event.request_id,
                timestamp=event.timestamp,
                model_version=event.model_version,
                rules_version=event.rules_version,
                model_score=event.model_score,
                final_score=event.final_score,
                rule_impacts_json=json.dumps(rule_impacts),
            )

            with self._get_session() as session:
                session.add(db_event)

        except Exception as e:
            logger.warning(f"Failed to log inference event to PostgreSQL: {e}")
            # Don't disable sink on single failure, but log for monitoring
