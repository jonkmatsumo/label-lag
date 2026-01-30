"""SQLAlchemy models mirroring Pydantic models."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Table,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

import enum

class RuleStatus(enum.Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    ACTIVE = "active"
    SHADOW = "shadow"
    DISABLED = "disabled"
    ARCHIVED = "archived"

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass

# Association table for PublishedRuleSet -> RuleVersion
published_ruleset_versions = Table(
    "published_ruleset_versions",
    Base.metadata,
    Column("ruleset_id", Integer, ForeignKey("published_rulesets.id"), primary_key=True),
    Column("version_id", Integer, ForeignKey("rule_versions.id"), primary_key=True),
)

class RuleDB(Base):
    """Logical rule identity."""
    __tablename__ = "rules"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    status: Mapped[RuleStatus] = mapped_column(Enum(RuleStatus), default=RuleStatus.DRAFT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("now()"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("now()"), onupdate=text("now()"), nullable=False)

    versions = relationship("RuleVersionDB", back_populates="rule", cascade="all, delete-orphan")

class RuleVersionDB(Base):
    """Immutable rule content version."""
    __tablename__ = "rule_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_id: Mapped[str] = mapped_column(String(100), ForeignKey("rules.id"), nullable=False, index=True)
    
    # Rule content (matches api.rules.Rule dataclass fields)
    field: Mapped[str] = mapped_column(String(100), nullable=False)
    op: Mapped[str] = mapped_column(String(20), nullable=False)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    severity: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)
    reason: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("now()"), nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    rule = relationship("RuleDB", back_populates="versions")

    __table_args__ = (
        Index("ix_rule_version_content", "rule_id", "content_hash"),
    )

class PublishedRuleSetDB(Base):
    """Snapshot of active rules used for inference."""
    __tablename__ = "published_rulesets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    published_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("now()"), nullable=False, index=True)
    published_by: Mapped[str] = mapped_column(String(100), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Many-to-many relationship to RuleVersionDB
    rule_versions = relationship("RuleVersionDB", secondary=published_ruleset_versions)


class GeneratedRecordDB(Base):
    """SQLAlchemy model for generated records."""

    __tablename__ = "generated_records"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # PII fields
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False)

    # Timestamps
    transaction_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_off_hours_txn: Mapped[bool] = mapped_column(Boolean, default=False)

    # Account snapshot
    available_balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    balance_to_transaction_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    # Behavior metrics
    avg_available_balance_30d: Mapped[Decimal] = mapped_column(
        Numeric(18, 2), nullable=False
    )
    balance_volatility_z_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Connection metrics
    bank_connections_count_24h: Mapped[int] = mapped_column(Integer, nullable=False)
    bank_connections_count_7d: Mapped[int] = mapped_column(Integer, nullable=False)
    bank_connections_avg_30d: Mapped[float] = mapped_column(Float, nullable=False)

    # Transaction evaluation
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    amount_to_avg_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    merchant_risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    is_returned: Mapped[bool] = mapped_column(Boolean, default=False)

    # Identity changes
    email_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    phone_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Labels
    is_fraudulent: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    fraud_type: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_fraud_type", "fraud_type"),
        Index("ix_transaction_timestamp", "transaction_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<GeneratedRecordDB(record_id={self.record_id}, "
            f"is_fraudulent={self.is_fraudulent})>"
        )


# Separate normalized tables for more complex queries


class AccountSnapshotRecord(Base):
    """Normalized account snapshot records."""

    __tablename__ = "account_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    available_balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    balance_to_transaction_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class BehaviorMetricsRecord(Base):
    """Normalized behavior metrics records."""

    __tablename__ = "behavior_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    avg_available_balance_30d: Mapped[Decimal] = mapped_column(
        Numeric(18, 2), nullable=False
    )
    balance_volatility_z_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_high_risk: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class ConnectionMetricsRecord(Base):
    """Normalized connection metrics records."""

    __tablename__ = "connection_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    bank_connections_count_24h: Mapped[int] = mapped_column(Integer, nullable=False)
    bank_connections_count_7d: Mapped[int] = mapped_column(Integer, nullable=False)
    bank_connections_avg_30d: Mapped[float] = mapped_column(Float, nullable=False)
    is_24h_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    is_7d_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class TransactionEvaluationRecord(Base):
    """Normalized transaction evaluation records."""

    __tablename__ = "transaction_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    amount_to_avg_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    merchant_risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    is_returned: Mapped[bool] = mapped_column(Boolean, default=False)
    is_amount_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    is_merchant_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class IdentityChangeRecord(Base):
    """Normalized identity change records."""

    __tablename__ = "identity_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    email_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    phone_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    has_recent_change: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class EvaluationMetadataDB(Base):
    """Evaluation metadata for model training/testing splits.

    This table tracks temporal relationships between transactions and fraud events.
    Used for evaluation only - should NOT be used as training features.
    """

    __tablename__ = "evaluation_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    record_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Sequence tracking
    sequence_number: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Transaction order for this user (1-indexed)"
    )

    # Fraud timing
    fraud_confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="When fraud was confirmed for this user"
    )

    # Pre/post fraud flags
    is_pre_fraud: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="Transaction occurred before fraud detection",
    )

    days_to_fraud: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Days until fraud event (negative if after)",
    )

    # Training eligibility
    is_train_eligible: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        index=True,
        comment="Can be used for training (False for post-fraud records)",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_eval_user_sequence", "user_id", "sequence_number"),
        Index("ix_eval_train_eligible", "is_train_eligible", "is_pre_fraud"),
    )


class FeatureSnapshotDB(Base):
    """Feature store snapshot for ML training.

    Contains point-in-time correct features computed from generated_records
    using window functions. No future data leakage.
    """

    __tablename__ = "feature_snapshots"

    # Primary key
    snapshot_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    # Foreign key to generated_records
    record_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("generated_records.record_id"),
        nullable=False,
        unique=True,
        index=True,
    )

    # User identifier for partitioning
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Velocity feature: transaction count in 24h window
    velocity_24h: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="COUNT(*) of transactions in 24h window for user",
    )

    # Amount ratio feature: current amount / 30-day rolling average
    amount_to_avg_ratio_30d: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Current amount / AVG(amount) over 30d window",
    )

    # Balance volatility z-score: (balance - avg) / stddev over 30d
    balance_volatility_z_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="(balance - AVG(balance)) / STDDEV(balance) over 30d",
    )

    # Flexible JSONB column for experimental signals
    experimental_signals: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="Flexible JSON for experimental features (device_trust_score, etc.)",
    )

    # Metadata
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_feature_user_id", "user_id"),
        Index("ix_feature_computed_at", "computed_at"),
    )

class FeatureMaterializationStateDB(Base):
    """Tracks incremental materialization progress."""
    __tablename__ = "feature_materialization_state"

    feature_set: Mapped[str] = mapped_column(String(100), primary_key=True)
    last_processed_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("now()"), onupdate=text("now()"), nullable=False)
    schema_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
