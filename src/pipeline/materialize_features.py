"""Feature materialization pipeline using SQL window functions.

This module provides SQL-based feature engineering that ensures Point-in-Time
correctness by using window functions that only look at past data (no future leakage).

Supports two materialization modes (via FEATURE_MATERIALIZATION_MODE env var):
- "legacy" (default): Offset-based batch processing
- "cursor": Cursor-based pagination using record_id for better scalability
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from synthetic_pipeline.db import DatabaseSession
from synthetic_pipeline.logging import configure_logging, get_logger

MaterializationMode = Literal["legacy", "cursor"]


def get_materialization_mode() -> MaterializationMode:
    """Get the configured feature materialization mode.

    Returns:
        The mode from FEATURE_MATERIALIZATION_MODE env var, defaulting to "legacy".
    """
    mode = os.getenv("FEATURE_MATERIALIZATION_MODE", "legacy")
    if mode not in ("legacy", "cursor"):
        return "legacy"
    return mode  # type: ignore[return-value]


# SQL query for computing features with window functions
# Uses RANGE BETWEEN to ensure Point-in-Time correctness (no future data leakage)
FEATURE_ENGINEERING_SQL = """
WITH feature_calculations AS (
    SELECT
        gr.record_id,
        gr.user_id,
        gr.transaction_timestamp,
        gr.amount,
        gr.available_balance,

        -- Velocity 24h: Count of transactions in preceding 24-hour window
        -- Uses RANGE BETWEEN to include only past transactions up to current row
        COUNT(*) OVER (
            PARTITION BY gr.user_id
            ORDER BY gr.transaction_timestamp
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS velocity_24h,

        -- Amount to average ratio (30-day window)
        -- Current amount / rolling average of amounts in past 30 days
        CASE
            WHEN AVG(gr.amount) OVER (
                PARTITION BY gr.user_id
                ORDER BY gr.transaction_timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
            ) > 0
            THEN gr.amount / AVG(gr.amount) OVER (
                PARTITION BY gr.user_id
                ORDER BY gr.transaction_timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
            )
            ELSE 1.0
        END AS amount_to_avg_ratio_30d,

        -- Balance volatility z-score (30-day window)
        -- (current_balance - avg_balance) / stddev_balance
        -- Handle edge cases where stddev is 0 or NULL
        CASE
            WHEN COALESCE(STDDEV(gr.available_balance) OVER (
                PARTITION BY gr.user_id
                ORDER BY gr.transaction_timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
            ), 0) > 0
            THEN (
                gr.available_balance - AVG(gr.available_balance) OVER (
                    PARTITION BY gr.user_id
                    ORDER BY gr.transaction_timestamp
                    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
                )
            ) / STDDEV(gr.available_balance) OVER (
                PARTITION BY gr.user_id
                ORDER BY gr.transaction_timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
            )
            ELSE 0.0
        END AS balance_volatility_z_score,

        -- Additional signals for experimental_signals JSONB
        -- Transaction count in 7-day window
        COUNT(*) OVER (
            PARTITION BY gr.user_id
            ORDER BY gr.transaction_timestamp
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
        ) AS velocity_7d,

        -- Max amount in 30-day window (for detecting unusual spikes)
        MAX(gr.amount) OVER (
            PARTITION BY gr.user_id
            ORDER BY gr.transaction_timestamp
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
        ) AS max_amount_30d,

        -- Off-hours transaction count in 7 days
        SUM(CASE WHEN gr.is_off_hours_txn THEN 1 ELSE 0 END) OVER (
            PARTITION BY gr.user_id
            ORDER BY gr.transaction_timestamp
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
        ) AS off_hours_count_7d,

        -- Bank connection velocity (from source data)
        gr.bank_connections_count_24h,
        gr.merchant_risk_score

    FROM generated_records gr
    WHERE gr.record_id NOT IN (SELECT record_id FROM feature_snapshots)
)
SELECT
    record_id,
    user_id,
    velocity_24h::INTEGER,
    amount_to_avg_ratio_30d::FLOAT,
    balance_volatility_z_score::FLOAT,
    -- Build experimental signals JSON
    jsonb_build_object(
        'velocity_7d', velocity_7d,
        'max_amount_30d', max_amount_30d::FLOAT,
        'off_hours_count_7d', off_hours_count_7d,
        'bank_connections_24h', bank_connections_count_24h,
        'merchant_risk_score', merchant_risk_score
    ) AS experimental_signals
FROM feature_calculations
ORDER BY user_id, record_id;
"""

# SQL for inserting computed features
INSERT_FEATURES_SQL = """
INSERT INTO feature_snapshots (
    record_id,
    user_id,
    velocity_24h,
    amount_to_avg_ratio_30d,
    balance_volatility_z_score,
    experimental_signals,
    computed_at
)
VALUES (
    :record_id,
    :user_id,
    :velocity_24h,
    :amount_to_avg_ratio_30d,
    :balance_volatility_z_score,
    :experimental_signals,
    :computed_at
)
ON CONFLICT (record_id) DO UPDATE SET
    velocity_24h = EXCLUDED.velocity_24h,
    amount_to_avg_ratio_30d = EXCLUDED.amount_to_avg_ratio_30d,
    balance_volatility_z_score = EXCLUDED.balance_volatility_z_score,
    experimental_signals = EXCLUDED.experimental_signals,
    computed_at = EXCLUDED.computed_at;
"""

# Batch insert SQL for better performance
BATCH_INSERT_SQL = """
INSERT INTO feature_snapshots (
    record_id,
    user_id,
    velocity_24h,
    amount_to_avg_ratio_30d,
    balance_volatility_z_score,
    experimental_signals,
    computed_at
)
SELECT
    fc.record_id,
    fc.user_id,
    fc.velocity_24h::INTEGER,
    fc.amount_to_avg_ratio_30d::FLOAT,
    fc.balance_volatility_z_score::FLOAT,
    fc.experimental_signals,
    NOW()
FROM (
    {feature_query}
) fc
ON CONFLICT (record_id) DO UPDATE SET
    velocity_24h = EXCLUDED.velocity_24h,
    amount_to_avg_ratio_30d = EXCLUDED.amount_to_avg_ratio_30d,
    balance_volatility_z_score = EXCLUDED.balance_volatility_z_score,
    experimental_signals = EXCLUDED.experimental_signals,
    computed_at = EXCLUDED.computed_at;
"""

# Cursor-based SQL for scalable pagination using record_id
# This approach uses keyset pagination instead of OFFSET for better performance
CURSOR_FEATURE_ENGINEERING_SQL = """
WITH pending_records AS (
    SELECT gr.record_id
    FROM generated_records gr
    WHERE gr.record_id NOT IN (SELECT record_id FROM feature_snapshots)
      AND gr.record_id > :cursor
    ORDER BY gr.record_id
    LIMIT :batch_size
),
feature_calculations AS (
    SELECT
        gr.record_id,
        gr.user_id,
        gr.transaction_timestamp,
        gr.amount,
        gr.available_balance,

        COUNT(*) OVER (
            PARTITION BY gr.user_id
            ORDER BY gr.transaction_timestamp
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS velocity_24h,

        CASE
            WHEN AVG(gr.amount) OVER (
                PARTITION BY gr.user_id
                ORDER BY gr.transaction_timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
            ) > 0
            THEN gr.amount / AVG(gr.amount) OVER (
                PARTITION BY gr.user_id
                ORDER BY gr.transaction_timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
            )
            ELSE 1.0
        END AS amount_to_avg_ratio_30d,

        CASE
            WHEN COALESCE(STDDEV(gr.available_balance) OVER (
                PARTITION BY gr.user_id
                ORDER BY gr.transaction_timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
            ), 0) > 0
            THEN (
                gr.available_balance - AVG(gr.available_balance) OVER (
                    PARTITION BY gr.user_id
                    ORDER BY gr.transaction_timestamp
                    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
                )
            ) / STDDEV(gr.available_balance) OVER (
                PARTITION BY gr.user_id
                ORDER BY gr.transaction_timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
            )
            ELSE 0.0
        END AS balance_volatility_z_score,

        COUNT(*) OVER (
            PARTITION BY gr.user_id
            ORDER BY gr.transaction_timestamp
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
        ) AS velocity_7d,

        MAX(gr.amount) OVER (
            PARTITION BY gr.user_id
            ORDER BY gr.transaction_timestamp
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
        ) AS max_amount_30d,

        SUM(CASE WHEN gr.is_off_hours_txn THEN 1 ELSE 0 END) OVER (
            PARTITION BY gr.user_id
            ORDER BY gr.transaction_timestamp
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
        ) AS off_hours_count_7d,

        gr.bank_connections_count_24h,
        gr.merchant_risk_score

    FROM generated_records gr
    WHERE gr.record_id IN (SELECT record_id FROM pending_records)
)
SELECT
    record_id,
    user_id,
    velocity_24h::INTEGER,
    amount_to_avg_ratio_30d::FLOAT,
    balance_volatility_z_score::FLOAT,
    jsonb_build_object(
        'velocity_7d', velocity_7d,
        'max_amount_30d', max_amount_30d::FLOAT,
        'off_hours_count_7d', off_hours_count_7d,
        'bank_connections_24h', bank_connections_count_24h,
        'merchant_risk_score', merchant_risk_score
    ) AS experimental_signals
FROM feature_calculations
ORDER BY record_id;
"""


class FeatureMaterializer:
    """Materializes features from generated_records into feature_snapshots.

    Uses SQL window functions to compute point-in-time correct features
    without future data leakage.

    Supports two modes:
    - "legacy": Offset-based batch processing (default)
    - "cursor": Cursor-based pagination for better scalability

    Attributes:
        db: DatabaseSession instance for database operations.
        log: Logger instance.
        mode: Materialization mode ("legacy" or "cursor").
    """

    def __init__(
        self,
        database_url: str | None = None,
        echo: bool = False,
        mode: MaterializationMode | None = None,
    ):
        """Initialize the feature materializer.

        Args:
            database_url: Database connection URL.
            echo: Whether to echo SQL statements.
            mode: Materialization mode. If None, uses FEATURE_MATERIALIZATION_MODE
                  env var, defaulting to "legacy".
        """
        self.db = DatabaseSession(database_url=database_url, echo=echo)
        self.log = get_logger("feature_materializer")
        self.mode = mode or get_materialization_mode()
        self.log.info(f"Feature materializer initialized with mode: {self.mode}")

    def create_table(self) -> None:
        """Create the feature_snapshots table if it doesn't exist."""
        self.db.create_tables()
        self.log.info("Feature snapshots table created/verified")

    def get_pending_record_count(self, session: Session) -> int:
        """Get count of records not yet in feature_snapshots.

        Args:
            session: SQLAlchemy session.

        Returns:
            Number of pending records.
        """
        result = session.execute(
            text("""
                SELECT COUNT(*)
                FROM generated_records gr
                WHERE gr.record_id NOT IN (
                    SELECT record_id FROM feature_snapshots
                )
            """)
        )
        return result.scalar() or 0

    def compute_features_batch(
        self,
        session: Session,
        batch_size: int = 1000,
    ) -> int:
        """Compute features for a batch of records.

        Args:
            session: SQLAlchemy session.
            batch_size: Maximum records to process.

        Returns:
            Number of features computed.
        """
        # Compute features using window functions
        result = session.execute(text(FEATURE_ENGINEERING_SQL))
        rows = result.fetchall()

        if not rows:
            return 0

        # Limit to batch size
        rows_to_process = rows[:batch_size]
        computed_at = datetime.utcnow()

        # Insert computed features
        for row in rows_to_process:
            (
                record_id,
                user_id,
                velocity_24h,
                amount_ratio,
                volatility_z,
                exp_signals,
            ) = row

            # Handle experimental_signals - ensure it's JSON string for binding
            exp_signals_json = json.dumps(dict(exp_signals)) if exp_signals else None

            session.execute(
                text(INSERT_FEATURES_SQL),
                {
                    "record_id": record_id,
                    "user_id": user_id,
                    "velocity_24h": velocity_24h,
                    "amount_to_avg_ratio_30d": float(amount_ratio),
                    "balance_volatility_z_score": float(volatility_z),
                    "experimental_signals": exp_signals_json,
                    "computed_at": computed_at,
                },
            )

        return len(rows_to_process)

    def compute_features_batch_cursor(
        self,
        session: Session,
        cursor: str,
        batch_size: int = 1000,
    ) -> tuple[int, str | None]:
        """Compute features for a batch of records using cursor pagination.

        Args:
            session: SQLAlchemy session.
            cursor: Last processed record_id (empty string for first batch).
            batch_size: Maximum records to process.

        Returns:
            Tuple of (records_processed, next_cursor).
            next_cursor is None when no more records.
        """
        result = session.execute(
            text(CURSOR_FEATURE_ENGINEERING_SQL),
            {"cursor": cursor, "batch_size": batch_size},
        )
        rows = result.fetchall()

        if not rows:
            return 0, None

        computed_at = datetime.utcnow()
        last_record_id = None

        for row in rows:
            (
                record_id,
                user_id,
                velocity_24h,
                amount_ratio,
                volatility_z,
                exp_signals,
            ) = row

            last_record_id = record_id
            exp_signals_json = json.dumps(dict(exp_signals)) if exp_signals else None

            session.execute(
                text(INSERT_FEATURES_SQL),
                {
                    "record_id": record_id,
                    "user_id": user_id,
                    "velocity_24h": velocity_24h,
                    "amount_to_avg_ratio_30d": float(amount_ratio),
                    "balance_volatility_z_score": float(volatility_z),
                    "experimental_signals": exp_signals_json,
                    "computed_at": computed_at,
                },
            )

        return len(rows), last_record_id

    def materialize_all(
        self,
        batch_size: int = 1000,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Materialize features for all pending records.

        Uses the mode specified at initialization (legacy or cursor).

        Args:
            batch_size: Records per batch.
            max_batches: Maximum number of batches (None for unlimited).

        Returns:
            Statistics dict with total_processed, batches, duration, mode.
        """
        if self.mode == "cursor":
            return self._materialize_cursor_mode(batch_size, max_batches)
        else:
            return self._materialize_legacy_mode(batch_size, max_batches)

    def _materialize_legacy_mode(
        self,
        batch_size: int,
        max_batches: int | None,
    ) -> dict[str, Any]:
        """Legacy offset-based materialization (default behavior)."""
        start_time = datetime.utcnow()
        total_processed = 0
        batch_count = 0

        with self.db.get_session() as session:
            pending = self.get_pending_record_count(session)
            self.log.info(
                "Starting feature materialization (legacy mode)",
                pending_records=pending,
            )

            while True:
                if max_batches is not None and batch_count >= max_batches:
                    self.log.info("Max batches reached", max_batches=max_batches)
                    break

                processed = self.compute_features_batch(session, batch_size)

                if processed == 0:
                    break

                total_processed += processed
                batch_count += 1

                self.log.info(
                    "Batch completed",
                    batch=batch_count,
                    processed=processed,
                    total=total_processed,
                )

            session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()

        stats = {
            "total_processed": total_processed,
            "batches": batch_count,
            "duration_seconds": duration,
            "records_per_second": total_processed / duration if duration > 0 else 0,
            "mode": "legacy",
        }

        self.log.info("Feature materialization complete", **stats)
        return stats

    def _materialize_cursor_mode(
        self,
        batch_size: int,
        max_batches: int | None,
    ) -> dict[str, Any]:
        """Cursor-based materialization for better scalability."""
        start_time = datetime.utcnow()
        total_processed = 0
        batch_count = 0
        cursor = ""  # Start from beginning

        with self.db.get_session() as session:
            pending = self.get_pending_record_count(session)
            self.log.info(
                "Starting feature materialization (cursor mode)",
                pending_records=pending,
            )

            while True:
                if max_batches is not None and batch_count >= max_batches:
                    self.log.info("Max batches reached", max_batches=max_batches)
                    break

                processed, next_cursor = self.compute_features_batch_cursor(
                    session, cursor, batch_size
                )

                if processed == 0 or next_cursor is None:
                    break

                cursor = next_cursor
                total_processed += processed
                batch_count += 1

                self.log.info(
                    "Batch completed",
                    batch=batch_count,
                    processed=processed,
                    total=total_processed,
                    cursor=cursor[:20] + "..." if len(cursor) > 20 else cursor,
                )

            session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()

        stats = {
            "total_processed": total_processed,
            "batches": batch_count,
            "duration_seconds": duration,
            "records_per_second": total_processed / duration if duration > 0 else 0,
            "mode": "cursor",
        }

        self.log.info("Feature materialization complete", **stats)
        return stats

    def get_feature_stats(self, session: Session) -> dict[str, Any]:
        """Get statistics about feature snapshots.

        Args:
            session: SQLAlchemy session.

        Returns:
            Statistics dict.
        """
        result = session.execute(
            text("""
                SELECT
                    COUNT(*) as total_snapshots,
                    AVG(velocity_24h) as avg_velocity_24h,
                    AVG(amount_to_avg_ratio_30d) as avg_amount_ratio,
                    AVG(balance_volatility_z_score) as avg_volatility_z,
                    MIN(computed_at) as oldest_snapshot,
                    MAX(computed_at) as newest_snapshot
                FROM feature_snapshots
            """)
        )
        row = result.fetchone()

        if row:
            return {
                "total_snapshots": row[0],
                "avg_velocity_24h": float(row[1]) if row[1] else 0,
                "avg_amount_ratio": float(row[2]) if row[2] else 0,
                "avg_volatility_z": float(row[3]) if row[3] else 0,
                "oldest_snapshot": row[4],
                "newest_snapshot": row[5],
            }
        return {}


def materialize_features(
    database_url: str | None = None,
    batch_size: int = 1000,
    max_batches: int | None = None,
    verbose: bool = False,
    mode: MaterializationMode | None = None,
) -> dict[str, Any]:
    """Convenience function to materialize all features.

    Args:
        database_url: Database connection URL.
        batch_size: Records per batch.
        max_batches: Maximum batches (None for unlimited).
        verbose: Enable verbose logging.
        mode: Materialization mode ("legacy" or "cursor"). If None, uses
              FEATURE_MATERIALIZATION_MODE env var, defaulting to "legacy".

    Returns:
        Statistics dict including mode used.
    """
    configure_logging(level="DEBUG" if verbose else "INFO")

    materializer = FeatureMaterializer(
        database_url=database_url,
        echo=verbose,
        mode=mode,
    )

    # Ensure table exists
    materializer.create_table()

    # Materialize features
    return materializer.materialize_all(
        batch_size=batch_size,
        max_batches=max_batches,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Materialize features from generated_records"
    )
    parser.add_argument(
        "--database-url",
        help="Database URL (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    stats = materialize_features(
        database_url=args.database_url,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        verbose=args.verbose,
    )

    print("\nFeature Materialization Complete:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Batches: {stats['batches']}")
    print(f"  Duration: {stats['duration_seconds']:.2f}s")
    print(f"  Rate: {stats['records_per_second']:.1f} records/sec")
