"""Data service layer for the dashboard.

Provides read-only database access and API client for the Streamlit UI.
This module is isolated from backend dependencies - uses raw SQL queries
instead of ORM models to avoid coupling.
"""

import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Configuration from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://synthetic:synthetic_dev_password@localhost:5432/synthetic_data",
)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# API timeout in seconds
API_TIMEOUT = 2.0

# Risk score threshold for alerts
ALERT_THRESHOLD = 80

# Module-level engine (lazy initialization)
_engine: Engine | None = None


def get_db_engine() -> Engine:
    """Get or create the database engine.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_size=3,
            max_overflow=5,
            pool_pre_ping=True,
        )
    return _engine


@contextmanager
def get_db_connection():
    """Get a database connection context manager.

    Yields:
        Database connection that auto-closes on exit.

    Example:
        with get_db_connection() as conn:
            result = conn.execute(text("SELECT 1"))
    """
    engine = get_db_engine()
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()


def fetch_daily_stats(days: int = 30) -> pd.DataFrame:
    """Fetch daily transaction statistics.

    Joins evaluation_metadata with feature_snapshots and generated_records
    to get daily aggregates of transactions, fraud rates, and amounts.

    Args:
        days: Number of days to look back. Default 30.

    Returns:
        DataFrame with columns:
        - date: Transaction date
        - total_transactions: Count of transactions
        - fraud_count: Count of fraudulent transactions
        - fraud_rate: Percentage of fraud
        - total_amount: Sum of transaction amounts
        - avg_risk_score: Average risk score (from balance_volatility_z_score)
    """
    cutoff_date = datetime.now(UTC) - timedelta(days=days)

    query = text("""
        SELECT
            DATE(em.created_at) as date,
            COUNT(*) as total_transactions,
            SUM(CASE WHEN gr.is_fraudulent THEN 1 ELSE 0 END) as fraud_count,
            ROUND(
                100.0 * SUM(CASE WHEN gr.is_fraudulent THEN 1 ELSE 0 END) / COUNT(*),
                2
            ) as fraud_rate,
            COALESCE(SUM(gr.amount), 0) as total_amount,
            ROUND(AVG(fs.balance_volatility_z_score)::numeric, 2) as avg_z_score
        FROM evaluation_metadata em
        LEFT JOIN generated_records gr ON em.record_id = gr.record_id
        LEFT JOIN feature_snapshots fs ON em.record_id = fs.record_id
        WHERE em.created_at >= :cutoff_date
        GROUP BY DATE(em.created_at)
        ORDER BY date DESC
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query, {"cutoff_date": cutoff_date})
            rows = result.fetchall()
            columns = result.keys()
            return pd.DataFrame(rows, columns=list(columns))
    except SQLAlchemyError as e:
        print(f"Database error in fetch_daily_stats: {e}")
        return pd.DataFrame()


def fetch_transaction_details(days: int = 7) -> pd.DataFrame:
    """Fetch individual transaction details.

    Joins evaluation_metadata with feature_snapshots and generated_records
    to get transaction-level data for analysis.

    Args:
        days: Number of days to look back. Default 7.

    Returns:
        DataFrame with transaction details including features and labels.
    """
    cutoff_date = datetime.now(UTC) - timedelta(days=days)

    query = text("""
        SELECT
            em.record_id,
            em.user_id,
            em.created_at,
            em.is_train_eligible,
            em.is_pre_fraud,
            gr.amount,
            gr.is_fraudulent,
            gr.fraud_type,
            gr.is_off_hours_txn,
            gr.merchant_risk_score,
            fs.velocity_24h,
            fs.amount_to_avg_ratio_30d,
            fs.balance_volatility_z_score
        FROM evaluation_metadata em
        LEFT JOIN generated_records gr ON em.record_id = gr.record_id
        LEFT JOIN feature_snapshots fs ON em.record_id = fs.record_id
        WHERE em.created_at >= :cutoff_date
        ORDER BY em.created_at DESC
        LIMIT 1000
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query, {"cutoff_date": cutoff_date})
            rows = result.fetchall()
            columns = result.keys()
            return pd.DataFrame(rows, columns=list(columns))
    except SQLAlchemyError as e:
        print(f"Database error in fetch_transaction_details: {e}")
        return pd.DataFrame()


def fetch_recent_alerts(limit: int = 50) -> pd.DataFrame:
    """Fetch recent high-risk transactions.

    Selects records where computed risk indicators exceed thresholds,
    simulating alerts that would be generated by a real system.

    High risk is determined by:
    - High velocity (velocity_24h > 5)
    - Unusual amount ratio (amount_to_avg_ratio_30d > 3.0)
    - Low balance volatility z-score (< -2.0)
    - High merchant risk score (> 70)

    Args:
        limit: Maximum number of alerts to return. Default 50.

    Returns:
        DataFrame with high-risk transaction details.
    """
    query = text("""
        SELECT
            em.record_id,
            em.user_id,
            em.created_at,
            gr.amount,
            gr.is_fraudulent,
            gr.fraud_type,
            gr.merchant_risk_score,
            fs.velocity_24h,
            fs.amount_to_avg_ratio_30d,
            fs.balance_volatility_z_score,
            -- Compute a simple risk score for display
            CASE
                WHEN fs.velocity_24h > 5 THEN 20 ELSE 0
            END +
            CASE
                WHEN fs.amount_to_avg_ratio_30d > 3.0 THEN 25 ELSE 0
            END +
            CASE
                WHEN fs.balance_volatility_z_score < -2.0 THEN 20 ELSE 0
            END +
            CASE
                WHEN gr.merchant_risk_score > 70 THEN 20 ELSE 0
            END +
            CASE
                WHEN gr.is_off_hours_txn THEN 15 ELSE 0
            END as computed_risk_score
        FROM evaluation_metadata em
        INNER JOIN generated_records gr ON em.record_id = gr.record_id
        INNER JOIN feature_snapshots fs ON em.record_id = fs.record_id
        WHERE
            fs.velocity_24h > 5
            OR fs.amount_to_avg_ratio_30d > 3.0
            OR fs.balance_volatility_z_score < -2.0
            OR gr.merchant_risk_score > 70
        ORDER BY em.created_at DESC
        LIMIT :limit
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query, {"limit": limit})
            rows = result.fetchall()
            columns = result.keys()
            df = pd.DataFrame(rows, columns=list(columns))

            # Filter to only high-risk (score > threshold)
            if len(df) > 0 and "computed_risk_score" in df.columns:
                df = df[df["computed_risk_score"] >= ALERT_THRESHOLD]

            return df
    except SQLAlchemyError as e:
        print(f"Database error in fetch_recent_alerts: {e}")
        return pd.DataFrame()


def fetch_fraud_summary() -> dict[str, Any]:
    """Fetch summary statistics for fraud metrics.

    Returns:
        Dictionary with summary statistics:
        - total_transactions: Total transaction count
        - total_fraud: Total fraud count
        - fraud_rate: Overall fraud rate percentage
        - total_amount: Sum of all amounts
        - fraud_amount: Sum of fraudulent amounts
    """
    query = text("""
        SELECT
            COUNT(*) as total_transactions,
            SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) as total_fraud,
            ROUND(
                100.0 * SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) /
                NULLIF(COUNT(*), 0),
                2
            ) as fraud_rate,
            COALESCE(SUM(amount), 0) as total_amount,
            COALESCE(
                SUM(CASE WHEN is_fraudulent THEN amount ELSE 0 END),
                0
            ) as fraud_amount
        FROM generated_records
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if row:
                return {
                    "total_transactions": row.total_transactions or 0,
                    "total_fraud": row.total_fraud or 0,
                    "fraud_rate": float(row.fraud_rate or 0),
                    "total_amount": float(row.total_amount or 0),
                    "fraud_amount": float(row.fraud_amount or 0),
                }
    except SQLAlchemyError as e:
        print(f"Database error in fetch_fraud_summary: {e}")

    return {
        "total_transactions": 0,
        "total_fraud": 0,
        "fraud_rate": 0.0,
        "total_amount": 0.0,
        "fraud_amount": 0.0,
    }


# --- Pure Helper Utilities ---


def compute_sample_fraction(total_rows: int, sample_size: int) -> float:
    """Compute the fraction of rows to sample.

    Args:
        total_rows: Total number of rows in the dataset.
        sample_size: Desired sample size.

    Returns:
        Fraction between 0.0 and 1.0. Returns 0.0 if total_rows is 0.
        Returns 1.0 if sample_size >= total_rows.
    """
    if total_rows == 0:
        return 0.0
    return min(1.0, sample_size / total_rows)


def split_stratified_counts(
    total: int,
    fraud_rate: float,
    sample_size: int,
    min_per_class: int,
) -> tuple[int, int]:
    """Split sample size into fraud and non-fraud counts maintaining ratio.

    Args:
        total: Total number of records.
        fraud_rate: Fraction of records that are fraudulent (0.0 to 1.0).
        sample_size: Total desired sample size.
        min_per_class: Minimum samples required per class.

    Returns:
        Tuple of (fraud_sample_size, non_fraud_sample_size).
        Both values will be at least min_per_class if possible.
    """
    if total == 0:
        return (0, 0)

    fraud_count = int(total * fraud_rate)
    non_fraud_count = total - fraud_count

    # If dataset is too small for minimums, return what we can
    if total < min_per_class * 2:
        fraud_sample = min(fraud_count, sample_size // 2)
        non_fraud_sample = min(non_fraud_count, sample_size - fraud_sample)
        return (fraud_sample, non_fraud_sample)

    # Calculate proportional sample sizes
    if fraud_count > 0 and non_fraud_count > 0:
        fraud_fraction = fraud_count / total
        fraud_sample = int(sample_size * fraud_fraction)
        non_fraud_sample = sample_size - fraud_sample
    elif fraud_count > 0:
        # All fraud
        fraud_sample = min(fraud_count, sample_size)
        non_fraud_sample = 0
    else:
        # All non-fraud
        fraud_sample = 0
        non_fraud_sample = min(non_fraud_count, sample_size)

    # Enforce minimums
    if fraud_sample < min_per_class and fraud_count >= min_per_class:
        fraud_sample = min_per_class
        non_fraud_sample = min(non_fraud_count, sample_size - fraud_sample)

    if non_fraud_sample < min_per_class and non_fraud_count >= min_per_class:
        non_fraud_sample = min_per_class
        fraud_sample = min(fraud_count, sample_size - non_fraud_sample)

    # Ensure we don't exceed available counts
    fraud_sample = min(fraud_sample, fraud_count)
    non_fraud_sample = min(non_fraud_sample, non_fraud_count)

    return (fraud_sample, non_fraud_sample)


def normalize_schema_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema DataFrame column names and ordering.

    Args:
        df: DataFrame with schema information (from information_schema).

    Returns:
        DataFrame with normalized column names (lowercase) and consistent
        column ordering: table_name, column_name, data_type, is_nullable,
        ordinal_position.
    """
    if df.empty:
        return df

    # Normalize column names to lowercase
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    # Define expected column order
    expected_cols = [
        "table_name",
        "column_name",
        "data_type",
        "is_nullable",
        "ordinal_position",
    ]

    # Reorder columns if they exist
    available_cols = [col for col in expected_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in expected_cols]
    df = df[available_cols + other_cols]

    return df


# --- Dataset Primitives ---


@st.cache_data
def get_dataset_fingerprint() -> dict[str, Any]:
    """Get a lightweight fingerprint of the dataset for cache invalidation.

    Returns aggregate statistics from generated_records and feature_snapshots
    that change whenever the dataset changes. Used to key other cached functions.

    Returns:
        Dictionary with stable keys:
        - generated_records: dict with count, max_created_at,
          max_transaction_timestamp, max_id
        - feature_snapshots: dict with count, max_computed_at, max_snapshot_id
        All timestamp and ID fields are None if the table is empty.
    """
    fingerprint = {
        "generated_records": {
            "count": 0,
            "max_created_at": None,
            "max_transaction_timestamp": None,
            "max_id": None,
        },
        "feature_snapshots": {
            "count": 0,
            "max_computed_at": None,
            "max_snapshot_id": None,
        },
    }

    # Query generated_records aggregates
    query_gr = text("""
        SELECT
            COUNT(*) as count,
            MAX(created_at) as max_created_at,
            MAX(transaction_timestamp) as max_transaction_timestamp,
            MAX(id) as max_id
        FROM generated_records
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query_gr)
            row = result.fetchone()
            if row:
                fingerprint["generated_records"] = {
                    "count": row.count or 0,
                    "max_created_at": row.max_created_at,
                    "max_transaction_timestamp": row.max_transaction_timestamp,
                    "max_id": row.max_id,
                }
    except SQLAlchemyError as e:
        print(f"Database error in get_dataset_fingerprint (generated_records): {e}")

    # Query feature_snapshots aggregates
    query_fs = text("""
        SELECT
            COUNT(*) as count,
            MAX(computed_at) as max_computed_at,
            MAX(snapshot_id) as max_snapshot_id
        FROM feature_snapshots
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query_fs)
            row = result.fetchone()
            if row:
                fingerprint["feature_snapshots"] = {
                    "count": row.count or 0,
                    "max_computed_at": row.max_computed_at,
                    "max_snapshot_id": row.max_snapshot_id,
                }
    except SQLAlchemyError as e:
        print(f"Database error in get_dataset_fingerprint (feature_snapshots): {e}")

    return fingerprint


@st.cache_data
def _cached_overview_metrics(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Internal cached function for overview metrics.

    Args:
        fingerprint: Dataset fingerprint dict from get_dataset_fingerprint().

    Returns:
        Dictionary with overview metrics.
    """
    query = text("""
        SELECT
            COUNT(*) as total_records,
            SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) as fraud_records,
            COUNT(DISTINCT user_id) as unique_users,
            MIN(transaction_timestamp) as min_transaction_timestamp,
            MAX(transaction_timestamp) as max_transaction_timestamp,
            MIN(created_at) as min_created_at,
            MAX(created_at) as max_created_at
        FROM generated_records
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if row:
                total_records = row.total_records or 0
                fraud_records = row.fraud_records or 0
                fraud_rate = (
                    (fraud_records / total_records * 100.0)
                    if total_records > 0
                    else 0.0
                )

                return {
                    "total_records": total_records,
                    "fraud_records": fraud_records,
                    "fraud_rate": fraud_rate,
                    "unique_users": row.unique_users or 0,
                    "min_transaction_timestamp": row.min_transaction_timestamp,
                    "max_transaction_timestamp": row.max_transaction_timestamp,
                    "min_created_at": row.min_created_at,
                    "max_created_at": row.max_created_at,
                }
    except SQLAlchemyError as e:
        print(f"Database error in fetch_overview_metrics: {e}")

    return {
        "total_records": 0,
        "fraud_records": 0,
        "fraud_rate": 0.0,
        "unique_users": 0,
        "min_transaction_timestamp": None,
        "max_transaction_timestamp": None,
        "min_created_at": None,
        "max_created_at": None,
    }


def fetch_overview_metrics() -> dict[str, Any]:
    """Fetch dataset overview metrics.

    Returns aggregate statistics about the dataset including record counts,
    fraud rates, unique users, and timestamp ranges.

    Returns:
        Dictionary with keys:
        - total_records: Total number of records
        - fraud_records: Number of fraudulent records
        - fraud_rate: Fraud rate as percentage (0.0 to 100.0)
        - unique_users: Number of distinct users
        - min_transaction_timestamp: Earliest transaction timestamp
        - max_transaction_timestamp: Latest transaction timestamp
        - min_created_at: Earliest record creation timestamp
        - max_created_at: Latest record creation timestamp
    """
    fingerprint = get_dataset_fingerprint()
    return _cached_overview_metrics(fingerprint)


def fetch_schema_summary(
    table_names: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch schema summary from information_schema.

    Args:
        table_names: List of table names to query. Defaults to
            ["generated_records", "feature_snapshots"] if None.

    Returns:
        DataFrame with columns: table_name, column_name, data_type,
        is_nullable, ordinal_position. Column names are normalized to lowercase.
    """
    if table_names is None:
        table_names = ["generated_records", "feature_snapshots"]

    query = text("""
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = ANY(:table_names)
        ORDER BY table_name, ordinal_position
    """)

    try:
        with get_db_connection() as conn:
            result = conn.execute(query, {"table_names": table_names})
            rows = result.fetchall()
            columns = result.keys()
            df = pd.DataFrame(rows, columns=list(columns))
            return normalize_schema_df(df)
    except SQLAlchemyError as e:
        print(f"Database error in fetch_schema_summary: {e}")
        return pd.DataFrame()


def _get_postgres_version(conn) -> int:
    """Get PostgreSQL major version number.

    Args:
        conn: Database connection.

    Returns:
        Major version number (e.g., 16 for PostgreSQL 16.x).
    """
    try:
        result = conn.execute(text("SELECT version()"))
        version_str = result.scalar()
        # Extract major version from string like "PostgreSQL 16.1 ..."
        if version_str:
            parts = version_str.split()
            for i, part in enumerate(parts):
                if part.startswith("PostgreSQL"):
                    if i + 1 < len(parts):
                        version_num = parts[i + 1].split(".")[0]
                        return int(version_num)
        return 0
    except Exception:
        return 0


def _get_table_stats(conn) -> tuple[int, int, int]:
    """Get table statistics for sampling strategy selection.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (min_id, max_id, count).
    """
    query = text("""
        SELECT
            MIN(id) as min_id,
            MAX(id) as max_id,
            COUNT(*) as count
        FROM generated_records
    """)
    result = conn.execute(query)
    row = result.fetchone()
    if row:
        return (row.min_id or 0, row.max_id or 0, row.count or 0)
    return (0, 0, 0)


def _sample_generated_records(
    conn,
    sample_size: int,
    stratify: bool,
    fraud_rate: float,
    min_per_class: int,
) -> pd.DataFrame:
    """Sample records from generated_records using appropriate strategy.

    Args:
        conn: Database connection.
        sample_size: Desired sample size.
        stratify: Whether to stratify by fraud class.
        fraud_rate: Fraud rate for stratification.
        min_per_class: Minimum samples per class.

    Returns:
        DataFrame with sampled records.
    """
    min_id, max_id, total_count = _get_table_stats(conn)

    if total_count == 0:
        return pd.DataFrame()

    # Determine sampling strategy
    use_tablesample = False
    use_id_range = False
    use_offset = False

    pg_version = _get_postgres_version(conn)
    if pg_version >= 16 and total_count > 100000:
        use_tablesample = True
    elif max_id > min_id and total_count > 10000:
        use_id_range = True
    elif total_count < 100000:
        use_offset = True

    if stratify:
        fraud_sample_size, non_fraud_sample_size = split_stratified_counts(
            total_count, fraud_rate, sample_size, min_per_class
        )

        # Sample fraud records
        fraud_df = pd.DataFrame()
        try:
            if use_tablesample:
                fraction = compute_sample_fraction(total_count, fraud_sample_size)
                query = text(
                    f"SELECT * FROM generated_records "
                    f"TABLESAMPLE SYSTEM ({fraction * 100}) "
                    f"WHERE is_fraudulent = true LIMIT {fraud_sample_size}"
                )
            elif use_id_range and max_id > min_id:
                # Use ID range with fraud filter
                query = text(
                    f"SELECT * FROM generated_records "
                    f"WHERE id BETWEEN {min_id} AND {max_id} "
                    f"AND is_fraudulent = true "
                    f"LIMIT {fraud_sample_size}"
                )
            elif use_offset:
                query = text(
                    f"SELECT * FROM generated_records "
                    f"WHERE is_fraudulent = true "
                    f"LIMIT {fraud_sample_size} OFFSET 0"
                )
            else:
                query = text(
                    f"SELECT * FROM generated_records "
                    f"WHERE is_fraudulent = true "
                    f"ORDER BY random() LIMIT {fraud_sample_size}"
                )

            fraud_result = conn.execute(query)
            fraud_rows = fraud_result.fetchall()
            if fraud_rows:
                fraud_columns = fraud_result.keys()
                fraud_df = pd.DataFrame(fraud_rows, columns=list(fraud_columns))
                if len(fraud_df) > fraud_sample_size:
                    fraud_df = fraud_df.head(fraud_sample_size)
        except Exception as e:
            print(f"Error sampling fraud records: {e}")

        # Sample non-fraud records
        non_fraud_df = pd.DataFrame()
        try:
            if use_tablesample:
                fraction = compute_sample_fraction(total_count, non_fraud_sample_size)
                query = text(
                    f"SELECT * FROM generated_records "
                    f"TABLESAMPLE SYSTEM ({fraction * 100}) "
                    f"WHERE is_fraudulent = false LIMIT {non_fraud_sample_size}"
                )
            elif use_id_range and max_id > min_id:
                query = text(
                    f"SELECT * FROM generated_records "
                    f"WHERE id BETWEEN {min_id} AND {max_id} "
                    f"AND is_fraudulent = false "
                    f"LIMIT {non_fraud_sample_size}"
                )
            elif use_offset:
                query = text(
                    f"SELECT * FROM generated_records "
                    f"WHERE is_fraudulent = false "
                    f"LIMIT {non_fraud_sample_size} OFFSET 0"
                )
            else:
                query = text(
                    f"SELECT * FROM generated_records "
                    f"WHERE is_fraudulent = false "
                    f"ORDER BY random() LIMIT {non_fraud_sample_size}"
                )

            non_fraud_result = conn.execute(query)
            non_fraud_rows = non_fraud_result.fetchall()
            if non_fraud_rows:
                non_fraud_columns = non_fraud_result.keys()
                non_fraud_df = pd.DataFrame(
                    non_fraud_rows, columns=list(non_fraud_columns)
                )
                if len(non_fraud_df) > non_fraud_sample_size:
                    non_fraud_df = non_fraud_df.head(non_fraud_sample_size)
        except Exception as e:
            print(f"Error sampling non-fraud records: {e}")

        # Combine results
        if not fraud_df.empty and not non_fraud_df.empty:
            return pd.concat([fraud_df, non_fraud_df], ignore_index=True)
        elif not fraud_df.empty:
            return fraud_df
        elif not non_fraud_df.empty:
            return non_fraud_df
        else:
            return pd.DataFrame()
    else:
        # Non-stratified sampling
        try:
            if use_tablesample:
                fraction = compute_sample_fraction(total_count, sample_size)
                query = text(
                    f"SELECT * FROM generated_records "
                    f"TABLESAMPLE SYSTEM ({fraction * 100}) "
                    f"LIMIT {sample_size}"
                )
            elif use_id_range and max_id > min_id:
                # Sample IDs uniformly
                step = max(1, (max_id - min_id) // sample_size)
                query = text(
                    f"SELECT * FROM generated_records "
                    f"WHERE id IN (SELECT generate_series({min_id}, {max_id}, {step}) "
                    f"LIMIT {sample_size})"
                )
            elif use_offset:
                query = text(
                    f"SELECT * FROM generated_records LIMIT {sample_size} OFFSET 0"
                )
            else:
                query = text(
                    f"SELECT * FROM generated_records "
                    f"ORDER BY random() LIMIT {sample_size}"
                )

            result = conn.execute(query)
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                df = pd.DataFrame(rows, columns=list(columns))
                if len(df) > sample_size:
                    df = df.head(sample_size)
                return df
        except Exception as e:
            print(f"Error sampling records: {e}")

    return pd.DataFrame()


@st.cache_data
def _cached_feature_sample(
    fingerprint: dict[str, Any],
    sample_size: int,
    stratify: bool,
) -> pd.DataFrame:
    """Internal cached function for feature sampling.

    Args:
        fingerprint: Dataset fingerprint dict.
        sample_size: Desired sample size.
        stratify: Whether to stratify by fraud class.

    Returns:
        DataFrame with sampled features.
    """
    try:
        with get_db_connection() as conn:
            # Get fraud rate for stratification
            fraud_rate = 0.0
            if stratify:
                overview = fetch_overview_metrics()
                total = overview.get("total_records", 0)
                fraud = overview.get("fraud_records", 0)
                if total > 0:
                    fraud_rate = fraud / total

            # Sample from generated_records
            sampled_df = _sample_generated_records(
                conn, sample_size, stratify, fraud_rate, min_per_class=10
            )

            if sampled_df.empty:
                return pd.DataFrame()

            # Join to feature_snapshots
            record_ids = sampled_df["record_id"].tolist()
            if not record_ids:
                return pd.DataFrame()

            query = text("""
                SELECT
                    gr.record_id,
                    gr.is_fraudulent,
                    fs.velocity_24h,
                    fs.amount_to_avg_ratio_30d,
                    fs.balance_volatility_z_score
                FROM generated_records gr
                INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
                WHERE gr.record_id = ANY(:record_ids)
            """)

            result = conn.execute(query, {"record_ids": record_ids})
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                df = pd.DataFrame(rows, columns=list(columns))
                # Ensure we don't exceed sample_size
                if len(df) > sample_size:
                    df = df.head(sample_size)
                return df
    except SQLAlchemyError as e:
        print(f"Database error in fetch_feature_sample: {e}")

    return pd.DataFrame()


def fetch_feature_sample(
    sample_size: int,
    stratify: bool = True,
) -> pd.DataFrame:
    """Fetch a sampled feature frame for diagnostics and analysis.

    Returns a bounded-size DataFrame with numeric features and labels,
    optionally stratified by fraud class.

    Args:
        sample_size: Maximum number of rows to return.
        stratify: Whether to stratify sampling by fraud class. Default True.

    Returns:
        DataFrame with columns: record_id, is_fraudulent, velocity_24h,
        amount_to_avg_ratio_30d, balance_volatility_z_score.
        DataFrame will have at most sample_size rows.
    """
    fingerprint = get_dataset_fingerprint()
    return _cached_feature_sample(fingerprint, sample_size, stratify)


# --- API Client ---


def predict_risk(
    user_id: str,
    amount: float,
    currency: str = "USD",
    client_txn_id: str | None = None,
) -> dict[str, Any] | None:
    """Send a transaction for risk evaluation via the API.

    Args:
        user_id: User identifier.
        amount: Transaction amount.
        currency: Currency code (default: USD).
        client_txn_id: Optional client transaction ID. Auto-generated if None.

    Returns:
        API response dictionary with score and risk_components,
        or None if the request fails.

    Example response:
        {
            "request_id": "req_abc123",
            "score": 85,
            "risk_components": [
                {"key": "velocity", "label": "high_transaction_velocity"}
            ],
            "model_version": "v1.0.0"
        }
    """
    if client_txn_id is None:
        client_txn_id = f"ui_txn_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

    url = f"{API_BASE_URL}/evaluate/signal"
    payload = {
        "user_id": user_id,
        "amount": amount,
        "currency": currency,
        "client_transaction_id": client_txn_id,
    }

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        print(f"API timeout after {API_TIMEOUT}s for user {user_id}")
        return None
    except requests.ConnectionError:
        print(f"API connection error: Could not connect to {url}")
        return None
    except requests.HTTPError as e:
        print(f"API HTTP error: {e}")
        return None
    except requests.RequestException as e:
        print(f"API request error: {e}")
        return None


def check_api_health() -> dict[str, Any] | None:
    """Check the API health status.

    Returns:
        Health check response or None if unavailable.
    """
    url = f"{API_BASE_URL}/health"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


# =============================================================================
# Rule Inspector API Clients (Phase 1)
# =============================================================================


def fetch_rules() -> dict[str, Any] | None:
    """Fetch the current production ruleset.

    Returns:
        Dict with version and rules, or None if unavailable.
    """
    url = f"{API_BASE_URL}/rules"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching rules: {e}")
        return None


def sandbox_evaluate(
    features: dict[str, Any],
    base_score: int = 50,
    ruleset: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate rules in sandbox mode.

    Args:
        features: Dict of feature values.
        base_score: Base score before rule application.
        ruleset: Optional custom ruleset dict.

    Returns:
        Evaluation result dict or None if request failed.
    """
    url = f"{API_BASE_URL}/rules/sandbox/evaluate"

    payload = {
        "features": features,
        "base_score": base_score,
    }
    if ruleset is not None:
        payload["ruleset"] = ruleset

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT * 2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error in sandbox evaluation: {e}")
        return None


def fetch_shadow_comparison(
    start_date: str,
    end_date: str,
    rule_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Fetch shadow mode comparison metrics.

    Args:
        start_date: Start date (ISO format).
        end_date: End date (ISO format).
        rule_ids: Optional list of rule IDs to filter.

    Returns:
        Comparison report dict or None if unavailable.
    """
    url = f"{API_BASE_URL}/metrics/shadow/comparison"

    params = {
        "start_date": start_date,
        "end_date": end_date,
    }
    if rule_ids:
        params["rule_ids"] = ",".join(rule_ids)

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT * 2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching shadow comparison: {e}")
        return None


def fetch_backtest_results(
    rule_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
) -> dict[str, Any] | None:
    """Fetch backtest results list.

    Args:
        rule_id: Optional rule ID filter.
        start_date: Optional start date filter (ISO format).
        end_date: Optional end date filter (ISO format).
        limit: Maximum results to return.

    Returns:
        Dict with results list or None if unavailable.
    """
    url = f"{API_BASE_URL}/backtest/results"

    params = {"limit": limit}
    if rule_id:
        params["rule_id"] = rule_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT * 2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching backtest results: {e}")
        return None


def fetch_backtest_result(job_id: str) -> dict[str, Any] | None:
    """Fetch a specific backtest result.

    Args:
        job_id: Backtest job identifier.

    Returns:
        Backtest result dict or None if not found.
    """
    url = f"{API_BASE_URL}/backtest/results/{job_id}"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching backtest result {job_id}: {e}")
        return None


def fetch_draft_rules(
    status: str | None = None,
) -> dict[str, Any] | None:
    """Fetch draft rules.

    Args:
        status: Optional status filter (draft, pending_review, approved, etc.).

    Returns:
        Dict with rules list or None if unavailable.
    """
    url = f"{API_BASE_URL}/rules/draft"

    params = {}
    if status:
        params["status"] = status

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching draft rules: {e}")
        return None


def publish_rule(
    rule_id: str, actor: str, reason: str | None = None
) -> dict[str, Any] | None:
    """Publish an approved rule to production.

    Args:
        rule_id: Rule identifier.
        actor: Who is publishing the rule.
        reason: Optional reason for publishing.

    Returns:
        Publish response dict or None if request failed.
    """
    url = f"{API_BASE_URL}/rules/{rule_id}/publish"

    payload = {"actor": actor}
    if reason:
        payload["reason"] = reason

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error publishing rule: {e}")
        return None


def fetch_approval_signals(rule_id: str) -> dict[str, Any] | None:
    """Fetch approval quality signals for a rule.

    Args:
        rule_id: Rule identifier.

    Returns:
        Dict with signals data or None if unavailable.
    """
    url = f"{API_BASE_URL}/rules/draft/{rule_id}/signals"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching approval signals for {rule_id}: {e}")
        return None


def fetch_heuristic_suggestions(
    field: str | None = None,
    min_confidence: float = 0.7,
    min_samples: int = 100,
) -> dict[str, Any] | None:
    """Fetch heuristic rule suggestions.

    Args:
        field: Optional feature field to filter.
        min_confidence: Minimum confidence threshold.
        min_samples: Minimum samples required for analysis.

    Returns:
        Dict with suggestions list or None if unavailable.
    """
    url = f"{API_BASE_URL}/suggestions/heuristic"

    params = {
        "min_confidence": min_confidence,
        "min_samples": min_samples,
    }
    if field:
        params["field"] = field

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT * 3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching suggestions: {e}")
        return None


def fetch_drift_status(hours: int = 24) -> dict[str, Any] | None:
    """Fetch drift status from the API.

    Args:
        hours: Hours of live data to analyze.

    Returns:
        Drift status dict or None if unavailable.
    """
    url = f"{API_BASE_URL}/monitoring/drift"
    params = {"hours": hours}

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT * 3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching drift status: {e}")
        return None


@st.cache_data(ttl=60)
def _cached_fetch_drift_status(hours: int = 24) -> dict[str, Any] | None:
    """Cached wrapper for fetch_drift_status to avoid redundant API calls.

    Args:
        hours: Hours of live data to analyze.

    Returns:
        Drift status dict or None if unavailable.
    """
    return fetch_drift_status(hours)


# =============================================================================
# Rule Version Diff Functions
# =============================================================================


def fetch_rule_versions(rule_id: str) -> list[dict[str, Any]] | None:
    """Fetch all versions of a rule.

    Args:
        rule_id: Rule identifier.

    Returns:
        List of version dicts or None if request failed.
    """
    url = f"{API_BASE_URL}/rules/{rule_id}/versions"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("versions", [])
    except requests.RequestException as e:
        print(f"Error fetching versions for rule {rule_id}: {e}")
        return None


def fetch_rule_diff(
    rule_id: str,
    version_a: str | None = None,
    version_b: str | None = None,
) -> dict[str, Any] | None:
    """Fetch diff between two rule versions.

    Args:
        rule_id: Rule identifier.
        version_a: Newer version ID (optional, defaults to latest).
        version_b: Older version ID (optional, defaults to predecessor).

    Returns:
        Diff result dict or None if request failed.
    """
    url = f"{API_BASE_URL}/rules/{rule_id}/diff"

    params = {}
    if version_a:
        params["version_a"] = version_a
    if version_b:
        params["version_b"] = version_b

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching diff for rule {rule_id}: {e}")
        return None


def run_backtest(
    ruleset_version: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    rule_id: str | None = None,
) -> dict[str, Any] | None:
    """Run a backtest via API.

    Args:
        ruleset_version: Ruleset version to test.
        start_date: Start date (ISO format).
        end_date: End date (ISO format).
        rule_id: Optional rule identifier (test single rule).

    Returns:
        Backtest result dict or None if request failed.
    """
    url = f"{API_BASE_URL}/backtest/run"

    payload = {
        "ruleset_version": ruleset_version,
        "start_date": start_date,
        "end_date": end_date,
        "rule_id": rule_id,
    }

    try:
        # Long timeout for backtests
        response = requests.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error running backtest: {e}")
        return None


def compare_backtests(
    base_version: str | None,
    candidate_version: str,
    start_date: str,
    end_date: str,
    rule_id: str | None = None,
) -> dict[str, Any] | None:
    """Compare two backtests (what-if simulation).

    Args:
        base_version: Baseline version (None = production).
        candidate_version: Candidate version to compare.
        start_date: Start date (ISO format).
        end_date: End date (ISO format).
        rule_id: Optional rule identifier.

    Returns:
        Comparison result dict with deltas or None if request failed.
    """
    url = f"{API_BASE_URL}/backtest/compare"

    payload = {
        "base_version": base_version,
        "candidate_version": candidate_version,
        "start_date": start_date,
        "end_date": end_date,
        "rule_id": rule_id,
    }

    try:
        # Long timeout for running two backtests
        response = requests.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error comparing backtests: {e}")
        return None


def get_rule_analytics(rule_id: str, days: int = 7) -> dict[str, Any] | None:
    """Fetch analytics for a single rule.

    Args:
        rule_id: Rule ID.
        days: Number of days to look back.

    Returns:
        Dict with rule analytics or None if failed.
    """
    url = f"{API_BASE_URL}/analytics/rules/{rule_id}"
    params = {"days": days}

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching analytics for rule {rule_id}: {e}")
        return None


def get_readiness_report(rule_id: str) -> dict[str, Any] | None:
    """Fetch readiness report for a rule.

    Args:
        rule_id: Rule ID.

    Returns:
        Readiness report dict or None if failed.
    """
    url = f"{API_BASE_URL}/rules/{rule_id}/readiness"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching readiness report for {rule_id}: {e}")
        return None


def get_rule_attribution(rule_id: str, days: int = 7) -> dict[str, Any] | None:
    """Fetch attribution metrics for a rule.

    Args:
        rule_id: Rule ID.
        days: Days lookback.

    Returns:
        Attribution dict or None.
    """
    url = f"{API_BASE_URL}/analytics/attribution"
    params = {"rule_id": rule_id, "days": days}

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching attribution for {rule_id}: {e}")
        return None
