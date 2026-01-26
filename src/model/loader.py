"""Data loader for XGBoost training with temporal splitting."""

import hashlib
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.schemas import SplitConfig
from synthetic_pipeline.db.session import DatabaseSession


@dataclass
class TrainTestSplit:
    """Container for train/test split data."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    split_manifest: dict | None = None

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    @property
    def train_fraud_rate(self) -> float:
        if len(self.y_train) == 0:
            return 0.0
        return self.y_train.mean()

    @property
    def test_fraud_rate(self) -> float:
        if len(self.y_test) == 0:
            return 0.0
        return self.y_test.mean()


class DataLoader:
    """Loads and prepares data for XGBoost training with strict temporal splitting.

    Implements two key concepts:
    1. Strict Temporal Splitting: Train on data before cutoff, test on data after.
    2. Label Maturity (Knowledge Horizon): In training set, only label fraud if
       fraud_confirmed_at <= cutoff. This simulates not knowing about fraud
       that hasn't been detected yet.
    """

    # Feature columns from feature_snapshots table
    FEATURE_COLUMNS = [
        "velocity_24h",
        "amount_to_avg_ratio_30d",
        "balance_volatility_z_score",
    ]

    # Columns that should not be used as training features
    DEFAULT_NON_FEATURE_COLUMNS = [
        "record_id",
        "snapshot_id",
        "computed_at",
        "user_id",
    ]

    def __init__(self, database_url: str | None = None):
        """Initialize DataLoader.

        Args:
            database_url: Database connection URL. Defaults to env vars.
        """
        self.db_session = DatabaseSession(database_url=database_url)

    def load_train_test_split(
        self,
        training_cutoff_date: str | datetime,
        session: Session | None = None,
        feature_columns: list[str] | None = None,
        split_config: SplitConfig | None = None,
    ) -> TrainTestSplit:
        """Load train/test split with temporal splitting and label maturity.

        Args:
            training_cutoff_date: Cutoff date for train/test split (e.g., '2024-04-01').
                Records with created_at < cutoff go to train, >= cutoff go to test.
            session: Optional existing database session.
            feature_columns: Optional list of feature columns to use. If None, uses
                default FEATURE_COLUMNS.
            split_config: Optional split/CV config. When provided, split_manifest
                is populated on the returned TrainTestSplit.

        Returns:
            TrainTestSplit containing X_train, y_train, X_test, y_test, and
            optionally split_manifest.

        Raises:
            ValueError: If any requested feature columns are missing from the data.
        """
        if isinstance(training_cutoff_date, str):
            cutoff = datetime.fromisoformat(training_cutoff_date)
        else:
            cutoff = training_cutoff_date

        if feature_columns is None:
            feature_columns = self.FEATURE_COLUMNS

        if session is not None:
            return self._load_with_session(
                session, cutoff, feature_columns, split_config
            )

        with self.db_session.get_session() as session:
            return self._load_with_session(
                session, cutoff, feature_columns, split_config
            )

    def _load_with_session(
        self,
        session: Session,
        cutoff: datetime,
        feature_columns: list[str],
        split_config: SplitConfig | None = None,
    ) -> TrainTestSplit:
        """Load data using provided session.

        Args:
            session: Database session.
            cutoff: Training cutoff date.
            feature_columns: List of feature columns to extract.
            split_config: Optional split config for manifest generation.

        Returns:
            TrainTestSplit with selected features and optionally split_manifest.

        Raises:
            ValueError: If any requested feature columns are missing from the data.
        """
        train_df = self._load_train_set(session, cutoff)
        test_df = self._load_test_set(session, cutoff)

        all_columns = set()
        if len(train_df) > 0:
            all_columns.update(train_df.columns)
        if len(test_df) > 0:
            all_columns.update(test_df.columns)

        if len(all_columns) > 0:
            missing_columns = [col for col in feature_columns if col not in all_columns]
            if missing_columns:
                raise ValueError(
                    f"Requested feature columns not found in data: {missing_columns}. "
                    f"Available columns: {sorted(all_columns)}"
                )

        if len(train_df) > 0:
            features_train = train_df[feature_columns]
            labels_train = train_df["label"]
        else:
            features_train = pd.DataFrame(columns=feature_columns)
            labels_train = pd.Series(dtype=int)

        if len(test_df) > 0:
            features_test = test_df[feature_columns]
            labels_test = test_df["label"]
        else:
            features_test = pd.DataFrame(columns=feature_columns)
            labels_test = pd.Series(dtype=int)

        manifest: dict | None = None
        if split_config is not None and "record_id" in all_columns:
            train_ids = (
                train_df["record_id"].astype(str).tolist() if len(train_df) > 0 else []
            )
            test_ids = (
                test_df["record_id"].astype(str).tolist() if len(test_df) > 0 else []
            )
            s = split_config.strategy

            # Compute time ranges
            train_time_range = None
            test_time_range = None
            if len(train_df) > 0 and "transaction_timestamp" in train_df.columns:
                train_timestamps = pd.to_datetime(train_df["transaction_timestamp"])
                train_time_range = {
                    "min": train_timestamps.min().isoformat(),
                    "max": train_timestamps.max().isoformat(),
                }
            if len(test_df) > 0 and "transaction_timestamp" in test_df.columns:
                test_timestamps = pd.to_datetime(test_df["transaction_timestamp"])
                test_time_range = {
                    "min": test_timestamps.min().isoformat(),
                    "max": test_timestamps.max().isoformat(),
                }

            # Compute unique user counts
            train_unique_users = 0
            test_unique_users = 0
            if len(train_df) > 0 and "user_id" in train_df.columns:
                train_unique_users = int(train_df["user_id"].nunique())
            if len(test_df) > 0 and "user_id" in test_df.columns:
                test_unique_users = int(test_df["user_id"].nunique())

            # Compute manifest hash (SHA-256 of sorted record IDs)
            all_ids = sorted(train_ids + test_ids)
            id_string = ",".join(all_ids)
            manifest_hash = hashlib.sha256(id_string.encode()).hexdigest()

            manifest = {
                "strategy": s.value if hasattr(s, "value") else str(s),
                "seed": split_config.seed,
                "training_cutoff_date": cutoff.isoformat(),
                "train_record_ids": train_ids,
                "test_record_ids": test_ids,
                "fold_assignments": None,  # Will be populated in train.py if k-fold
                "train_size": len(features_train),
                "test_size": len(features_test),
                "train_fraud_rate": (
                    float(labels_train.mean()) if len(labels_train) > 0 else 0.0
                ),
                "test_fraud_rate": (
                    float(labels_test.mean()) if len(labels_test) > 0 else 0.0
                ),
                "train_time_range": train_time_range,
                "test_time_range": test_time_range,
                "train_unique_users": train_unique_users,
                "test_unique_users": test_unique_users,
                "manifest_hash": f"sha256:{manifest_hash}",
            }

        return TrainTestSplit(
            X_train=features_train,
            y_train=labels_train,
            X_test=features_test,
            y_test=labels_test,
            split_manifest=manifest,
        )

    def _load_train_set(self, session: Session, cutoff: datetime) -> pd.DataFrame:
        """Load training set with label maturity enforcement.

        Train Set Rules:
        - transaction_timestamp < cutoff
        - is_train_eligible = True
        - Label is fraud ONLY IF fraud_confirmed_at <= cutoff (knowledge horizon)
        """
        query = text("""
            SELECT
                fs.record_id,
                fs.user_id,
                fs.velocity_24h,
                fs.amount_to_avg_ratio_30d,
                fs.balance_volatility_z_score,
                fs.experimental_signals,
                em.is_train_eligible,
                em.fraud_confirmed_at,
                gr.is_fraudulent,
                gr.transaction_timestamp,
                -- Knowledge Horizon: Only label fraud if confirmed before cutoff
                CASE
                    WHEN gr.is_fraudulent = TRUE
                         AND em.fraud_confirmed_at IS NOT NULL
                         AND em.fraud_confirmed_at <= :cutoff
                    THEN 1
                    ELSE 0
                END AS label
            FROM feature_snapshots fs
            INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
            INNER JOIN generated_records gr ON fs.record_id = gr.record_id
            WHERE gr.transaction_timestamp < :cutoff
              AND em.is_train_eligible = TRUE
            ORDER BY gr.transaction_timestamp
        """)

        result = session.execute(query, {"cutoff": cutoff})
        rows = result.fetchall()
        columns = result.keys()

        return pd.DataFrame(rows, columns=list(columns))

    def _load_test_set(self, session: Session, cutoff: datetime) -> pd.DataFrame:
        """Load test set (all records after cutoff).

        Test Set Rules:
        - transaction_timestamp >= cutoff
        - Uses actual fraud label (no knowledge horizon restriction)
        """
        query = text("""
            SELECT
                fs.record_id,
                fs.user_id,
                fs.velocity_24h,
                fs.amount_to_avg_ratio_30d,
                fs.balance_volatility_z_score,
                fs.experimental_signals,
                em.is_train_eligible,
                em.fraud_confirmed_at,
                gr.is_fraudulent,
                gr.transaction_timestamp,
                -- Test set uses actual fraud label
                CASE WHEN gr.is_fraudulent = TRUE THEN 1 ELSE 0 END AS label
            FROM feature_snapshots fs
            INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
            INNER JOIN generated_records gr ON fs.record_id = gr.record_id
            WHERE gr.transaction_timestamp >= :cutoff
            ORDER BY gr.transaction_timestamp
        """)

        result = session.execute(query, {"cutoff": cutoff})
        rows = result.fetchall()
        columns = result.keys()

        return pd.DataFrame(rows, columns=list(columns))

    def get_split_summary(self, split: TrainTestSplit) -> dict:
        """Get summary statistics for the train/test split.

        Args:
            split: TrainTestSplit object.

        Returns:
            Dictionary with summary statistics.
        """
        train_fraud = int(split.y_train.sum()) if len(split.y_train) > 0 else 0
        test_fraud = int(split.y_test.sum()) if len(split.y_test) > 0 else 0

        return {
            "train_size": split.train_size,
            "test_size": split.test_size,
            "train_fraud_rate": split.train_fraud_rate,
            "test_fraud_rate": split.test_fraud_rate,
            "train_fraud_count": train_fraud,
            "test_fraud_count": test_fraud,
        }
