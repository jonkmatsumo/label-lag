"""Data loader for XGBoost training with temporal splitting via Analytics service."""

import hashlib
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from api.crud_client import get_crud_client
from api.schemas import SplitConfig


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

    In compute-only mode, the logic for temporal splitting and label maturity
    is moved to the Analytics gRPC service.
    """

    # Feature columns from feature_snapshots table
    FEATURE_COLUMNS = [
        "velocity_24h",
        "amount_to_avg_ratio_30d",
        "balance_volatility_z_score",
    ]

    def __init__(self, database_url: str | None = None):
        """Initialize DataLoader.

        Args:
            database_url: Ignored in compute-only mode.
        """
        pass

    def load_train_test_split(
        self,
        training_cutoff_date: str | datetime,
        session=None,
        feature_columns: list[str] | None = None,
        split_config: SplitConfig | None = None,
    ) -> TrainTestSplit:
        """Load train/test split via Analytics service.

        Args:
            training_cutoff_date: Cutoff date for train/test split.
            session: Ignored.
            feature_columns: Optional list of feature columns to use.
            split_config: Optional split/CV config.

        Returns:
            TrainTestSplit containing X_train, y_train, X_test, y_test.
        """
        from google.protobuf.timestamp_pb2 import Timestamp

        if isinstance(training_cutoff_date, str):
            cutoff = datetime.fromisoformat(training_cutoff_date)
        else:
            cutoff = training_cutoff_date

        if feature_columns is None:
            feature_columns = self.FEATURE_COLUMNS

        # Convert datetime to proto Timestamp
        ts = Timestamp()
        ts.FromDatetime(cutoff)

        client = get_crud_client()
        resp = client.get_training_data(cutoff_date=ts)

        train_df = self._records_to_df(resp.train_records)
        test_df = self._records_to_df(resp.test_records)

        return self._prepare_split(
            train_df, test_df, cutoff, feature_columns, split_config
        )

    def _records_to_df(self, records) -> pd.DataFrame:
        """Convert proto records to DataFrame."""
        data = []
        for r in records:
            data.append(
                {
                    "record_id": r.record_id,
                    "user_id": r.user_id,
                    "velocity_24h": r.velocity_24h,
                    "amount_to_avg_ratio_30d": r.amount_to_avg_ratio_30d,
                    "balance_volatility_z_score": r.balance_volatility_z_score,
                    "label": 1 if r.is_fraudulent else 0,
                    "transaction_timestamp": r.created_at.ToDatetime(),
                }
            )
        return pd.DataFrame(data)

    def _prepare_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cutoff: datetime,
        feature_columns: list[str],
        split_config: SplitConfig | None = None,
    ) -> TrainTestSplit:
        """Prepare TrainTestSplit from DataFrames."""
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

        manifest = None
        if split_config is not None and "record_id" in all_columns:
            train_ids = (
                train_df["record_id"].astype(str).tolist() if len(train_df) > 0 else []
            )
            test_ids = (
                test_df["record_id"].astype(str).tolist() if len(test_df) > 0 else []
            )

            # Compute unique user counts
            train_unique_users = (
                int(train_df["user_id"].nunique()) if len(train_df) > 0 else 0
            )
            test_unique_users = (
                int(test_df["user_id"].nunique()) if len(test_df) > 0 else 0
            )

            # Compute manifest hash
            all_ids = sorted(train_ids + test_ids)
            id_string = ",".join(all_ids)
            manifest_hash = hashlib.sha256(id_string.encode()).hexdigest()

            manifest = {
                "strategy": str(split_config.strategy),
                "seed": split_config.seed,
                "training_cutoff_date": cutoff.isoformat(),
                "train_record_ids": train_ids,
                "test_record_ids": test_ids,
                "train_size": len(features_train),
                "test_size": len(features_test),
                "train_fraud_rate": (
                    float(labels_train.mean()) if len(labels_train) > 0 else 0.0
                ),
                "test_fraud_rate": (
                    float(labels_test.mean()) if len(labels_test) > 0 else 0.0
                ),
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

    def get_split_summary(self, split: TrainTestSplit) -> dict:
        """Get summary statistics for the train/test split."""
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