"""Tests for split strategies."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from api.schemas import SplitConfig, SplitStrategy
from model.loader import DataLoader
from model.split_strategies import (
    GroupTemporalSplitStrategy,
    SplitResult,
    TemporalSplitStrategy,
    TimeSeriesKFoldStrategy,
    get_strategy,
)


class TestTemporalSplit:
    """TemporalSplitStrategy respects cutoff."""

    def test_temporal_split_respects_cutoff(self):
        """Train before cutoff, test after."""
        df = pd.DataFrame(
            {
                "transaction_timestamp": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-05",
                        "2024-01-10",
                        "2024-01-15",
                        "2024-01-20",
                    ]
                ),
                "x": [1, 2, 3, 4, 5],
            }
        )
        config = SplitConfig(seed=42)
        cutoff = pd.Timestamp("2024-01-12")
        result = TemporalSplitStrategy().split(df, cutoff, config)
        assert isinstance(result, SplitResult)
        assert len(result.train_indices) == 3
        assert len(result.test_indices) == 2
        assert result.fold_assignments is None
        assert set(result.train_indices) == {0, 1, 2}
        assert set(result.test_indices) == {3, 4}


class TestTimeSeriesKFold:
    """TimeSeriesKFoldStrategy produces n folds."""

    def test_kfold_generates_n_folds(self):
        """fold_assignments has n_folds entries."""
        n = 100
        df = pd.DataFrame({"x": np.arange(n)})
        config = SplitConfig(strategy=SplitStrategy.KFOLD_TEMPORAL, n_folds=5)
        result = TimeSeriesKFoldStrategy().split(df, None, config)
        assert result.fold_assignments is not None
        assert len(result.fold_assignments) == 5
        for i in range(5):
            key = f"fold_{i}"
            assert key in result.fold_assignments
            assert "train" in result.fold_assignments[key]
            assert "val" in result.fold_assignments[key]


class TestGroupTemporalSplit:
    """GroupTemporalSplitStrategy avoids user overlap."""

    def test_group_split_no_user_overlap(self):
        """Train and test users are disjoint when possible."""
        df = pd.DataFrame(
            {
                "transaction_timestamp": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-10",
                        "2024-01-11",
                    ]
                ),
                "user_id": ["u1", "u2", "u3", "u4"],
            }
        )
        config = SplitConfig(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            group_column="user_id",
        )
        cutoff = pd.Timestamp("2024-01-05")
        result = GroupTemporalSplitStrategy().split(df, cutoff, config)
        train_users = {df.loc[i, "user_id"] for i in result.train_indices}
        test_users = {df.loc[i, "user_id"] for i in result.test_indices}
        assert train_users & test_users == set()


class TestSplitManifestStructure:
    """Split manifest from loader has expected keys."""

    def test_split_manifest_structure(self):
        """Manifest contains strategy, seed, train/test ids, sizes."""
        loader = DataLoader(database_url="postgresql://x:x@localhost/x")
        mock = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                "r1",
                "u1",
                1.0,
                1.0,
                0.0,
                None,
                True,
                None,
                False,
                pd.Timestamp("2024-01-01"),
                0,
            ),
        ]
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "transaction_timestamp",
            "label",
        ]
        mock.execute.return_value = mock_result
        config = SplitConfig(seed=42)
        split = loader.load_train_test_split(
            "2024-04-01",
            session=mock,
            feature_columns=["velocity_24h", "amount_to_avg_ratio_30d"],
            split_config=config,
        )
        assert split.split_manifest is not None
        m = split.split_manifest
        assert "strategy" in m
        assert "seed" in m
        assert "training_cutoff_date" in m
        assert "train_record_ids" in m
        assert "test_record_ids" in m
        assert "train_size" in m
        assert "test_size" in m


class TestGetStrategy:
    """Factory returns correct strategy."""

    def test_get_strategy_returns_impl(self):
        """Each strategy enum maps to an implementation."""
        for s in SplitStrategy:
            impl = get_strategy(s)
            assert impl is not None


class TestEnhancedManifest:
    """Enhanced split manifest with time ranges, user counts, and hash."""

    def test_manifest_has_time_ranges(self):
        """Manifest includes train_time_range and test_time_range."""
        from datetime import datetime
        from unittest.mock import MagicMock

        from model.loader import DataLoader

        loader = DataLoader(database_url="postgresql://test:test@localhost/test")
        session = MagicMock()

        # Create mock data with timestamps
        train_data = [
            (
                "rec1",
                "u1",
                1,
                1.0,
                0.0,
                None,
                True,
                None,
                False,
                datetime(2024, 1, 1),
                0,
            ),
            (
                "rec2",
                "u2",
                2,
                2.0,
                0.1,
                None,
                True,
                None,
                False,
                datetime(2024, 3, 31),
                0,
            ),
        ]
        test_data = [
            (
                "rec3",
                "u3",
                3,
                3.0,
                0.2,
                None,
                True,
                None,
                False,
                datetime(2024, 4, 1),
                0,
            ),
        ]

        mock_result_train = MagicMock()
        mock_result_train.fetchall.return_value = train_data
        mock_result_train.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "transaction_timestamp",
            "label",
        ]

        mock_result_test = MagicMock()
        mock_result_test.fetchall.return_value = test_data
        mock_result_test.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "transaction_timestamp",
            "label",
        ]

        def execute_side_effect(query, params):
            if "transaction_timestamp <" in str(query):
                return mock_result_train
            return mock_result_test

        session.execute.side_effect = execute_side_effect

        config = SplitConfig(seed=42)
        split = loader.load_train_test_split(
            datetime(2024, 4, 1),
            session=session,
            feature_columns=["velocity_24h", "amount_to_avg_ratio_30d"],
            split_config=config,
        )

        assert split.split_manifest is not None
        assert "train_time_range" in split.split_manifest
        assert "test_time_range" in split.split_manifest
        assert split.split_manifest["train_time_range"] is not None
        assert split.split_manifest["test_time_range"] is not None
        assert "min" in split.split_manifest["train_time_range"]
        assert "max" in split.split_manifest["train_time_range"]

    def test_manifest_has_unique_user_counts(self):
        """Manifest includes train_unique_users and test_unique_users."""
        from datetime import datetime
        from unittest.mock import MagicMock

        from model.loader import DataLoader

        loader = DataLoader(database_url="postgresql://test:test@localhost/test")
        session = MagicMock()

        # Create mock data with different users
        train_data = [
            (
                "rec1",
                "u1",
                1,
                1.0,
                0.0,
                None,
                True,
                None,
                False,
                datetime(2024, 3, 1),
                0,
            ),
            (
                "rec2",
                "u2",
                2,
                2.0,
                0.1,
                None,
                True,
                None,
                False,
                datetime(2024, 3, 2),
                0,
            ),
        ]
        test_data = [
            (
                "rec3",
                "u3",
                3,
                3.0,
                0.2,
                None,
                True,
                None,
                False,
                datetime(2024, 4, 1),
                0,
            ),
        ]

        mock_result_train = MagicMock()
        mock_result_train.fetchall.return_value = train_data
        mock_result_train.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "transaction_timestamp",
            "label",
        ]

        mock_result_test = MagicMock()
        mock_result_test.fetchall.return_value = test_data
        mock_result_test.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "transaction_timestamp",
            "label",
        ]

        def execute_side_effect(query, params):
            if "transaction_timestamp <" in str(query):
                return mock_result_train
            return mock_result_test

        session.execute.side_effect = execute_side_effect

        config = SplitConfig(seed=42)
        split = loader.load_train_test_split(
            datetime(2024, 4, 1),
            session=session,
            feature_columns=["velocity_24h", "amount_to_avg_ratio_30d"],
            split_config=config,
        )

        assert split.split_manifest is not None
        assert "train_unique_users" in split.split_manifest
        assert "test_unique_users" in split.split_manifest
        assert split.split_manifest["train_unique_users"] == 2
        assert split.split_manifest["test_unique_users"] == 1

    def test_manifest_has_hash(self):
        """Manifest includes manifest_hash."""
        from datetime import datetime
        from unittest.mock import MagicMock

        from model.loader import DataLoader

        loader = DataLoader(database_url="postgresql://test:test@localhost/test")
        session = MagicMock()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (
                "rec1",
                "u1",
                1,
                1.0,
                0.0,
                None,
                True,
                None,
                False,
                datetime(2024, 3, 1),
                0,
            ),
        ]
        mock_result.keys.return_value = [
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "is_train_eligible",
            "fraud_confirmed_at",
            "is_fraudulent",
            "transaction_timestamp",
            "label",
        ]
        session.execute.return_value = mock_result

        config = SplitConfig(seed=42)
        split = loader.load_train_test_split(
            datetime(2024, 4, 1),
            session=session,
            feature_columns=["velocity_24h", "amount_to_avg_ratio_30d"],
            split_config=config,
        )

        assert split.split_manifest is not None
        assert "manifest_hash" in split.split_manifest
        assert split.split_manifest["manifest_hash"].startswith("sha256:")
