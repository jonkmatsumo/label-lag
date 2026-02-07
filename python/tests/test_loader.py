"""Tests for DataLoader with analytics-backed data loading.

Tests focus on feature column selection, split logic, and manifest generation.
The actual temporal splitting and label maturity logic is delegated to the
Analytics service.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from model.loader import DataLoader, TrainTestSplit
from training_server.schemas import SplitConfig


class TestTrainTestSplit:
    """Tests for TrainTestSplit dataclass."""

    def test_train_size(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2, 3]}),
            y_train=pd.Series([0, 1, 0]),
            X_test=pd.DataFrame({"a": [4, 5]}),
            y_test=pd.Series([1, 1]),
        )
        assert split.train_size == 3

    def test_test_size(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2, 3]}),
            y_train=pd.Series([0, 1, 0]),
            X_test=pd.DataFrame({"a": [4, 5]}),
            y_test=pd.Series([1, 1]),
        )
        assert split.test_size == 2

    def test_train_fraud_rate(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2, 3, 4]}),
            y_train=pd.Series([0, 1, 0, 1]),
            X_test=pd.DataFrame(),
            y_test=pd.Series(dtype=int),
        )
        assert split.train_fraud_rate == 0.5

    def test_test_fraud_rate(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame(),
            y_train=pd.Series(dtype=int),
            X_test=pd.DataFrame({"a": [1, 2, 3, 4, 5]}),
            y_test=pd.Series([1, 1, 0, 0, 0]),
        )
        assert split.test_fraud_rate == 0.4

    def test_empty_train_fraud_rate(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame(),
            y_train=pd.Series(dtype=int),
            X_test=pd.DataFrame({"a": [1]}),
            y_test=pd.Series([1]),
        )
        assert split.train_fraud_rate == 0.0

    def test_empty_test_fraud_rate(self):
        split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1]}),
            y_train=pd.Series([0]),
            X_test=pd.DataFrame(),
            y_test=pd.Series(dtype=int),
        )
        assert split.test_fraud_rate == 0.0


class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_init_accepts_database_url(self):
        """Test DataLoader accepts database_url (ignored in compute-only mode)."""
        loader = DataLoader(database_url="postgresql://test:test@localhost/test")
        assert loader is not None

    def test_feature_columns_defined(self):
        """Test that FEATURE_COLUMNS constant is defined."""
        assert "velocity_24h" in DataLoader.FEATURE_COLUMNS
        assert "amount_to_avg_ratio_30d" in DataLoader.FEATURE_COLUMNS
        assert "balance_volatility_z_score" in DataLoader.FEATURE_COLUMNS


def _make_mock_record(
    record_id: str,
    user_id: str,
    velocity_24h: int,
    amount_to_avg_ratio_30d: float,
    balance_volatility_z_score: float,
    is_fraudulent: bool,
    created_at: datetime,
) -> MagicMock:
    """Create a mock proto record."""
    record = MagicMock()
    record.record_id = record_id
    record.user_id = user_id
    record.velocity_24h = velocity_24h
    record.amount_to_avg_ratio_30d = amount_to_avg_ratio_30d
    record.balance_volatility_z_score = balance_volatility_z_score
    record.is_fraudulent = is_fraudulent
    record.created_at = MagicMock()
    record.created_at.ToDatetime.return_value = created_at
    return record


class TestFeatureColumnOverride:
    """Tests for feature column override behavior."""

    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        return DataLoader()

    def test_load_uses_default_columns_when_none_provided(self, loader, monkeypatch):
        """Verify default behavior when feature_columns is None."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record("rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1))
        ]
        mock_response.test_records = [
            _make_mock_record("rec2", "user2", 10, 2.5, 0.5, True, datetime(2024, 5, 1))
        ]
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        split = loader.load_train_test_split(datetime(2024, 4, 1))

        # Should use default FEATURE_COLUMNS
        assert list(split.X_train.columns) == DataLoader.FEATURE_COLUMNS
        assert list(split.X_test.columns) == DataLoader.FEATURE_COLUMNS

    def test_load_uses_override_list_when_provided(self, loader, monkeypatch):
        """Verify custom feature list is used when provided."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record("rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1))
        ]
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        custom_columns = ["velocity_24h", "amount_to_avg_ratio_30d"]
        split = loader.load_train_test_split(
            datetime(2024, 4, 1), feature_columns=custom_columns
        )

        # Should use custom columns
        assert list(split.X_train.columns) == custom_columns

    def test_load_raises_on_missing_columns(self, loader, monkeypatch):
        """Verify ValueError is raised when requested columns are missing."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record("rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1))
        ]
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        missing_columns = ["velocity_24h", "nonexistent_column"]

        with pytest.raises(ValueError) as exc_info:
            loader.load_train_test_split(
                datetime(2024, 4, 1), feature_columns=missing_columns
            )

        assert "nonexistent_column" in str(exc_info.value)
        assert "Requested feature columns not found" in str(exc_info.value)

    def test_load_handles_empty_override_list(self, loader, monkeypatch):
        """Verify empty feature list creates empty DataFrame columns."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = []
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        empty_columns = []
        split = loader.load_train_test_split(
            datetime(2024, 4, 1), feature_columns=empty_columns
        )

        assert list(split.X_train.columns) == empty_columns
        assert list(split.X_test.columns) == empty_columns


class TestDataLoaderTemporalSplit:
    """Tests for temporal splitting behavior."""

    @pytest.fixture
    def loader(self):
        return DataLoader()

    def test_cutoff_date_string_parsing(self, loader, monkeypatch):
        """Test that string dates are parsed correctly."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = []
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        split = loader.load_train_test_split("2024-04-01")
        assert isinstance(split, TrainTestSplit)

        # Verify cutoff was passed to client
        mock_client.get_training_data.assert_called_once()

    def test_cutoff_date_datetime_accepted(self, loader, monkeypatch):
        """Test that datetime objects are accepted."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = []
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        split = loader.load_train_test_split(datetime(2024, 4, 1))
        assert isinstance(split, TrainTestSplit)

    def test_train_records_are_from_train_set(self, loader, monkeypatch):
        """Test that train records are correctly processed."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record(
                "rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1)
            ),
            _make_mock_record(
                "rec2", "user2", 10, 2.5, -0.5, True, datetime(2024, 3, 15)
            ),
        ]
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        split = loader.load_train_test_split(datetime(2024, 4, 1))

        assert split.train_size == 2
        assert split.test_size == 0

    def test_test_records_are_from_test_set(self, loader, monkeypatch):
        """Test that test records are correctly processed."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = []
        mock_response.test_records = [
            _make_mock_record(
                "rec3", "user3", 15, 3.5, 1.0, True, datetime(2024, 5, 1)
            ),
        ]
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        split = loader.load_train_test_split(datetime(2024, 4, 1))

        assert split.train_size == 0
        assert split.test_size == 1


class TestDataLoaderLabels:
    """Tests for label handling."""

    @pytest.fixture
    def loader(self):
        return DataLoader()

    def test_fraudulent_records_labeled_as_1(self, loader, monkeypatch):
        """Test that fraudulent records have label=1."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record("rec1", "user1", 5, 1.5, 0.2, True, datetime(2024, 3, 1)),
        ]
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        split = loader.load_train_test_split(datetime(2024, 4, 1))

        assert split.y_train.iloc[0] == 1

    def test_non_fraudulent_records_labeled_as_0(self, loader, monkeypatch):
        """Test that non-fraudulent records have label=0."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record(
                "rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1)
            ),
        ]
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        split = loader.load_train_test_split(datetime(2024, 4, 1))

        assert split.y_train.iloc[0] == 0


class TestSplitConfig:
    """Tests for split_config and manifest."""

    def test_loader_accepts_split_config(self, monkeypatch):
        """With split_config, split_manifest is populated."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record(
                "rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1)
            ),
        ]
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        loader = DataLoader()
        config = SplitConfig(seed=42)
        split = loader.load_train_test_split(
            datetime(2024, 4, 1),
            feature_columns=["velocity_24h", "amount_to_avg_ratio_30d"],
            split_config=config,
        )

        assert split.split_manifest is not None
        assert split.split_manifest["seed"] == 42
        assert "train_record_ids" in split.split_manifest

    def test_loader_default_config_unchanged(self, monkeypatch):
        """Without split_config, split_manifest is None."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = []
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        loader = DataLoader()
        split = loader.load_train_test_split(datetime(2024, 4, 1))

        assert split.split_manifest is None

    def test_manifest_has_record_ids(self, monkeypatch):
        """Test that manifest contains train and test record IDs."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record(
                "train_rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1)
            ),
        ]
        mock_response.test_records = [
            _make_mock_record(
                "test_rec1", "user2", 10, 2.5, 0.5, True, datetime(2024, 5, 1)
            ),
        ]
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        loader = DataLoader()
        config = SplitConfig(seed=42)
        split = loader.load_train_test_split(datetime(2024, 4, 1), split_config=config)

        assert "train_rec1" in split.split_manifest["train_record_ids"]
        assert "test_rec1" in split.split_manifest["test_record_ids"]

    def test_manifest_has_unique_user_counts(self, monkeypatch):
        """Test that manifest contains unique user counts."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record(
                "rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1)
            ),
            _make_mock_record(
                "rec2", "user1", 6, 1.6, 0.3, False, datetime(2024, 3, 2)
            ),
            _make_mock_record(
                "rec3", "user2", 7, 1.7, 0.4, False, datetime(2024, 3, 3)
            ),
        ]
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        loader = DataLoader()
        config = SplitConfig(seed=42)
        split = loader.load_train_test_split(datetime(2024, 4, 1), split_config=config)

        assert split.split_manifest["train_unique_users"] == 2

    def test_manifest_has_hash(self, monkeypatch):
        """Test that manifest contains a hash."""
        from training_server import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.train_records = [
            _make_mock_record(
                "rec1", "user1", 5, 1.5, 0.2, False, datetime(2024, 3, 1)
            ),
        ]
        mock_response.test_records = []
        mock_client.get_training_data.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        loader = DataLoader()
        config = SplitConfig(seed=42)
        split = loader.load_train_test_split(datetime(2024, 4, 1), split_config=config)

        assert split.split_manifest["manifest_hash"].startswith("sha256:")
