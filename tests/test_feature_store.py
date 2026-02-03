"""Tests for the feature store and materialization pipeline."""

from synthetic_pipeline.db.models import FeatureSnapshotDB


class TestFeatureSnapshotModel:
    """Tests for FeatureSnapshotDB model."""

    def test_model_tablename(self):
        """Test that model has correct table name."""
        assert FeatureSnapshotDB.__tablename__ == "feature_snapshots"

    def test_model_has_required_columns(self):
        """Test that model has all required columns."""
        columns = FeatureSnapshotDB.__table__.columns
        column_names = [c.name for c in columns]

        required_columns = [
            "snapshot_id",
            "record_id",
            "user_id",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
            "experimental_signals",
            "computed_at",
        ]

        for col in required_columns:
            assert col in column_names, f"Missing column: {col}"

    def test_record_id_is_unique(self):
        """Test that record_id has unique constraint."""
        record_id_col = FeatureSnapshotDB.__table__.columns["record_id"]
        assert record_id_col.unique is True

    def test_record_id_foreign_key(self):
        """Test that record_id has foreign key to generated_records."""
        record_id_col = FeatureSnapshotDB.__table__.columns["record_id"]
        fk_targets = [fk.target_fullname for fk in record_id_col.foreign_keys]
        assert "generated_records.record_id" in fk_targets

    def test_experimental_signals_is_jsonb(self):
        """Test that experimental_signals is JSONB type."""
        exp_col = FeatureSnapshotDB.__table__.columns["experimental_signals"]
        # Check it's a JSONB type (PostgreSQL-specific)
        assert "JSONB" in str(exp_col.type)


class TestFeatureMaterializer:
    """Tests for feature materialization via Analytics service."""

    def test_materialize_function_imports(self):
        """Test that materialize_features function can be imported."""
        from pipeline.materialize_features import materialize_features

        assert materialize_features is not None

    def test_materialization_mode_imports(self):
        """Test that materialization mode function can be imported."""
        from pipeline.materialize_features import get_materialization_mode

        assert get_materialization_mode is not None

    def test_materialization_mode_default_is_legacy(self, monkeypatch):
        """Test that default materialization mode is legacy."""
        monkeypatch.delenv("FEATURE_MATERIALIZATION_MODE", raising=False)
        from pipeline.materialize_features import get_materialization_mode

        assert get_materialization_mode() == "legacy"

    def test_materialize_features_via_analytics(self, monkeypatch):
        """Test that materialize_features calls analytics service."""
        from unittest.mock import MagicMock
        from api import crud_client
        from pipeline.materialize_features import materialize_features

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.total_processed = 100
        mock_client.materialize_features.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        result = materialize_features(batch_size=500)

        assert result["success"] is True
        assert result["total_processed"] == 100
        mock_client.materialize_features.assert_called_once_with(batch_size=500)

    def test_materialize_features_handles_errors(self, monkeypatch):
        """Test that materialize_features handles analytics errors gracefully."""
        from unittest.mock import MagicMock
        from api import crud_client
        from pipeline.materialize_features import materialize_features

        mock_client = MagicMock()
        mock_client.materialize_features.side_effect = Exception("Analytics error")
        monkeypatch.setattr(crud_client, "_client", mock_client)

        result = materialize_features()

        assert result["success"] is False
        assert "error" in result
        assert "Analytics error" in result["error"]


class TestMigrationSQL:
    """Tests for SQL migration script."""

    def test_migration_file_exists(self):
        """Test that migration file exists."""
        import os

        migration_path = (
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        )
        assert os.path.exists(migration_path)

    def test_migration_creates_table(self):
        """Test that migration creates the table."""
        with open(
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        ) as f:
            sql = f.read()

        assert "CREATE TABLE IF NOT EXISTS feature_snapshots" in sql

    def test_migration_has_foreign_key(self):
        """Test that migration includes foreign key constraint."""
        with open(
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        ) as f:
            sql = f.read()

        assert "FOREIGN KEY (record_id)" in sql
        assert "REFERENCES generated_records(record_id)" in sql

    def test_migration_has_indexes(self):
        """Test that migration creates required indexes."""
        with open(
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        ) as f:
            sql = f.read()

        assert "CREATE INDEX" in sql
        assert "ix_feature_snapshots_record_id" in sql
        assert "ix_feature_snapshots_user_id" in sql

    def test_migration_has_gin_index_for_jsonb(self):
        """Test that migration creates GIN index for JSONB column."""
        with open(
            "src/synthetic_pipeline/db/migrations/001_create_feature_snapshots.sql"
        ) as f:
            sql = f.read()

        assert "USING GIN (experimental_signals)" in sql
