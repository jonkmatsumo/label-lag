"""Feature set storage interface and implementation."""

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod

from features.spec import FeatureSetSpec

logger = logging.getLogger(__name__)


class FeatureSetStore(ABC):
    """Interface for storing and retrieving FeatureSetSpecs."""

    @abstractmethod
    def save(self, spec: FeatureSetSpec) -> None:
        """Save a feature set specification."""
        pass

    @abstractmethod
    def get(self, feature_set_id: str) -> FeatureSetSpec | None:
        """Retrieve a feature set specification by ID."""
        pass

    @abstractmethod
    def list(
        self, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[FeatureSetSpec], str | None]:
        """List feature sets with optional pagination."""
        pass


class MLflowFeatureSetStore(FeatureSetStore):
    """MLflow-backed storage for FeatureSetSpecs using runs and tags."""

    def __init__(
        self, tracking_uri: str | None = None, experiment_name: str | None = None
    ):
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "ach-fraud-detection"
        )
        self._mlflow.set_experiment(self.experiment_name)

    def save(self, spec: FeatureSetSpec) -> None:
        """Save spec to MLflow.

        Note: This is typically done during training within an active run.
        If an active run exists, we log to it. If not, we start a new one?
        For now, we assume this is called within a training context or we explicitly
        log it. But TrainModel already logs it.

        This method ensures it is persisted if not already.
        """
        # If we are in an active run, log it.
        if self._mlflow.active_run():
            self._log_spec_to_active_run(spec)
        else:
            # Check if it already exists to avoid duplicates?
            # Or just start a ephemeral run to register it?
            # For the purpose of "Store", we might want a dedicated run
            # or just rely on training runs.
            # Let's assume we search for it first.
            existing = self.get(spec.id)
            if existing:
                return
            with self._mlflow.start_run(run_name=f"FeatureSet_{spec.id}"):
                self._log_spec_to_active_run(spec)

    def _log_spec_to_active_run(self, spec: FeatureSetSpec) -> None:
        self._mlflow.set_tag("feature_set_id", spec.id)
        self._mlflow.set_tag("feature_set_hash", spec.hash)
        self._mlflow.log_dict(spec.model_dump(), "feature_set.json")

    def get(self, feature_set_id: str) -> FeatureSetSpec | None:
        """Retrieve spec by searching runs for feature_set_id tag."""
        try:
            client = self._mlflow.MlflowClient()
            # Search for runs with the tag
            experiment = client.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.warning(f"Experiment {self.experiment_name} not found")
                return None

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.feature_set_id = '{feature_set_id}'",
                max_results=1,
            )

            if not runs:
                logger.warning(f"No feature set found with id {feature_set_id}")
                return None

            run_id = runs[0].info.run_id
            # Download artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                path = client.download_artifacts(run_id, "feature_set.json", tmpdir)
                with open(path) as f:
                    data = json.load(f)
                return FeatureSetSpec(**data)

        except Exception as e:
            logger.error(f"Error retrieving feature set {feature_set_id}: {e}")
            return None

    def list(
        self, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[FeatureSetSpec], str | None]:
        """List feature sets by searching for runs with feature_set_id tag."""
        try:
            client = self._mlflow.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if not experiment:
                return [], None

            # MLflow search_runs uses page_token for pagination
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.feature_set_id != ''",
                max_results=limit,
                page_token=cursor,
            )

            specs = []
            for run in runs:
                # We can optimize by NOT downloading every artifact if we
                # only need metadata, but the spec has the full feature list.
                # For a "list" view, maybe we should return a summary?
                # The requirements say "ID + hash + feature count".
                # Full spec has these.
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        path = client.download_artifacts(
                            run.info.run_id, "feature_set.json", tmpdir
                        )
                        with open(path) as f:
                            data = json.load(f)
                        specs.append(FeatureSetSpec(**data))
                except Exception as e:
                    logger.debug(f"Failed to load spec for run {run.info.run_id}: {e}")

            return specs, runs.token

        except Exception as e:
            logger.error(f"Error listing feature sets: {e}")
            return [], None


def get_feature_store() -> FeatureSetStore:
    """Factory for default store."""
    return MLflowFeatureSetStore()
