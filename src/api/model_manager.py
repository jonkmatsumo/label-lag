"""Model manager for dynamic model loading from MLflow/MinIO.

Provides a singleton ModelManager that handles:
- Loading production models from MLflow model registry
- Fallback to local model if MLflow/MinIO is unavailable
- Thread-safe model access
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from threading import Lock
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from api.rules import RuleSet

# Configure logging
logger = logging.getLogger(__name__)

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Model registry name
MODEL_NAME = "ach-fraud-detection"

# Fallback model path
FALLBACK_MODEL_PATH = Path(__file__).parent.parent / "model" / "fallback_model.pkl"


class ModelManager:
    """Singleton manager for ML model loading and inference.

    Handles loading models from MLflow registry with fallback to local model
    if the registry is unavailable.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls) -> "ModelManager":
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the model manager."""
        if self._initialized:
            return

        self._model = None
        self._model_version: str | None = None
        self._model_source: str = "none"
        self._required_features: list[str] | None = None
        self._ruleset: RuleSet | None = None
        self._initialized = True

    @property
    def model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None

    @property
    def model_version(self) -> str:
        """Get the current model version."""
        return self._model_version or "unknown"

    @property
    def model_source(self) -> str:
        """Get the model source (mlflow, fallback, or none)."""
        return self._model_source

    @property
    def required_features(self) -> list[str]:
        """Get the list of required feature columns for this model.

        Returns:
            List of required feature column names. Falls back to default
            FEATURE_COLUMNS if not available.
        """
        if self._required_features is not None:
            return self._required_features

        # Fallback to default feature columns
        from model.loader import DataLoader

        return DataLoader.FEATURE_COLUMNS

    @property
    def ruleset(self) -> RuleSet | None:
        """Get the loaded ruleset.

        Returns:
            RuleSet instance if loaded, None otherwise.
        """
        return self._ruleset

    @property
    def rules_version(self) -> str | None:
        """Get the version of loaded rules.

        Returns:
            Rules version string, or None if no rules loaded.
        """
        if self._ruleset is None:
            return None
        return self._ruleset.version

    def load_production_model(self) -> bool:
        """Load the production model from MLflow registry.

        Attempts to load from MLflow first. If that fails, falls back to
        a local pickle file if available.

        Returns:
            True if a model was loaded successfully, False otherwise.
        """
        # Try loading from MLflow first
        if self._load_from_mlflow():
            return True

        # Fall back to local model
        if self._load_fallback_model():
            return True

        logger.error("No model available - both MLflow and fallback failed")
        return False

    def _load_from_mlflow(self) -> bool:
        """Attempt to load model from MLflow registry.

        Returns:
            True if successful, False otherwise.
        """
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_uri = f"models:/{MODEL_NAME}/Production"
            logger.info(f"Loading model from MLflow: {model_uri}")

            self._model = mlflow.pyfunc.load_model(model_uri)
            self._model_version = self._get_production_version()
            self._model_source = "mlflow"

            # Try to load feature_columns.json artifact
            self._load_feature_columns_artifact()

            # Try to load rules.json artifact
            self._load_rules_artifact()

            logger.info(
                f"Successfully loaded model version {self._model_version} from MLflow"
            )

            # Benchmark inference latency after load
            self._benchmark_inference()

            return True

        except Exception as e:
            logger.critical(
                f"Failed to load model from MLflow/MinIO: {e}. "
                "Attempting fallback to local model."
            )
            return False

    def _load_feature_columns_artifact(self) -> None:
        """Load feature_columns.json artifact from the model run.

        Sets self._required_features if artifact is found. If not found,
        falls back to default FEATURE_COLUMNS.
        """
        try:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if v.current_stage == "Production":
                    run_id = v.run_id
                    # Try to download the artifact
                    try:
                        artifact_path = client.download_artifacts(
                            run_id, "feature_columns.json"
                        )
                        with open(artifact_path) as f:
                            self._required_features = json.load(f)
                        logger.info(
                            f"Loaded feature columns from artifact: "
                            f"{self._required_features}"
                        )
                        return
                    except Exception:
                        # Artifact not found - this is OK for older models
                        logger.debug(
                            "feature_columns.json artifact not found, "
                            "using default columns"
                        )
                        break
        except Exception as e:
            logger.debug(f"Could not load feature_columns artifact: {e}")

    def _load_rules_artifact(self) -> None:
        """Load rules.json artifact from the model run.

        Tries to load from MLflow artifacts first. If not found, falls back
        to config/default_rules.json. If that also fails, uses empty RuleSet.
        """
        # Try loading from MLflow artifacts
        try:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if v.current_stage == "Production":
                    run_id = v.run_id
                    try:
                        artifact_path = client.download_artifacts(run_id, "rules.json")
                        with open(artifact_path) as f:
                            data = json.load(f)
                        self._ruleset = RuleSet.from_dict(data)
                        logger.info(
                            f"Loaded rules version {self._ruleset.version} "
                            f"from MLflow artifact"
                        )
                        return
                    except Exception:
                        # Artifact not found - try fallback
                        logger.debug(
                            "rules.json artifact not found in MLflow, trying fallback"
                        )
                        break
        except Exception as e:
            logger.debug(f"Could not load rules from MLflow: {e}")

        # Fallback to default rules file
        try:
            default_rules_path = (
                Path(__file__).parent.parent / "config" / "default_rules.json"
            )
            self._ruleset = RuleSet.load_from_file(default_rules_path)
            logger.info(
                f"Loaded rules version {self._ruleset.version} from default file"
            )
        except (FileNotFoundError, ValueError) as e:
            # Use empty ruleset if default file also fails
            logger.debug(f"Could not load default rules file: {e}")
            self._ruleset = RuleSet.empty()
            logger.debug("Using empty ruleset")

    def _benchmark_inference(self, n_samples: int = 100) -> None:
        """Benchmark inference latency and log to MLflow.

        Args:
            n_samples: Number of samples to use for benchmarking.
        """
        if self._model is None:
            return

        try:
            # Create sample data matching required features
            required = self.required_features
            sample_data = pd.DataFrame(
                {feat: np.random.rand(n_samples) for feat in required}
            )

            # Measure latencies
            latencies_ms = []
            for _ in range(n_samples):
                start = time.perf_counter()
                self._model.predict(sample_data.iloc[[0]])
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies_ms.append(elapsed_ms)

            # Calculate percentiles
            latencies_sorted = sorted(latencies_ms)
            p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
            p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

            # Log to MLflow (find the run_id from the model version)
            try:
                client = mlflow.MlflowClient()
                versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                for v in versions:
                    if (
                        v.current_stage == "Production"
                        and v.version == self._model_version
                    ):
                        run_id = v.run_id
                        client.log_metric(run_id, "inference_latency_p50_ms", p50)
                        client.log_metric(run_id, "inference_latency_p95_ms", p95)
                        client.log_metric(run_id, "inference_latency_p99_ms", p99)
                        logger.info(
                            f"Logged inference latency: p50={p50:.2f}ms, "
                            f"p95={p95:.2f}ms, p99={p99:.2f}ms"
                        )
                        break
            except Exception as e:
                logger.debug(f"Could not log inference latency to MLflow: {e}")

        except Exception as e:
            logger.debug(f"Inference benchmarking failed: {e}")

    def _load_fallback_model(self) -> bool:
        """Attempt to load fallback model from local file.

        Returns:
            True if successful, False otherwise.
        """
        if not FALLBACK_MODEL_PATH.exists():
            logger.warning(f"Fallback model not found at {FALLBACK_MODEL_PATH}")
            return False

        try:
            logger.info(f"Loading fallback model from {FALLBACK_MODEL_PATH}")

            with open(FALLBACK_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)

            self._model_version = "fallback"
            self._model_source = "fallback"

            logger.warning(
                "Using fallback model - MLflow registry was unavailable. "
                "Model predictions may be outdated."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return False

    def _get_production_version(self) -> str:
        """Get the version number of the production model.

        Returns:
            Version string or 'unknown'.
        """
        try:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if v.current_stage == "Production":
                    return f"v{v.version}"
        except Exception:
            pass
        return "unknown"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the loaded model.

        Args:
            features: DataFrame with feature columns matching model input.

        Returns:
            Array of prediction probabilities.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_production_model() first.")

        try:
            # MLflow pyfunc models return predictions directly
            if self._model_source == "mlflow":
                predictions = self._model.predict(features)
            else:
                # Fallback sklearn model - get probability of positive class
                predictions = self._model.predict_proba(features)[:, 1]

            return np.asarray(predictions)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_single(self, features: dict[str, Any]) -> float | None:
        """Generate prediction for a single observation.

        Args:
            features: Dictionary of feature name -> value.

        Returns:
            Prediction probability for positive class, or None if required
            features are missing.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_production_model() first.")

        # Validate that all required features are present
        required = self.required_features
        missing_features = [f for f in required if f not in features]
        if missing_features:
            logger.warning(
                f"Missing required features for prediction: {missing_features}. "
                f"Required: {required}, Provided: {list(features.keys())}"
            )
            return None

        # Build DataFrame with only required features in correct order
        ordered_features = {f: features[f] for f in required}
        df = pd.DataFrame([ordered_features])
        predictions = self.predict(df)
        return float(predictions[0])

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from the loaded model.

        Returns:
            Dictionary mapping feature names to importance scores, or None if
            feature importance cannot be extracted.
        """
        if self._model is None:
            return None

        try:
            # Try to access underlying XGBoost model
            # MLflow pyfunc models wrap the original model
            underlying_model = (
                self._model._model_impl
                if hasattr(self._model, "_model_impl")
                else self._model
            )

            # Check if it's an XGBoost model
            if hasattr(underlying_model, "get_booster"):
                # XGBoost model - get feature importance
                booster = underlying_model.get_booster()
                importance_dict = booster.get_score(importance_type="gain")

                # Map feature indices to feature names
                feature_names = self.required_features
                if len(importance_dict) == len(feature_names):
                    # Importance dict uses f0, f1, etc. as keys
                    importance_map = {}
                    for i, feature_name in enumerate(feature_names):
                        key = f"f{i}"
                        if key in importance_dict:
                            importance_map[feature_name] = float(importance_dict[key])
                    return importance_map
                else:
                    # Try direct mapping if keys are feature names
                    return {
                        k: float(v)
                        for k, v in importance_dict.items()
                        if k in feature_names
                    }
            elif hasattr(underlying_model, "feature_importances_"):
                # Scikit-learn style model
                importances = underlying_model.feature_importances_
                feature_names = self.required_features
                if len(importances) == len(feature_names):
                    return {
                        name: float(imp)
                        for name, imp in zip(feature_names, importances)
                    }
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        return None


# Module-level singleton instance
_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance.

    Returns:
        The ModelManager singleton.
    """
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
