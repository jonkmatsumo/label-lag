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

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5005")

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
        self._calibrator = None
        self._calibrator_loaded = False
        self._baseline_distribution = None
        self._cached_feature_importance = None
        self._initialized = True

    @property
    def model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None

    @property
    def calibrator_loaded(self) -> bool:
        """Check if a calibration artifact is loaded."""
        return self._calibrator_loaded

    @property
    def calibrator(self):
        """Get the loaded calibrator or a default one."""
        if self._calibrator is not None:
            return self._calibrator

        # Fallback to default ScoreCalibrator
        from model.evaluate import ScoreCalibrator

        return ScoreCalibrator()

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
        from forecast.metrics import model_fallback_total

        if self._load_fallback_model():
            model_fallback_total.labels(reason="mlflow_unavailable").inc()
            return True

        logger.error("No model available - both MLflow and fallback failed")
        return False

    def _load_from_mlflow(self) -> bool:
        """Attempt to load model from MLflow registry.

        Returns:
            True if successful, False otherwise.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_uri = f"models:/{MODEL_NAME}/Production"
            logger.info(f"Loading model from MLflow: {model_uri}")

            self._model = mlflow.pyfunc.load_model(model_uri)
            self._model_version = self._get_production_version()
            self._model_source = "mlflow"

            # Try to load required_features.json artifact (FF5)
            # Falls back to feature_columns.json if missing (legacy)
            self._load_required_features_artifact()

            # Validate loaded features against registry (Commit 7)
            from features.registry import FeatureRegistry

            if self._required_features:
                registered_features = set(FeatureRegistry.list_features())
                unknown_features = [
                    f for f in self._required_features if f not in registered_features
                ]

                enforce_strict = (
                    os.getenv("ENFORCE_MODEL_FEATURES", "false").lower() == "true"
                )

                if unknown_features:
                    msg = (
                        f"Model requires features not in registry: {unknown_features}. "
                        "This may indicate a registry/model sync issue."
                    )
                    if enforce_strict:
                        logger.critical(f"STRICT ENFORCEMENT FAILURE: {msg}")
                        raise ValueError(msg)
                    logger.warning(msg)
                    # Check for missing required features in registry?
                # (Registry is source of truth for what exists,
                # model is source of truth for what it NEEDS).
                # Mismatch means model needs more than what registry defines.

            # Try to load calibrator.pkl artifact (C2)
            self._load_calibrator_artifact()

            # Try to load score_distribution.json artifact (C3)
            self._load_baseline_distribution_artifact()

            # Cache feature importance (C4)
            self._cached_feature_importance = self.get_feature_importance()

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

    def _load_required_features_artifact(self) -> None:
        """Load required_features.json artifact from the model run.

        Attempts to load required_features.json (new format).
        Falls back to feature_columns.json (legacy format) if missing.
        """
        try:
            import mlflow

            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if (
                    v.current_stage == "Production"
                    and v.version == self._model_version.lstrip("v")
                ):
                    run_id = v.run_id
                    # 1. Try new format: required_features.json
                    try:
                        path = client.download_artifacts(
                            run_id, "required_features.json"
                        )
                        with open(path) as f:
                            data = json.load(f)
                        self._required_features = data["features"]
                        logger.info(
                            f"Loaded required features from metadata: "
                            f"{self._required_features}"
                        )
                        return
                    except Exception:
                        logger.debug("required_features.json not found, trying legacy")

                    # 2. Try legacy format: feature_columns.json
                    try:
                        path = client.download_artifacts(run_id, "feature_columns.json")
                        with open(path) as f:
                            self._required_features = json.load(f)
                        logger.warning(
                            f"Loaded features from legacy artifact: "
                            f"{self._required_features}. "
                            "Please retrain model to generate full metadata."
                        )
                        return
                    except Exception:
                        logger.warning(
                            "No feature artifacts found. "
                            "Falling back to default columns."
                        )
                        break
        except Exception as e:
            logger.debug(f"Could not load required features metadata: {e}")

    def _load_calibrator_artifact(self) -> None:
        """Load calibrator.pkl artifact from the model run.

        Sets self._calibrator and self._calibrator_loaded if artifact is found.
        """
        try:
            import joblib
            import mlflow

            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if v.current_stage == "Production":
                    run_id = v.run_id
                    try:
                        artifact_path = client.download_artifacts(
                            run_id, "calibrator.pkl"
                        )
                        self._calibrator = joblib.load(artifact_path)
                        self._calibrator_loaded = True
                        logger.info(
                            "Successfully loaded calibrator artifact from MLflow"
                        )
                        return
                    except Exception as e:
                        logger.debug(
                            f"calibrator.pkl artifact not found or failed to load: {e}"
                        )
                        break
        except Exception as e:
            logger.debug(f"Could not load calibrator artifact: {e}")

    def _load_baseline_distribution_artifact(self) -> None:
        """Load score_distribution.json artifact from the model run.

        Sets self._baseline_distribution if artifact is found.
        """
        try:
            import mlflow

            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if v.current_stage == "Production":
                    run_id = v.run_id
                    try:
                        artifact_path = client.download_artifacts(
                            run_id, "score_distribution.json"
                        )
                        with open(artifact_path) as f:
                            self._baseline_distribution = json.load(f)
                        logger.info(
                            "Successfully loaded baseline score distribution "
                            "from MLflow"
                        )
                        return
                    except Exception as e:
                        logger.debug(f"score_distribution.json artifact not found: {e}")
                        break
        except Exception as e:
            logger.debug(f"Could not load baseline distribution artifact: {e}")

    @property
    def baseline_distribution(self) -> dict | None:
        """Get the baseline score distribution."""
        return self._baseline_distribution

    @property
    def cached_feature_importance(self) -> dict[str, float] | None:
        """Get the cached feature importance."""
        return self._cached_feature_importance

    def _benchmark_inference(self, n_samples: int = 100) -> None:
        """Benchmark inference latency and log to MLflow.

        Args:
            n_samples: Number of samples to use for benchmarking.
        """
        if self._model is None:
            return

        try:
            import mlflow

            # Create sample data matching required features
            required = self.required_features
            rng = np.random.default_rng(0)
            sample_data = pd.DataFrame(
                {feat: rng.random(n_samples) for feat in required}
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
                normalized_model_version = (self._model_version or "").lstrip("v")
                matched_version = False
                for v in versions:
                    if (
                        v.current_stage == "Production"
                        and v.version == normalized_model_version
                    ):
                        matched_version = True
                        run_id = v.run_id
                        client.log_metric(run_id, "inference_latency_p50_ms", p50)
                        client.log_metric(run_id, "inference_latency_p95_ms", p95)
                        client.log_metric(run_id, "inference_latency_p99_ms", p99)
                        logger.info(
                            f"Logged inference latency: p50={p50:.2f}ms, "
                            f"p95={p95:.2f}ms, p99={p99:.2f}ms"
                        )
                        break
                if not matched_version:
                    logger.debug(
                        (
                            "No matching production model version found for benchmark "
                            "metrics"
                        ),
                        model_version=self._model_version,
                    )
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
            import mlflow

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
