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
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
import pandas as pd

from training.schemas import ErrorCategory

# Configure logging
logger = logging.getLogger(__name__)

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5005")

# Model registry name
MODEL_NAME = "ach-fraud-detection"

# Fallback model path
FALLBACK_MODEL_PATH = Path(__file__).parent.parent / "model" / "fallback_model.pkl"


@dataclass
class ModelStateBundle:
    """Bundle of model and associated metadata for atomic swap."""

    model: Any
    version: str
    source: str
    required_features: list[str]
    calibrator: Any
    calibrator_loaded: bool
    baseline_distribution: dict | None
    feature_importance: dict | None


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

        self._bundle: ModelStateBundle | None = None
        self._state: Literal["idle", "loading", "ready", "failed"] = "idle"
        self._last_error: str | None = None
        self._initialized = True

    def _transition_to(
        self,
        state: Literal["idle", "loading", "ready", "failed"],
        error: str | None = None,
    ) -> None:
        """Atomically transition to a new state and log it."""
        with self._lock:
            old_state = self._state
            self._state = state
            self._last_error = error
            logger.info(
                f"ModelManager state transition: {old_state} -> {state}"
                + (f" (error: {error})" if error else "")
            )

    @property
    def model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._bundle is not None and self._bundle.model is not None

    @property
    def calibrator_loaded(self) -> bool:
        """Check if a calibration artifact is loaded."""
        return self._bundle is not None and self._bundle.calibrator_loaded

    @property
    def calibrator(self):
        """Get the loaded calibrator or a default one."""
        if self._bundle and self._bundle.calibrator is not None:
            return self._bundle.calibrator

        # Fallback to default ScoreCalibrator
        from model.evaluate import ScoreCalibrator

        return ScoreCalibrator()

    @property
    def model_version(self) -> str:
        """Get the current model version."""
        if self._bundle:
            return self._bundle.version
        return "unknown"

    @property
    def model_source(self) -> str:
        """Get the model source (mlflow, fallback, or none)."""
        if self._bundle:
            return self._bundle.source
        return "none"

    @property
    def required_features(self) -> list[str]:
        """Get the list of required feature columns for this model.

        Returns:
            List of required feature column names. Falls back to default
            FEATURE_COLUMNS if not available.
        """
        if self._bundle and self._bundle.required_features:
            return self._bundle.required_features

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
        # Only transition to loading if we don't have a model yet (atomic swap)
        if not self.model_loaded:
            self._transition_to("loading")

        # Try loading from MLflow first
        bundle = self._load_from_mlflow()
        if bundle:
            with self._lock:
                self._bundle = bundle
            self._transition_to("ready")
            return True

        # Fall back to local model
        from forecast.metrics import model_fallback_total

        bundle = self._load_fallback_model()
        if bundle:
            model_fallback_total.labels(reason=ErrorCategory.MLFLOW_UNAVAILABLE).inc()
            with self._lock:
                self._bundle = bundle
            self._transition_to("ready")
            return True

        self._transition_to("failed", error="Both MLflow and fallback failed")
        logger.error("No model available - both MLflow and fallback failed")
        return False

    def _load_from_mlflow(self) -> ModelStateBundle | None:
        """Attempt to load model from MLflow registry.

        Returns:
            ModelStateBundle if successful, None otherwise.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_uri = f"models:/{MODEL_NAME}/Production"
            logger.info(f"Loading model from MLflow: {model_uri}")

            model = mlflow.pyfunc.load_model(model_uri)
            version = self._get_production_version()
            source = "mlflow"

            # Try to load required_features.json artifact (FF5)
            # Falls back to feature_columns.json if missing (legacy)
            required_features = self._load_required_features_artifact(version)

            # Validate loaded features against registry (Commit 7)
            from features.registry import FeatureRegistry

            if required_features:
                registered_features = set(FeatureRegistry.list_features())
                unknown_features = [
                    f for f in required_features if f not in registered_features
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

            # Try to load calibrator.pkl artifact (C2)
            calibrator, calibrator_loaded = self._load_calibrator_artifact()

            # Try to load score_distribution.json artifact (C3)
            baseline_distribution = self._load_baseline_distribution_artifact()

            # Cache feature importance (C4)
            feature_importance = self.get_feature_importance_from_model(
                model, required_features or []
            )

            logger.info(f"Successfully loaded model version {version} from MLflow")

            bundle = ModelStateBundle(
                model=model,
                version=version,
                source=source,
                required_features=required_features or [],
                calibrator=calibrator,
                calibrator_loaded=calibrator_loaded,
                baseline_distribution=baseline_distribution,
                feature_importance=feature_importance,
            )

            # Benchmark inference latency after load
            self._benchmark_inference(bundle)

            return bundle

        except Exception as e:
            logger.critical(
                f"Failed to load model from MLflow/MinIO: {e}. "
                "Attempting fallback to local model."
            )
            return None

    def _load_required_features_artifact(self, version: str) -> list[str] | None:
        """Load required_features.json artifact from the model run.

        Attempts to load required_features.json (new format).
        Falls back to feature_columns.json (legacy format) if missing.
        """
        try:
            import mlflow

            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if v.current_stage == "Production" and v.version == version.lstrip("v"):
                    run_id = v.run_id
                    # 1. Try new format: required_features.json
                    try:
                        path = client.download_artifacts(
                            run_id, "required_features.json"
                        )
                        with open(path) as f:
                            data = json.load(f)
                        return data["features"]
                    except Exception:
                        logger.debug("required_features.json not found, trying legacy")

                    # 2. Try legacy format: feature_columns.json
                    try:
                        path = client.download_artifacts(run_id, "feature_columns.json")
                        with open(path) as f:
                            return json.load(f)
                    except Exception:
                        logger.warning("No feature artifacts found.")
                        break
        except Exception as e:
            logger.debug(f"Could not load required features metadata: {e}")
        return None

    def _load_calibrator_artifact(self) -> tuple[Any, bool]:
        """Load calibrator.pkl artifact from the model run.

        Returns:
            Tuple of (calibrator, loaded_flag).
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
                        return joblib.load(artifact_path), True
                    except Exception as e:
                        logger.debug(
                            f"calibrator.pkl artifact not found or failed to load: {e}"
                        )
                        break
        except Exception as e:
            logger.debug(f"Could not load calibrator artifact: {e}")
        return None, False

    def _load_baseline_distribution_artifact(self) -> dict | None:
        """Load score_distribution.json artifact from the model run.

        Returns:
            Score distribution dict if found, None otherwise.
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
                            return json.load(f)
                    except Exception as e:
                        logger.debug(f"score_distribution.json artifact not found: {e}")
                        break
        except Exception as e:
            logger.debug(f"Could not load baseline distribution artifact: {e}")
        return None

    @property
    def baseline_distribution(self) -> dict | None:
        """Get the baseline score distribution."""
        if self._bundle:
            return self._bundle.baseline_distribution
        return None

    @property
    def cached_feature_importance(self) -> dict[str, float] | None:
        """Get the cached feature importance."""
        if self._bundle:
            return self._bundle.feature_importance
        return None

    def _benchmark_inference(
        self, bundle: ModelStateBundle, n_samples: int = 100
    ) -> None:
        """Benchmark inference latency and log to MLflow.

        Args:
            bundle: The model bundle to benchmark.
            n_samples: Number of samples to use for benchmarking.
        """
        if bundle.model is None:
            return

        try:
            import mlflow

            # Create sample data matching required features
            required = bundle.required_features
            rng = np.random.default_rng(0)
            sample_data = pd.DataFrame(
                {feat: rng.random(n_samples) for feat in required}
            )

            # Measure latencies
            latencies_ms = []
            for _ in range(n_samples):
                start = time.perf_counter()
                if bundle.source == "mlflow":
                    bundle.model.predict(sample_data.iloc[[0]])
                else:
                    bundle.model.predict_proba(sample_data.iloc[[0]])
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
                matched_version = False
                for v in versions:
                    if (
                        v.current_stage == "Production"
                        and f"v{v.version}" == bundle.version
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

    def _load_fallback_model(self) -> ModelStateBundle | None:
        """Attempt to load fallback model from local file.

        Returns:
            ModelStateBundle if successful, None otherwise.
        """
        if not FALLBACK_MODEL_PATH.exists():
            logger.warning(f"Fallback model not found at {FALLBACK_MODEL_PATH}")
            return None

        try:
            logger.info(f"Loading fallback model from {FALLBACK_MODEL_PATH}")

            with open(FALLBACK_MODEL_PATH, "rb") as f:
                model = pickle.load(f)

            version = "fallback"
            source = "fallback"

            # Use default features for fallback
            from model.loader import DataLoader

            required_features = DataLoader.FEATURE_COLUMNS

            return ModelStateBundle(
                model=model,
                version=version,
                source=source,
                required_features=required_features,
                calibrator=None,
                calibrator_loaded=False,
                baseline_distribution=None,
                feature_importance=self.get_feature_importance_from_model(
                    model, required_features
                ),
            )

        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return None

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
            RuntimeError: If no model is loaded or in loading state.
        """
        state = self._state
        bundle = self._bundle

        if state == "loading":
            raise RuntimeError("Model reload in progress", "reload_in_progress")

        if bundle is None or bundle.model is None:
            if state == "failed":
                raise RuntimeError(
                    f"Model loading failed: {self._last_error}", "load_failed"
                )
            raise RuntimeError(
                "No model loaded. Call load_production_model() first.", "not_loaded"
            )

        if state == "failed":
            logger.warning(
                "Serving from old model bundle because latest reload failed: "
                f"{self._last_error}"
            )

        try:
            # MLflow pyfunc models return predictions directly
            if bundle.source == "mlflow":
                predictions = bundle.model.predict(features)
            else:
                # Fallback sklearn model - get probability of positive class
                predictions = bundle.model.predict_proba(features)[:, 1]

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
            RuntimeError: If no model is loaded or in loading state.
        """
        # predict() will handle state checks
        bundle = self._bundle
        if bundle is None:
            # Re-check state if bundle is None to provide better error
            state = self._state
            if state == "loading":
                raise RuntimeError("Model reload in progress", "reload_in_progress")
            if state == "failed":
                raise RuntimeError(
                    f"Model loading failed: {self._last_error}", "load_failed"
                )
            raise RuntimeError(
                "No model loaded. Call load_production_model() first.", "not_loaded"
            )

        # Validate that all required features are present
        required = bundle.required_features
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

    def get_feature_importance_from_model(
        self, model: Any, feature_names: list[str]
    ) -> dict[str, float] | None:
        """Extract feature importance from a model instance.

        Args:
            model: The model instance (MLflow pyfunc or sklearn).
            feature_names: List of feature names in order.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if model is None:
            return None

        try:

            def _iter_candidates(model_obj):
                queue = [model_obj]
                seen: set[int] = set()
                attrs = ("_model_impl", "_model", "model")
                while queue:
                    candidate = queue.pop(0)
                    if candidate is None:
                        continue
                    candidate_id = id(candidate)
                    if candidate_id in seen:
                        continue
                    seen.add(candidate_id)
                    yield candidate
                    for attr in attrs:
                        if hasattr(candidate, attr):
                            queue.append(getattr(candidate, attr, None))

            dense_importances = None
            for candidate in _iter_candidates(model):
                if hasattr(candidate, "feature_importances_"):
                    dense_importances = np.asarray(candidate.feature_importances_)
                    break

            if dense_importances is None:
                logger.warning(
                    "Could not extract dense feature_importances_ from loaded model."
                )
                return None

            if len(dense_importances) != len(feature_names):
                logger.warning(
                    "Feature importance length mismatch: importances=%s features=%s. "
                    "Returning best-effort mapping.",
                    len(dense_importances),
                    len(feature_names),
                )

            overlap = min(len(dense_importances), len(feature_names))
            importance_map = {
                feature_names[i]: float(dense_importances[i]) for i in range(overlap)
            }
            if len(feature_names) > overlap:
                for feature_name in feature_names[overlap:]:
                    importance_map[feature_name] = 0.0
            return importance_map
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
