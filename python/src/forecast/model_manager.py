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
    schema_mismatch_detected: bool = False
    last_reload_ts: float | None = None


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
        self._benchmarked_versions: set[str] = set()
        # Backward-compatible legacy fields used by older tests/callers.
        self._model: Any = None
        self._model_version: str = "unknown"
        self._model_source: str = "none"
        self._required_features: list[str] = []
        self._calibrator: Any = None
        self._calibrator_loaded: bool = False
        self._baseline_distribution: dict | None = None
        self._feature_importance: dict[str, float] | None = None
        self._schema_mismatch_detected: bool = False
        self._mlflow_failure_reason: str = "unknown"
        self._benchmark_last_run_ts: float | None = None
        self._benchmark_last_status: str | None = None
        self._initialized = True

    @staticmethod
    def _bundle_like(candidate: Any) -> bool:
        """Check whether an object looks like a model bundle."""
        return (
            candidate is not None
            and hasattr(candidate, "model")
            and hasattr(candidate, "version")
            and hasattr(candidate, "source")
            and hasattr(candidate, "required_features")
        )

    @staticmethod
    def _coerce_feature_names(features: Any) -> list[str]:
        """Normalize feature-name collections into a list of strings."""
        if features is None:
            return []
        if isinstance(features, list):
            return [str(name) for name in features]
        if isinstance(features, tuple):
            return [str(name) for name in features]
        if isinstance(features, (str, bytes)):
            return []
        try:
            return [str(name) for name in list(features)]
        except Exception:
            return []

    @staticmethod
    def _normalize_calibrator_result(result: Any) -> tuple[Any, bool]:
        """Handle mocked artifact loader return shapes safely."""
        if isinstance(result, tuple) and len(result) == 2:
            calibrator, loaded = result
            return calibrator, bool(loaded)
        return None, False

    def _resolve_runtime_bundle(
        self, bundle: ModelStateBundle | None = None
    ) -> ModelStateBundle | Any | None:
        """Resolve bundle from explicit arg, current bundle, or legacy fields."""
        if self._bundle_like(bundle):
            return bundle

        current = self._bundle
        if self._bundle_like(current):
            return current

        legacy_model = getattr(self, "_model", None)
        if legacy_model is None:
            return None

        legacy_source = str(getattr(self, "_model_source", "") or "")
        if legacy_source in {"", "none"}:
            legacy_source = "mlflow"

        return ModelStateBundle(
            model=legacy_model,
            version=str(getattr(self, "_model_version", "unknown")),
            source=legacy_source,
            required_features=self._coerce_feature_names(
                getattr(self, "_required_features", [])
            ),
            calibrator=getattr(self, "_calibrator", None),
            calibrator_loaded=bool(getattr(self, "_calibrator_loaded", False)),
            baseline_distribution=getattr(self, "_baseline_distribution", None),
            feature_importance=getattr(self, "_feature_importance", None),
            last_reload_ts=time.time(),
        )

    def _sync_legacy_from_bundle(self, bundle: Any) -> None:
        """Keep legacy private fields in sync with the active bundle."""
        self._model = getattr(bundle, "model", None)
        self._model_version = str(getattr(bundle, "version", "unknown"))
        self._model_source = str(getattr(bundle, "source", "none"))
        self._required_features = self._coerce_feature_names(
            getattr(bundle, "required_features", [])
        )
        self._calibrator = getattr(bundle, "calibrator", None)
        self._calibrator_loaded = bool(getattr(bundle, "calibrator_loaded", False))
        self._baseline_distribution = getattr(bundle, "baseline_distribution", None)
        self._feature_importance = getattr(bundle, "feature_importance", None)
        self._schema_mismatch_detected = bool(
            getattr(bundle, "schema_mismatch_detected", False)
        )

    def _store_loaded_bundle_if_valid(self, bundle: Any) -> bool:
        """Store and sync a valid bundle-like object."""
        if not self._bundle_like(bundle):
            return False

        with self._lock:
            self._bundle = bundle
            self._sync_legacy_from_bundle(bundle)
        return True

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
        bundle = self._resolve_runtime_bundle()
        return bundle is not None and getattr(bundle, "model", None) is not None

    @property
    def calibrator_loaded(self) -> bool:
        """Check if a calibration artifact is loaded."""
        bundle = self._resolve_runtime_bundle()
        if bundle is not None:
            return bool(getattr(bundle, "calibrator_loaded", False))
        return bool(getattr(self, "_calibrator_loaded", False))

    @property
    def calibrator(self):
        """Get the loaded calibrator or a default one."""
        bundle = self._resolve_runtime_bundle()
        loaded_calibrator = getattr(bundle, "calibrator", None) if bundle else None
        if loaded_calibrator is None:
            loaded_calibrator = getattr(self, "_calibrator", None)
        if loaded_calibrator is not None:
            return loaded_calibrator

        # Fallback to default ScoreCalibrator
        from model.evaluate import ScoreCalibrator

        return ScoreCalibrator()

    @property
    def model_version(self) -> str:
        """Get the current model version."""
        bundle = self._resolve_runtime_bundle()
        if bundle is not None:
            version = getattr(bundle, "version", None)
            if version:
                return str(version)
        legacy_version = getattr(self, "_model_version", None)
        if legacy_version:
            return str(legacy_version)
        return "unknown"

    @property
    def model_source(self) -> str:
        """Get the model source (mlflow, fallback, or none)."""
        bundle = self._resolve_runtime_bundle()
        if bundle is not None:
            source = getattr(bundle, "source", None)
            if source:
                return str(source)
        legacy_source = getattr(self, "_model_source", None)
        if legacy_source:
            return str(legacy_source)
        return "none"

    @property
    def schema_mismatch_detected(self) -> bool:
        """Check if a feature schema mismatch was detected for the active model."""
        bundle = self._resolve_runtime_bundle()
        if bundle is not None:
            return bool(getattr(bundle, "schema_mismatch_detected", False))
        return bool(getattr(self, "_schema_mismatch_detected", False))

    @property
    def required_features(self) -> list[str]:
        """Get the list of required feature columns for this model."""
        bundle = self._resolve_runtime_bundle()
        if bundle is not None:
            resolved = self._coerce_feature_names(
                getattr(bundle, "required_features", None)
            )
            if resolved:
                return resolved

        legacy = self._coerce_feature_names(getattr(self, "_required_features", None))
        if legacy:
            return legacy

        # Fallback to default feature columns
        from model.loader import DataLoader

        return DataLoader.FEATURE_COLUMNS

    def get_diagnostics(self) -> dict[str, Any]:
        """Get a diagnostic snapshot of the ModelManager state.

        Returns:
            Dict containing state, version, source, error, and flags.
        """
        with self._lock:
            bundle = self._resolve_runtime_bundle()
            return {
                "state": self._state,
                "model_version": self.model_version,
                "model_source": self.model_source,
                "last_error": self._last_error,
                "schema_mismatch_detected": self.schema_mismatch_detected,
                "calibrator_loaded": self.calibrator_loaded,
                "has_bundle": bundle is not None,
                "last_reload_ts": getattr(bundle, "last_reload_ts", None)
                if bundle
                else None,
                "last_reload_status": "success"
                if self._state == "ready"
                else "failed"
                if self._state == "failed"
                else "idle",
                "benchmark_last_run_ts": self._benchmark_last_run_ts,
                "benchmark_last_status": self._benchmark_last_status,
                "active_model_version": self.model_version,
            }

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
        if self._store_loaded_bundle_if_valid(bundle):
            self._transition_to("ready")
            return True
        if bundle:
            # Backward-compatibility for tests that mock loader methods as booleans.
            logger.warning(
                "Model loader returned non-bundle truthy value; treating as success."
            )
            self._transition_to("ready")
            return True

        # Fall back to local model
        from forecast.metrics import model_fallback_total

        bundle = self._load_fallback_model()
        if bundle:
            model_fallback_total.labels(reason=ErrorCategory.MLFLOW_UNAVAILABLE).inc()
            if self._store_loaded_bundle_if_valid(bundle):
                self._transition_to("ready")
                return True
            logger.warning(
                "Fallback loader returned non-bundle truthy value; treating as success."
            )
            self._transition_to("ready")
            return True

        self._transition_to("failed", error="Both MLflow and fallback failed")
        logger.error("No model available - both MLflow and fallback failed")
        from forecast.metrics import model_reload_failure_total

        # Use mlflow failure reason as the primary reason if both fail
        reason = self._mlflow_failure_reason
        model_reload_failure_total.labels(reason=reason).inc()
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
            required_features = self._coerce_feature_names(
                self._load_required_features_artifact(version)
            )

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
            calibrator, calibrator_loaded = self._normalize_calibrator_result(
                self._load_calibrator_artifact()
            )

            # Try to load score_distribution.json artifact (C3)
            baseline_distribution = self._load_baseline_distribution_artifact()

            # Cache feature importance (C4)
            feature_importance = self.get_feature_importance_from_model(
                model, required_features
            )

            logger.info(f"Successfully loaded model version {version} from MLflow")

            # Phase 2.2: Validate feature schema hash
            schema_mismatch = False
            try:
                import hashlib

                stored_hash_info = self._load_feature_schema_hash_artifact(version)
                if stored_hash_info:
                    stored_hash = stored_hash_info.get("feature_schema_hash")
                    # Compute expected hash from inference-side required_features
                    ordered_features = sorted(required_features)
                    expected_json = json.dumps(ordered_features)
                    expected_hash = hashlib.sha256(
                        expected_json.encode("utf-8")
                    ).hexdigest()

                    if stored_hash and stored_hash != expected_hash:
                        schema_mismatch = True
                        logger.warning(
                            "Feature schema hash mismatch detected! "
                            "Stored: %s (count=%s), "
                            "Expected: %s (count=%d). "
                            "Inference feature ordering or set may "
                            "differ from training.",
                            stored_hash,
                            stored_hash_info.get("feature_count"),
                            expected_hash,
                            len(ordered_features),
                        )
                        from forecast.metrics import model_schema_mismatch_total

                        model_schema_mismatch_total.inc()
            except Exception as e:
                logger.debug(f"Feature schema hash validation failed: {e}")

            bundle = ModelStateBundle(
                model=model,
                version=version,
                source=source,
                required_features=required_features,
                calibrator=calibrator,
                calibrator_loaded=calibrator_loaded,
                baseline_distribution=baseline_distribution,
                feature_importance=feature_importance,
                schema_mismatch_detected=schema_mismatch,
                last_reload_ts=time.time(),
            )

            # Benchmark inference latency after load
            self._benchmark_inference(bundle, log_to_mlflow=False)

            return bundle

        except Exception as e:
            error_str = str(e).lower()
            if "artifact" in error_str or "not found" in error_str:
                self._mlflow_failure_reason = "artifact_missing"
            elif (
                "mlflow" in error_str
                or "connection" in error_str
                or "http" in error_str
            ):
                self._mlflow_failure_reason = "mlflow_fetch"
            else:
                self._mlflow_failure_reason = "unknown"

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

    def _load_feature_schema_hash_artifact(self, version: str) -> dict | None:
        """Load feature_schema_hash.json artifact from the model run."""
        try:
            import mlflow

            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for v in versions:
                if v.current_stage == "Production" and v.version == version.lstrip("v"):
                    run_id = v.run_id
                    try:
                        path = client.download_artifacts(
                            run_id, "feature_schema_hash.json"
                        )
                        with open(path) as f:
                            return json.load(f)
                    except Exception:
                        break
        except Exception as e:
            logger.debug(f"Could not load feature schema hash artifact: {e}")
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
        bundle = self._resolve_runtime_bundle()
        if bundle is not None:
            return getattr(bundle, "baseline_distribution", None)
        if getattr(self, "_baseline_distribution", None) is not None:
            return self._baseline_distribution
        return None

    @property
    def cached_feature_importance(self) -> dict[str, float] | None:
        """Get the cached feature importance."""
        bundle = self._resolve_runtime_bundle()
        if bundle is not None:
            return getattr(bundle, "feature_importance", None)
        if getattr(self, "_feature_importance", None) is not None:
            return self._feature_importance
        return None

    def get_feature_importance(self) -> dict[str, float] | None:
        """Backward-compatible accessor for feature importance extraction."""
        cached = self.cached_feature_importance
        if cached is not None:
            return cached

        bundle = self._resolve_runtime_bundle()
        if bundle is None:
            return None

        importance = self.get_feature_importance_from_model(
            getattr(bundle, "model", None),
            self._coerce_feature_names(getattr(bundle, "required_features", [])),
        )
        if importance is not None:
            self._feature_importance = importance
            if self._bundle_like(self._bundle):
                try:
                    self._bundle.feature_importance = importance
                except Exception:
                    pass
        return importance

    def _benchmark_inference(
        self,
        bundle: ModelStateBundle | None = None,
        n_samples: int = 100,
        *,
        log_to_mlflow: bool | None = None,
    ) -> None:
        """Benchmark inference latency and emit runtime metrics.

        Args:
            bundle: Optional bundle to benchmark.
                If omitted, resolves from current state.
            n_samples: Number of samples to use for benchmarking.
            log_to_mlflow: Whether to also emit benchmark metrics to MLflow run.
                Defaults to True for explicit bundle calls and False otherwise.
        """
        if log_to_mlflow is None:
            log_to_mlflow = bundle is not None

        runtime_bundle = self._resolve_runtime_bundle(bundle)
        model = getattr(runtime_bundle, "model", None) if runtime_bundle else None
        if model is None:
            return

        version = str(getattr(runtime_bundle, "version", None) or self.model_version)
        if version in self._benchmarked_versions:
            logger.debug(f"Skipping benchmark for version {version} (already run)")
            return

        logger.info(f"Starting inference benchmark for version {version}")

        try:
            sample_count = max(int(n_samples), 1)
            required = self._coerce_feature_names(
                getattr(runtime_bundle, "required_features", [])
            )
            rng = np.random.default_rng(0)
            sample_data = pd.DataFrame(
                {feat: rng.random(sample_count) for feat in required},
                index=range(sample_count),
            )

            # Measure latencies
            source = str(getattr(runtime_bundle, "source", "mlflow") or "mlflow")
            use_predict = source == "mlflow" or not hasattr(model, "predict_proba")
            latencies_ms = []
            for _ in range(sample_count):
                start = time.perf_counter()
                if use_predict:
                    model.predict(sample_data.iloc[[0]])
                else:
                    model.predict_proba(sample_data.iloc[[0]])
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies_ms.append(elapsed_ms)

            # Calculate percentiles
            latencies_sorted = sorted(latencies_ms)
            p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
            p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

            # Emit runtime metrics only. Do not fail model load on metrics failures.
            try:
                from forecast.metrics import (
                    inference_benchmark_percentile_latency_ms,
                    inference_benchmark_sample_latency_ms,
                )

                for latency in latencies_ms:
                    inference_benchmark_sample_latency_ms.observe(latency)

                inference_benchmark_percentile_latency_ms.labels(percentile="p50").set(
                    p50
                )
                inference_benchmark_percentile_latency_ms.labels(percentile="p95").set(
                    p95
                )
                inference_benchmark_percentile_latency_ms.labels(percentile="p99").set(
                    p99
                )
                logger.info(
                    f"Benchmark inference latency for {version}: p50={p50:.2f}ms, "
                    f"p95={p95:.2f}ms, p99={p99:.2f}ms"
                )
            except Exception as e:
                logger.debug(f"Could not emit inference benchmark runtime metrics: {e}")

            if log_to_mlflow:
                try:
                    import mlflow

                    client = mlflow.MlflowClient()
                    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                    target_version = version.lstrip("v")
                    run_id = None
                    for v in versions:
                        if (
                            v.current_stage == "Production"
                            and str(v.version) == target_version
                        ):
                            run_id = v.run_id
                            break
                    if run_id:
                        client.log_metric(run_id, "inference_latency_p50_ms", p50)
                        client.log_metric(run_id, "inference_latency_p95_ms", p95)
                        client.log_metric(run_id, "inference_latency_p99_ms", p99)
                except Exception as e:
                    logger.debug(
                        f"Could not emit inference benchmark MLflow metrics: {e}"
                    )

            # Mark as benchmarked even if metrics emit fails, to avoid retries.
            self._benchmarked_versions.add(version)
            self._benchmark_last_status = "success"
        except Exception as e:
            logger.debug(f"Inference benchmarking failed: {e}")
            self._benchmark_last_status = "failed"

        self._benchmark_last_run_ts = time.time()

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
                last_reload_ts=time.time(),
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
        bundle = self._resolve_runtime_bundle()

        if state == "loading":
            raise RuntimeError("Model reload in progress", "reload_in_progress")

        if bundle is None or getattr(bundle, "model", None) is None:
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
            model = getattr(bundle, "model", None)
            source = str(getattr(bundle, "source", "mlflow") or "mlflow")
            # MLflow pyfunc models return predictions directly
            if source == "mlflow":
                predictions = model.predict(features)
            else:
                # Fallback sklearn model - get probability of positive class
                predictions = model.predict_proba(features)[:, 1]

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
        bundle = self._resolve_runtime_bundle()
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
        required = self._coerce_feature_names(getattr(bundle, "required_features", []))
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
                # Safety limit to prevent infinite loop
                iterations = 0
                while queue and iterations < 1000:
                    iterations += 1
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

            n_importances = len(dense_importances)
            n_features = len(feature_names)

            if n_importances != n_features:
                logger.warning(
                    "Feature importance alignment mismatch: model has %s importances "
                    "but registry/metadata expects %s features. "
                    "Using first-N mapping.",
                    n_importances,
                    n_features,
                )

            # Explicitly zip and order by registry/required order
            importance_map = {}
            for i, name in enumerate(feature_names):
                if i < n_importances:
                    importance_map[name] = float(dense_importances[i])
                else:
                    # Registry expects a feature the model doesn't have importance for
                    importance_map[name] = 0.0

            logger.info(
                "Feature importance aligned for version %s (mode: %s)",
                self.model_version,
                "strict" if n_importances == n_features else "best_effort",
            )
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
