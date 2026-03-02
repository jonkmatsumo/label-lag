"""Centralized reason/status code constants for ML hardening observability."""

from __future__ import annotations

from enum import Enum


class ReloadFailureReason(str, Enum):
    """Reason codes for failed model reload attempts."""

    ARTIFACT_MISSING = "artifact_missing"
    MLFLOW_FETCH = "mlflow_fetch"
    UNKNOWN = "unknown"


class SchemaMismatchReason(str, Enum):
    """Reason codes for schema mismatch diagnostics."""

    SCHEMA_MISMATCH = "schema_mismatch"


class CalibrationSkipReason(str, Enum):
    """Reason codes for skipped calibration fitting."""

    INSUFFICIENT_SAMPLES = "calibration_skipped_insufficient_samples"
    INSUFFICIENT_POSITIVES = "calibration_skipped_insufficient_positives"
    INSUFFICIENT_NEGATIVES = "calibration_skipped_insufficient_negatives"


class DriftFallbackReason(str, Enum):
    """Reason codes for drift bucketing fallback/guardrails."""

    TIED_QUANTILES = "tied_quantiles"
    INSUFFICIENT_BUCKET_MASS = "insufficient_bucket_mass"


class BenchmarkStatus(str, Enum):
    """Status codes for inference benchmark execution."""

    SKIPPED_DISABLED = "skipped_disabled"
    SKIPPED_SAMPLED_OUT = "skipped_sampled_out"
    SUCCESS = "success"
    FAILED = "failed"


class ModelManagerState(str, Enum):
    """Lifecycle states for ModelManager diagnostics."""

    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ReloadStatus(str, Enum):
    """Normalized reload status values in diagnostics snapshots."""

    IDLE = "idle"
    SUCCESS = "success"
    FAILED = "failed"


class ResumeValidationReason(str, Enum):
    """Reason codes for Optuna resume validation outcomes."""

    LEGACY_STUDY = "optuna_resume_legacy_study"
    INVARIANT_MISMATCH = "optuna_resume_invariant_mismatch"
    INVARIANT_MISMATCH_STRICT = "optuna_resume_invariant_mismatch_strict"


class DiagnosticsDegradedReason(str, Enum):
    """Bounded degraded-reason vocabulary for diagnostics snapshots."""

    RELOAD_FAILED = "reload_failed"
    SCHEMA_MISMATCH = SchemaMismatchReason.SCHEMA_MISMATCH.value
    FEATURE_COVERAGE_WARNING = "feature_coverage_warning"


RELOAD_FAILURE_REASONS = frozenset(reason.value for reason in ReloadFailureReason)
SCHEMA_MISMATCH_REASONS = frozenset(reason.value for reason in SchemaMismatchReason)
CALIBRATION_SKIP_REASONS = frozenset(reason.value for reason in CalibrationSkipReason)
DRIFT_FALLBACK_REASONS = frozenset(reason.value for reason in DriftFallbackReason)
BENCHMARK_STATUSES = frozenset(reason.value for reason in BenchmarkStatus)
MODEL_MANAGER_STATES = frozenset(reason.value for reason in ModelManagerState)
RELOAD_STATUSES = frozenset(reason.value for reason in ReloadStatus)
RESUME_VALIDATION_REASONS = frozenset(reason.value for reason in ResumeValidationReason)
DIAGNOSTICS_DEGRADED_REASONS = frozenset(
    reason.value for reason in DiagnosticsDegradedReason
)

# Diagnostics snapshot keys used by ModelManager.get_diagnostics().
DIAGNOSTIC_KEY_STATE = "state"
DIAGNOSTIC_KEY_MODEL_VERSION = "model_version"
DIAGNOSTIC_KEY_MODEL_SOURCE = "model_source"
DIAGNOSTIC_KEY_LAST_ERROR = "last_error"
DIAGNOSTIC_KEY_SCHEMA_MISMATCH_DETECTED = "schema_mismatch_detected"
DIAGNOSTIC_KEY_CALIBRATOR_LOADED = "calibrator_loaded"
DIAGNOSTIC_KEY_HAS_BUNDLE = "has_bundle"
DIAGNOSTIC_KEY_LAST_RELOAD_TS = "last_reload_ts"
DIAGNOSTIC_KEY_LAST_RELOAD_STATUS = "last_reload_status"
DIAGNOSTIC_KEY_LAST_RELOAD_REASON = "last_reload_reason"
DIAGNOSTIC_KEY_BENCHMARK_LAST_RUN_TS = "benchmark_last_run_ts"
DIAGNOSTIC_KEY_BENCHMARK_LAST_STATUS = "benchmark_last_status"
DIAGNOSTIC_KEY_DEGRADED_REASONS = "degraded_reasons"
DIAGNOSTIC_KEY_ACTIVE_MODEL_VERSION = "active_model_version"
DIAGNOSTIC_KEY_FEATURE_COVERAGE_WARNING_ACTIVE = "feature_coverage_warning_active"
DIAGNOSTIC_KEY_FEATURE_COVERAGE_WARNING_LAST_SEEN_TS = (
    "feature_coverage_warning_last_seen_ts"
)
DIAGNOSTIC_KEY_ML_TRAINING_RUN_ID = "ml.training.run_id"
DIAGNOSTIC_KEY_ML_MODEL_VERSION = "ml.model.version"
DIAGNOSTIC_KEY_ML_FEATURE_SCHEMA_HASH = "ml.feature.schema_hash"
DIAGNOSTIC_KEY_ML_HEALTH = "ml_health"
DIAGNOSTIC_KEY_ML_HEALTH = "ml_health"
DIAGNOSTIC_KEY_CONFIG = "config"

MODEL_MANAGER_BASELINE_DIAGNOSTIC_KEYS = frozenset(
    {
        DIAGNOSTIC_KEY_STATE,
        DIAGNOSTIC_KEY_ACTIVE_MODEL_VERSION,
        DIAGNOSTIC_KEY_LAST_RELOAD_STATUS,
        DIAGNOSTIC_KEY_SCHEMA_MISMATCH_DETECTED,
    }
)

# Keys for training identity payloads/artifacts.
TRAINING_IDENTITY_KEY_SCHEMA_VERSION = "schema_version"
TRAINING_IDENTITY_KEY_MLFLOW_RUN_ID = "mlflow_run_id"
TRAINING_IDENTITY_KEY_MODEL_NAME = "model_name"
TRAINING_IDENTITY_KEY_MODEL_VERSION = "model_version"
TRAINING_IDENTITY_KEY_FEATURE_SCHEMA_HASH = "feature_schema_hash"

# Trace/span metadata keys for model reload observability.
TRACE_KEY_ML_TRAINING_RUN_ID = "ml.training.run_id"
TRACE_KEY_ML_MODEL_VERSION = "ml.model.version"
TRACE_KEY_ML_FEATURE_SCHEMA_HASH = "ml.feature.schema_hash"

# Stable MLflow metadata keys used for auditing/reason tracking.
MLFLOW_PARAM_CALIBRATION_SKIP_REASON = "calibration_skip_reason"
MLFLOW_TAG_TUNING_RESUME_REASON = "tuning_resume_reason"
MLFLOW_TAG_TRAINING_CONFIG_HASH = "training_config_hash"
MLFLOW_TAG_FEATURE_SET_HASH = "feature_set_hash"
MLFLOW_TAG_FEATURE_SCHEMA_HASH = "feature_schema_hash"
MLFLOW_TAG_TRAINING_RUN_SPEC_VERSION = "training_run_spec_version"
MLFLOW_TAG_TRAINING_IDENTITY_RUN_ID = "training_identity.mlflow_run_id"
MLFLOW_TAG_TRAINING_IDENTITY_MODEL_VERSION = "training_identity.model_version"
MLFLOW_TAG_TRAINING_IDENTITY_FEATURE_SCHEMA_HASH = (
    "training_identity.feature_schema_hash"
)
MLFLOW_TAG_BEST_TRIAL_NUMBER = "best_trial_number"
MLFLOW_TAG_BEST_PARAMS_JSON = "best_params_json"
