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


class DriftErrorCode(str, Enum):
    """Canonical top-level drift error codes."""

    NO_REFERENCE_DATA = "no_reference_data"
    INSUFFICIENT_REFERENCE_SAMPLES = "insufficient_reference_samples"
    NO_LIVE_DATA = "no_live_data"
    INSUFFICIENT_BUCKET_MASS = DriftFallbackReason.INSUFFICIENT_BUCKET_MASS.value


class DriftResolutionMode(str, Enum):
    """Canonical drift reference resolution modes."""

    ALIAS = "alias"
    STAGE = "stage"
    LATEST = "latest"
    NONE = "none"


class DriftReferenceResolutionWarning(str, Enum):
    """Canonical warning codes describing reference resolution fallback behavior."""

    ALIAS_NOT_FOUND_FALLBACK = "alias_not_found_fallback"
    ALIAS_AMBIGUOUS_SELECTED_HIGHEST = "alias_ambiguous_selected_highest"
    STAGE_FALLBACK_USED = "stage_fallback_used"
    LATEST_FALLBACK_USED = "latest_fallback_used"
    NO_REFERENCE_VERSIONS_AVAILABLE = "no_reference_versions_available"


class DriftFeatureStatus(str, Enum):
    """Canonical per-feature drift status labels."""

    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class BenchmarkStatus(str, Enum):
    """Status codes for inference benchmark execution."""

    SKIPPED_DISABLED = "skipped_disabled"
    SKIPPED_SAMPLED_OUT = "skipped_sampled_out"
    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"


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


class OperabilityStatus(str, Enum):
    """Canonical operator-facing status vocabulary for health summaries."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    NOT_RUN = "not_run"


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


class DiagnosticsWarningCode(str, Enum):
    """Compact bounded warning codes for operator-facing diagnostics summaries."""

    SCHEMA_MISMATCH_DETECTED = "schema_mismatch_detected"
    RELOAD_FAILED_USING_LAST_KNOWN_GOOD = "reload_failed_using_last_known_good"
    FEATURE_COVERAGE_BELOW_THRESHOLD = "feature_coverage_below_threshold"
    DRIFT_REFERENCE_UNAVAILABLE = "drift_reference_unavailable"


RELOAD_FAILURE_REASONS = frozenset(reason.value for reason in ReloadFailureReason)
SCHEMA_MISMATCH_REASONS = frozenset(reason.value for reason in SchemaMismatchReason)
CALIBRATION_SKIP_REASONS = frozenset(reason.value for reason in CalibrationSkipReason)
DRIFT_FALLBACK_REASONS = frozenset(reason.value for reason in DriftFallbackReason)
DRIFT_ERROR_CODES = frozenset(reason.value for reason in DriftErrorCode)
DRIFT_RESOLUTION_MODES = frozenset(reason.value for reason in DriftResolutionMode)
DRIFT_REFERENCE_RESOLUTION_WARNING_CODES = frozenset(
    reason.value for reason in DriftReferenceResolutionWarning
)
DRIFT_FEATURE_STATUSES = frozenset(reason.value for reason in DriftFeatureStatus)
BENCHMARK_STATUSES = frozenset(reason.value for reason in BenchmarkStatus)
MODEL_MANAGER_STATES = frozenset(reason.value for reason in ModelManagerState)
RELOAD_STATUSES = frozenset(reason.value for reason in ReloadStatus)
OPERABILITY_STATUSES = frozenset(reason.value for reason in OperabilityStatus)
RESUME_VALIDATION_REASONS = frozenset(reason.value for reason in ResumeValidationReason)
DIAGNOSTICS_DEGRADED_REASONS = frozenset(
    reason.value for reason in DiagnosticsDegradedReason
)
DIAGNOSTICS_WARNING_CODES = frozenset(reason.value for reason in DiagnosticsWarningCode)

# Diagnostics snapshot keys used by ModelManager.get_diagnostics().
DIAGNOSTIC_KEY_STATE = "state"
DIAGNOSTIC_KEY_STATUS = "status"
DIAGNOSTIC_KEY_WARNINGS = "warnings"
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
DIAGNOSTIC_KEY_FEATURE_COVERAGE_LAST_RATIO = "feature_coverage_last_ratio"
DIAGNOSTIC_KEY_FEATURE_COVERAGE_WARNING_LAST_SEEN_TS = (
    "feature_coverage_warning_last_seen_ts"
)
DIAGNOSTIC_KEY_FEATURE_COVERAGE_LAST_RATIO = "feature_coverage_last_ratio"
DIAGNOSTIC_KEY_ML_TRAINING_RUN_ID = "ml.training.run_id"
DIAGNOSTIC_KEY_ML_MODEL_VERSION = "ml.model.version"
DIAGNOSTIC_KEY_ML_FEATURE_SCHEMA_HASH = "ml.feature.schema_hash"
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
