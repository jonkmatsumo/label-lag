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


class ResumeValidationReason(str, Enum):
    """Reason codes for Optuna resume validation outcomes."""

    LEGACY_STUDY = "optuna_resume_legacy_study"
    INVARIANT_MISMATCH = "optuna_resume_invariant_mismatch"
    INVARIANT_MISMATCH_STRICT = "optuna_resume_invariant_mismatch_strict"


class DiagnosticsDegradedReason(str, Enum):
    """Bounded degraded-reason vocabulary for diagnostics snapshots."""

    RELOAD_FAILED = "reload_failed"
    FEATURE_COVERAGE_WARNING = "feature_coverage_warning"


RELOAD_FAILURE_REASONS = frozenset(reason.value for reason in ReloadFailureReason)
SCHEMA_MISMATCH_REASONS = frozenset(reason.value for reason in SchemaMismatchReason)
CALIBRATION_SKIP_REASONS = frozenset(reason.value for reason in CalibrationSkipReason)
DRIFT_FALLBACK_REASONS = frozenset(reason.value for reason in DriftFallbackReason)
BENCHMARK_STATUSES = frozenset(reason.value for reason in BenchmarkStatus)
RESUME_VALIDATION_REASONS = frozenset(reason.value for reason in ResumeValidationReason)

MLFLOW_PARAM_CALIBRATION_SKIP_REASON = "calibration_skip_reason"
MLFLOW_TAG_TUNING_RESUME_REASON = "tuning_resume_reason"
