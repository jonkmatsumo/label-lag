"""Reason/status vocabulary guardrails for ML operability."""

from training.reason_codes import (
    BENCHMARK_STATUSES,
    CALIBRATION_SKIP_REASONS,
    DRIFT_FALLBACK_REASONS,
    MLFLOW_PARAM_CALIBRATION_SKIP_REASON,
    MLFLOW_TAG_TUNING_RESUME_REASON,
    MODEL_MANAGER_STATES,
    RELOAD_FAILURE_REASONS,
    RELOAD_STATUSES,
    RESUME_VALIDATION_REASONS,
    SCHEMA_MISMATCH_REASONS,
)


def test_reason_code_sets_are_bounded_and_stable():
    assert RELOAD_FAILURE_REASONS == {"artifact_missing", "mlflow_fetch", "unknown"}
    assert SCHEMA_MISMATCH_REASONS == {"schema_mismatch"}
    assert CALIBRATION_SKIP_REASONS == {
        "calibration_skipped_insufficient_samples",
        "calibration_skipped_insufficient_positives",
        "calibration_skipped_insufficient_negatives",
    }
    assert DRIFT_FALLBACK_REASONS == {"tied_quantiles", "insufficient_bucket_mass"}
    assert BENCHMARK_STATUSES == {
        "skipped_disabled",
        "skipped_sampled_out",
        "success",
        "failed",
    }
    assert MODEL_MANAGER_STATES == {"idle", "loading", "ready", "failed"}
    assert RELOAD_STATUSES == {"idle", "success", "failed"}
    assert RESUME_VALIDATION_REASONS == {
        "optuna_resume_legacy_study",
        "optuna_resume_invariant_mismatch",
        "optuna_resume_invariant_mismatch_strict",
    }


def test_mlflow_reason_metadata_keys_stable():
    assert MLFLOW_PARAM_CALIBRATION_SKIP_REASON == "calibration_skip_reason"
    assert MLFLOW_TAG_TUNING_RESUME_REASON == "tuning_resume_reason"
