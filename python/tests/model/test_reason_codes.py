from training.reason_codes import (
    BENCHMARK_STATUSES,
    CALIBRATION_SKIP_REASONS,
    DRIFT_FALLBACK_REASONS,
    RELOAD_FAILURE_REASONS,
    RESUME_VALIDATION_REASONS,
    SCHEMA_MISMATCH_REASONS,
)


def test_reason_code_vocabulary_is_stable():
    """Guard against accidental reason-code drift across observability surfaces."""
    assert RELOAD_FAILURE_REASONS == {
        "artifact_missing",
        "mlflow_fetch",
        "unknown",
    }
    assert SCHEMA_MISMATCH_REASONS == {"schema_mismatch"}
    assert CALIBRATION_SKIP_REASONS == {
        "calibration_skipped_insufficient_samples",
        "calibration_skipped_insufficient_positives",
        "calibration_skipped_insufficient_negatives",
    }
    assert DRIFT_FALLBACK_REASONS == {
        "tied_quantiles",
        "insufficient_bucket_mass",
    }
    assert BENCHMARK_STATUSES == {
        "skipped_disabled",
        "skipped_sampled_out",
        "success",
        "failed",
    }
    assert RESUME_VALIDATION_REASONS == {
        "optuna_resume_legacy_study",
        "optuna_resume_invariant_mismatch",
        "optuna_resume_invariant_mismatch_strict",
    }
