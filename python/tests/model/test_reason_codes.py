from training.reason_codes import (
    BENCHMARK_STATUSES,
    CALIBRATION_SKIP_REASONS,
    DIAGNOSTICS_DEGRADED_REASONS,
    DRIFT_ERROR_CODES,
    DRIFT_FALLBACK_REASONS,
    DRIFT_RESOLUTION_MODES,
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
    assert DRIFT_ERROR_CODES == {
        "no_reference_data",
        "insufficient_reference_samples",
        "no_live_data",
        "insufficient_bucket_mass",
    }
    assert DRIFT_RESOLUTION_MODES == {"alias", "stage", "latest", "none"}
    assert BENCHMARK_STATUSES == {
        "skipped_disabled",
        "skipped_sampled_out",
        "success",
        "failed",
        "unknown",
    }
    assert DIAGNOSTICS_DEGRADED_REASONS == {
        "reload_failed",
        "schema_mismatch",
        "feature_coverage_warning",
    }
    assert RESUME_VALIDATION_REASONS == {
        "optuna_resume_legacy_study",
        "optuna_resume_invariant_mismatch",
        "optuna_resume_invariant_mismatch_strict",
    }
