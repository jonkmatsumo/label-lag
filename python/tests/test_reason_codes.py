"""Reason/status vocabulary guardrails for ML operability."""

from training.reason_codes import (
    BENCHMARK_STATUSES,
    CALIBRATION_SKIP_REASONS,
    DIAGNOSTIC_KEY_ACTIVE_MODEL_VERSION,
    DIAGNOSTIC_KEY_LAST_RELOAD_STATUS,
    DIAGNOSTIC_KEY_ML_FEATURE_SCHEMA_HASH,
    DIAGNOSTIC_KEY_ML_MODEL_VERSION,
    DIAGNOSTIC_KEY_ML_TRAINING_RUN_ID,
    DIAGNOSTIC_KEY_SCHEMA_MISMATCH_DETECTED,
    DIAGNOSTIC_KEY_STATE,
    DRIFT_FALLBACK_REASONS,
    MLFLOW_PARAM_CALIBRATION_SKIP_REASON,
    MLFLOW_TAG_BEST_PARAMS_JSON,
    MLFLOW_TAG_BEST_TRIAL_NUMBER,
    MLFLOW_TAG_FEATURE_SCHEMA_HASH,
    MLFLOW_TAG_FEATURE_SET_HASH,
    MLFLOW_TAG_TRAINING_CONFIG_HASH,
    MLFLOW_TAG_TRAINING_IDENTITY_FEATURE_SCHEMA_HASH,
    MLFLOW_TAG_TRAINING_IDENTITY_MODEL_VERSION,
    MLFLOW_TAG_TRAINING_IDENTITY_RUN_ID,
    MLFLOW_TAG_TRAINING_RUN_SPEC_VERSION,
    MLFLOW_TAG_TUNING_RESUME_REASON,
    MODEL_MANAGER_BASELINE_DIAGNOSTIC_KEYS,
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
    assert MLFLOW_TAG_TRAINING_CONFIG_HASH == "training_config_hash"
    assert MLFLOW_TAG_FEATURE_SET_HASH == "feature_set_hash"
    assert MLFLOW_TAG_FEATURE_SCHEMA_HASH == "feature_schema_hash"
    assert MLFLOW_TAG_TRAINING_RUN_SPEC_VERSION == "training_run_spec_version"
    assert MLFLOW_TAG_TRAINING_IDENTITY_RUN_ID == "training_identity.mlflow_run_id"
    assert (
        MLFLOW_TAG_TRAINING_IDENTITY_FEATURE_SCHEMA_HASH
        == "training_identity.feature_schema_hash"
    )
    assert (
        MLFLOW_TAG_TRAINING_IDENTITY_MODEL_VERSION == "training_identity.model_version"
    )
    assert MLFLOW_TAG_BEST_TRIAL_NUMBER == "best_trial_number"
    assert MLFLOW_TAG_BEST_PARAMS_JSON == "best_params_json"


def test_model_manager_diagnostics_metadata_keys_stable():
    assert DIAGNOSTIC_KEY_STATE == "state"
    assert DIAGNOSTIC_KEY_ACTIVE_MODEL_VERSION == "active_model_version"
    assert DIAGNOSTIC_KEY_LAST_RELOAD_STATUS == "last_reload_status"
    assert DIAGNOSTIC_KEY_SCHEMA_MISMATCH_DETECTED == "schema_mismatch_detected"
    assert DIAGNOSTIC_KEY_ML_TRAINING_RUN_ID == "ml.training.run_id"
    assert DIAGNOSTIC_KEY_ML_MODEL_VERSION == "ml.model.version"
    assert DIAGNOSTIC_KEY_ML_FEATURE_SCHEMA_HASH == "ml.feature.schema_hash"
    assert MODEL_MANAGER_BASELINE_DIAGNOSTIC_KEYS == {
        "state",
        "active_model_version",
        "last_reload_status",
        "schema_mismatch_detected",
    }
