# ModelManager Diagnostics Snapshot Contract

This contract describes the payload returned by
`forecast.model_manager.ModelManager.get_diagnostics()`.

## Guardrail Baseline Subset (Always Present)

These keys are guaranteed to exist for operability guardrails, including idle state:

- `state`
- `active_model_version`
- `last_reload_status`
- `schema_mismatch_detected`

## Required Baseline Fields (Always Present)

| Field | Type | Notes |
| --- | --- | --- |
| `state` | `string` | ModelManager lifecycle state. |
| `model_version` | `string` | Active model version or `"unknown"`. |
| `model_source` | `string` | Active source (`mlflow`, `fallback`, `none`). |
| `last_error` | `string \| null` | Last error text when present. |
| `schema_mismatch_detected` | `bool` | Feature schema mismatch indicator. |
| `calibrator_loaded` | `bool` | Whether `calibrator.pkl` was loaded. |
| `has_bundle` | `bool` | Whether a runtime model bundle exists. |
| `last_reload_ts` | `float \| null` | Last successful bundle load timestamp. |
| `last_reload_status` | `string` | Derived reload status. |
| `last_reload_reason` | `string \| null` | Reload failure reason code when `last_reload_status=failed`. |
| `benchmark_last_run_ts` | `float \| null` | Last benchmark attempt epoch timestamp. |
| `benchmark_last_status` | `string \| null` | Benchmark status code when benchmark path has executed. |
| `degraded_reasons` | `list[string]` | Bounded degraded-reason vocabulary. |
| `active_model_version` | `string` | Backward-compatible alias for currently active model version. |
| `feature_coverage_warning_active` | `bool` | Coverage warning latch. |
| `feature_coverage_warning_last_seen_ts` | `float \| null` | Last warning observation timestamp. |
| `ml.training.run_id` | `string \| null` | Training run correlation id. |
| `ml.model.version` | `string \| null` | Training-side model version value. |
| `ml.feature.schema_hash` | `string \| null` | Training-side feature schema hash. |
| `config` | `object` | Effective strict-mode runtime config (`strict_feature_schema`, `strict_tuning_resume_validation`, `strict_split_strategy_validation`). |
| `ml_health` | `object` | Compact bounded operator summary derived from diagnostics + last drift cache snapshot when available. |

## Nullable Value Conditions (Keys Still Present)

| Field | Nullability Condition |
| --- | --- |
| `last_reload_ts` | Null before first successful bundle load. |
| `last_error` | Null when no reload/serving error has been recorded. |
| `last_reload_reason` | Null unless `last_reload_status=failed`. |
| `benchmark_last_run_ts` | Null until benchmark path has run or skipped. |
| `benchmark_last_status` | Null until benchmark path has run or skipped. |
| `feature_coverage_warning_last_seen_ts` | Null until coverage warning has been observed. |
| `ml.training.run_id` / `ml.model.version` / `ml.feature.schema_hash` | Null when training identity artifact is unavailable. |
| `ml_health.last_reload_ts` / `ml_health.benchmark_status` / `ml_health.drift_last_computed_ts` / `ml_health.drift_last_error_code` | Null when source value has not been observed yet. |
| `ml_health.drift_reference_available` | Null when no drift run has been cached yet. |

## Allowed Enum / Code Values

`state`:
- `idle`
- `loading`
- `ready`
- `failed`

`last_reload_status`:
- `idle`
- `success`
- `failed`

`last_reload_reason` (when present):
- `artifact_missing`
- `mlflow_fetch`
- `unknown`

`benchmark_last_status` (when present):
- `skipped_disabled`
- `skipped_sampled_out`
- `success`
- `failed`

`degraded_reasons` (bounded list values):
- `reload_failed`
- `schema_mismatch`
- `feature_coverage_warning`

`model_source`:
- `mlflow`
- `fallback`
- `none`

`ml_health.feature_coverage_status`:
- `ok`
- `warning`

`ml_health.drift_resolution_mode`:
- `alias`
- `stage`
- `latest`
- `none`

## `ml_health` Shape

`ml_health` is bounded and scalar-only except for nested `config` booleans:

- `state`
- `active_model_version`
- `last_reload_status`
- `last_reload_ts`
- `schema_mismatch_detected`
- `benchmark_status`
- `feature_coverage_status`
- `feature_coverage_last_seen_ts`
- `drift_reference_available`
- `drift_resolution_mode`
- `drift_last_computed_ts`
- `drift_last_error_code`
- `config`

`ml_health.config` keys:

- `strict_feature_schema`
- `strict_tuning_resume_validation`
- `strict_split_strategy_validation`

## Drift Error Fields (Canonical)

`training.detect_drift.detect_drift()` includes additive canonical fields:

- `error_code`: bounded reason code or `null`.
- `error_message`: short operator-facing message or `null`.
- `resolution_mode`: canonical reference resolution mode.

Allowed `error_code` values:

- `no_reference_data`
- `insufficient_reference_samples`
- `no_live_data`
- `insufficient_bucket_mass`

Allowed `resolution_mode` values:

- `alias`
- `stage`
- `latest`
- `none`

## Example Payload

```json
{
  "state": "ready",
  "model_version": "v17",
  "model_source": "mlflow",
  "last_error": null,
  "schema_mismatch_detected": false,
  "calibrator_loaded": true,
  "has_bundle": true,
  "last_reload_ts": 1769246400.125,
  "last_reload_status": "success",
  "last_reload_reason": null,
  "benchmark_last_run_ts": 1769246400.412,
  "benchmark_last_status": "success",
  "degraded_reasons": [],
  "active_model_version": "v17",
  "feature_coverage_warning_active": false,
  "feature_coverage_warning_last_seen_ts": null,
  "ml.training.run_id": "bf0e9d26c4f94ce5b8ef93f7bdf98b2a",
  "ml.model.version": "17",
  "ml.feature.schema_hash": "f38aa5...",
  "config": {
    "strict_feature_schema": false,
    "strict_tuning_resume_validation": false,
    "strict_split_strategy_validation": false
  },
  "ml_health": {
    "state": "ready",
    "active_model_version": "v17",
    "last_reload_status": "success",
    "last_reload_ts": 1769246400.125,
    "schema_mismatch_detected": false,
    "benchmark_status": "success",
    "feature_coverage_status": "ok",
    "feature_coverage_last_seen_ts": null,
    "drift_reference_available": true,
    "drift_resolution_mode": "alias",
    "drift_last_computed_ts": 1769246700.05,
    "drift_last_error_code": null,
    "config": {
      "strict_feature_schema": false,
      "strict_tuning_resume_validation": false,
      "strict_split_strategy_validation": false
    }
  }
}
```
