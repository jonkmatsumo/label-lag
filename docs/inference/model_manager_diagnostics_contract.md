# ML Operability Contract (Diagnostics + Drift)

This is the canonical contract reference for:

- `forecast.model_manager.ModelManager.get_diagnostics()`
- `diagnostics.ml_health`
- strict-config visibility (`diagnostics.config` / `diagnostics.ml_health.config`)
- `training.detect_drift.detect_drift()` drift result/error payloads

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
| `status` | `string` | Canonical operator status (`success`, `failure`, `unknown`, `not_run`). |
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
| `config` | `object` | Effective strict-mode runtime config (`strict_feature_schema`, `strict_tuning_resume_validation`, `strict_split_strategy_validation`). Also projected to forecast `GetHealth` component flags. |
| `warnings` | `list[string]` | Compact bounded warning-code summary. |
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
| `warnings` / `ml_health.warnings` | Always present as a bounded list (possibly empty), never null. |
| `ml.training.run_id` / `ml.model.version` / `ml.feature.schema_hash` | Null when training identity artifact is unavailable. |
| `ml_health.model.last_reload_ts` / `ml_health.benchmark.last_status` / `ml_health.benchmark.last_run_ts` / `ml_health.drift.last_error_code` / legacy aliases (`ml_health.last_reload_ts`, `ml_health.benchmark_status`, `ml_health.drift_last_computed_ts`, `ml_health.drift_last_error_code`) | Null when source value has not been observed yet. |
| `ml_health.drift_reference_available` | Null when no drift run has been cached yet. |

## Allowed Enum / Code Values

`state`:
- `idle`
- `loading`
- `ready`
- `failed`

`status` (canonical operator summary on diagnostics + `ml_health`):
- `success`
- `failure`
- `unknown`
- `not_run`

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
- `unknown`

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

`warnings` / `ml_health.warnings` (bounded list values):
- `schema_mismatch_detected`
- `reload_failed_using_last_known_good`
- `feature_coverage_below_threshold`
- `drift_reference_unavailable`

`ml_health.drift.reference_resolution_mode` and legacy alias `ml_health.drift_resolution_mode`:
- `alias`
- `stage`
- `latest`
- `none`

## `ml_health` Shape

`ml_health` contains stable grouped sections and keeps legacy scalar aliases for
backward compatibility.

- `model`
  - `state`
  - `active_model_version`
  - `last_reload_status`
  - `last_reload_ts`
  - `schema_mismatch_detected`
- `benchmark`
  - `enabled`
  - `last_status`
  - `last_run_ts`
- `drift`
  - `reference_resolution_mode`
  - `last_error_code`
- `feature_coverage`
  - `last_ratio`
  - `below_threshold`
- `config`
- `warnings` (bounded list of warning codes)
- summary block
  - `status`
  - `overall_status`
  - `degraded`
  - `has_warnings`
  - `warning_count`

`ml_health.config` keys (always present, default `false`):

- `strict_feature_schema`
- `strict_tuning_resume_validation`
- `strict_split_strategy_validation`

Effective strict config values are normalized from env/runtime values using
boolean semantics (`1/true/yes/on` => `true`, `0/false/no/off` => `false`).

Forecast `GetHealth` strict flag projection:

- Reads `diagnostics.config` when present.
- Falls back to `diagnostics.ml_health.config` when top-level `config` is missing.
- Defaults each strict flag to `false` when neither source is available.

Legacy aliases retained in `ml_health`:

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

Additional canonical top-level summary fields in `ml_health`:

- `status`
- `overall_status` (authoritative alias of `status`)
- `degraded` (derived from canonical `degraded_reasons` + `overall_status`)
- `has_warnings` (derived from canonical warnings list)
- `warning_count` (derived from canonical warnings list length)
- `warnings`

`ml_health.feature_coverage.last_ratio` semantics:

- Bounded float in `[0.0, 1.0]` when observed.
- `null` before first coverage observation.
- Mirrors root diagnostic `feature_coverage_last_ratio`.

## Drift Contract (Canonical)

`training.detect_drift.detect_drift()` returns the same top-level core shape for
success and failure modes.

Core top-level keys (always present):

- `timestamp`
- `hours_analyzed`
- `threshold`
- `reference_size`
- `live_size`
- `features`
- `drift_detected`
- `drifted_features`
- `drift_error`
- `error_code`
- `error_message`
- `error`
- `resolution_mode`
- `alerts`
- `reference_resolution`
- `reference_model_version`
- `reference_resolution_mode_requested`
- `reference_resolution_mode`
- `reference_model_version_chosen`
- `reference_alias_requested`
- `reference_resolution_warning`

Field semantics:

- `error_code`: bounded reason code or `null` (max 64 chars).
- `error_message`: short operator-facing message or `null`, capped at 200 chars.
- `error`: legacy alias that mirrors `error_message` (`null` on no-error paths).
- `resolution_mode`: canonical reference resolution mode.
- `reference_resolution_mode_requested`: requested reference-resolution mode.
- `reference_resolution_mode`: resolved reference-resolution mode (mirrors `resolution_mode`).
- `reference_model_version`: selected reference model version when available.
- `reference_model_version_chosen`: additive alias for selected reference model version.
- `reference_alias_requested`: requested alias (when configured).
- `reference_resolution_warning`: bounded optional warning code explaining fallback/ambiguity.
- `drift_error`: legacy drift code mirror; either canonical `error_code` value or `null`.

`reference_resolution` shape (always present):

- `requested_alias`: string or `null`
- `resolution_strategy`: canonical mode (`alias`/`stage`/`latest`/`none`)
- `resolution_mode`: canonical mode (`alias`/`stage`/`latest`/`none`)
- `alias_candidate_count`: integer (`0` when none)
- `alias_ambiguous`: boolean
- `selected_model_version`: string or `null`
- `selected_run_id`: string or `null`

`features.<feature_name>` shape (for included monitored features):

- `psi`: float
- `status`: `OK` / `WARNING` / `CRITICAL`
- `drift_error`: canonical error code or `null`
- `bucketing`: normalized metadata map with bounded keys:
  - `buckettype_requested`
  - `buckettype_used`
  - `buckets_requested`
  - `buckets_used`
  - `bucketing_fallback_reason`
  - `reference_sample_size`
  - `nonempty_buckets`
  - `nonempty_buckets_ratio`
  - `min_expected_count`
  - `bucket_mass_ok`
  - `bucket_mass_guardrail_applied`
  - `drift_error`
  - `breakpoints`

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

`resolution_mode` semantics:

- `alias`: resolved by configured alias (`DRIFT_REFERENCE_MODEL_ALIAS`).
- `stage`: resolved by legacy Production stage fallback.
- `latest`: resolved by latest version fallback.
- `none`: no reference could be resolved.

Legacy normalization behavior:

- Legacy resolution values (`production_stage`, `latest_version`) are normalized
  to canonical `resolution_mode` (`stage`, `latest`).
- Canonical `resolution_mode` values (`alias`, `stage`, `latest`, `none`) are
  passed through unchanged.
- If top-level `resolution_mode` is missing/invalid, the value is derived from
  `reference_resolution.resolution_mode` or
  `reference_resolution.resolution_strategy`.
- When only legacy error fields are present, canonical `error_code` /
  `error_message` are backfilled.
- Legacy error-code aliases (for example `insufficient_reference_data`,
  `no_reference_model`, `no_live_window`) are normalized into canonical
  `error_code` values.
- `error_message` is capped at 200 characters and is null when `error_code` is null.

### Reference Resolution Metadata (Always Present)

`reference_resolution` is always a map with these keys:

- `requested_alias`
- `resolution_strategy`
- `resolution_mode`
- `alias_candidate_count`
- `alias_ambiguous`
- `selected_model_version`
- `selected_run_id`

Notes:

- `resolution_mode` is canonical (`alias|stage|latest|none`).
- `resolution_strategy` is canonical and mirrors `resolution_mode`.
- `alias_candidate_count` is always an integer (default `0`).
- `alias_ambiguous` is always a boolean (default `false`).

Top-level reference metadata clarifies requested vs resolved outcomes:

- `reference_resolution_mode_requested`: canonical requested mode.
- `reference_resolution_mode`: canonical resolved mode (same as `resolution_mode`).
- `reference_model_version_chosen`: chosen model version (`null` when unresolved).
- `reference_alias_requested`: requested alias (`null` when not configured).
- `reference_resolution_warning`: optional bounded warning code:
  - `alias_not_found_fallback`
  - `alias_ambiguous_selected_highest`
  - `stage_fallback_used`
  - `latest_fallback_used`
  - `no_reference_versions_available`

### Per-Feature Bucketing Metadata (When Feature Is Evaluated)

Each `features.<name>.bucketing` map includes:

- `buckettype_requested`
- `buckettype_used`
- `buckets_requested`
- `buckets_used`
- `bucketing_fallback_reason`
- `breakpoints` (bounded list)
- `reference_sample_size`
- `nonempty_buckets`
- `nonempty_buckets_ratio`
- `min_expected_count`
- `bucket_mass_ok`
- `bucket_mass_guardrail_applied`
- `drift_error`

Bucketing normalization semantics:

- `buckettype_requested` / `buckettype_used`: `bins` / `quantiles` / `null`.
- `bucketing_fallback_reason`: canonical fallback reason (`tied_quantiles`, `insufficient_bucket_mass`) or `null`.
- `drift_error`: canonical `error_code` value or `null`.

## Example Payload

```json
{
  "state": "ready",
  "status": "success",
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
  "feature_coverage_last_ratio": 1.0,
  "feature_coverage_warning_last_seen_ts": null,
  "ml.training.run_id": "bf0e9d26c4f94ce5b8ef93f7bdf98b2a",
  "ml.model.version": "17",
  "ml.feature.schema_hash": "f38aa5...",
  "config": {
    "strict_feature_schema": false,
    "strict_tuning_resume_validation": false,
    "strict_split_strategy_validation": false
  },
  "warnings": [],
  "ml_health": {
    "model": {
      "state": "ready",
      "active_model_version": "v17",
      "last_reload_status": "success",
      "last_reload_ts": 1769246400.125,
      "schema_mismatch_detected": false
    },
    "benchmark": {
      "enabled": true,
      "last_status": "success",
      "last_run_ts": 1769246400.412
    },
    "drift": {
      "reference_resolution_mode": "alias",
      "last_error_code": null
    },
    "feature_coverage": {
      "last_ratio": 1.0,
      "below_threshold": false
    },
    "config": {
      "strict_feature_schema": false,
      "strict_tuning_resume_validation": false,
      "strict_split_strategy_validation": false
    },
    "warnings": [],
    "status": "success",
    "overall_status": "success",
    "degraded": false,
    "has_warnings": false,
    "warning_count": 0,
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
    "drift_last_error_code": null
  }
}
```

### Drift Result Example (Success Path)

```json
{
  "timestamp": "2026-01-24T12:00:00+00:00",
  "hours_analyzed": 24,
  "threshold": 0.25,
  "reference_size": 1000,
  "live_size": 1000,
  "features": {
    "velocity_24h": {
      "psi": 0.0,
      "status": "OK",
      "bucketing": {
        "buckettype_requested": "quantiles",
        "buckettype_used": "quantiles",
        "buckets_requested": 10,
        "buckets_used": 10,
        "bucketing_fallback_reason": null,
        "breakpoints": [0.0, 100.0],
        "reference_sample_size": 1000,
        "nonempty_buckets": 10,
        "nonempty_buckets_ratio": 1.0,
        "min_expected_count": 100.0,
        "bucket_mass_ok": true,
        "bucket_mass_guardrail_applied": true,
        "drift_error": null
      }
    }
  },
  "drift_detected": false,
  "drifted_features": [],
  "drift_error": null,
  "error_code": null,
  "error_message": null,
  "resolution_mode": "alias",
  "alerts": [],
  "reference_resolution": {
    "requested_alias": "champion",
    "resolution_strategy": "alias",
    "resolution_mode": "alias",
    "alias_candidate_count": 1,
    "alias_ambiguous": false,
    "selected_model_version": "9",
    "selected_run_id": "run-v9"
  },
  "reference_model_version": "9",
  "reference_resolution_mode_requested": "alias",
  "reference_resolution_mode": "alias",
  "reference_model_version_chosen": "9",
  "reference_alias_requested": "champion",
  "reference_resolution_warning": null
}
```
