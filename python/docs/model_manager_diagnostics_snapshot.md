# ModelManager Diagnostics Snapshot Schema

This document describes the payload returned by
`forecast.model_manager.ModelManager.get_diagnostics()`.

## State Model

`state` transitions follow this lifecycle:

1. `idle` -> initial state (no load attempt yet).
2. `loading` -> active reload in progress.
3. `ready` -> model bundle loaded and serving.
4. `failed` -> reload failed (service may still serve prior bundle if present).

`last_reload_status` is derived from `state`:
- `ready` => `success`
- `failed` => `failed`
- `idle`/`loading` => `idle`

## Field Schema

### Always present keys

| Field | Type | Meaning |
|---|---|---|
| `state` | `string` | Lifecycle state: `idle`, `loading`, `ready`, or `failed`. |
| `model_version` | `string` | Active model version (or `"unknown"`). |
| `model_source` | `string` | Source for active model (`mlflow`, `fallback`, or `none`). |
| `last_error` | `string \| null` | Last error message associated with failed reload/serving state. |
| `schema_mismatch_detected` | `bool` | `true` when feature schema hash mismatch is detected for active model. |
| `calibrator_loaded` | `bool` | Whether a persisted calibrator artifact was loaded. |
| `has_bundle` | `bool` | Whether a bundle-like model payload is currently available. |
| `last_reload_status` | `string` | `success`, `failed`, or `idle` (derived from `state`). |
| `last_reload_reason` | `string \| null` | Standard reload reason code (for failed reload status). |
| `benchmark_last_run_ts` | `float \| null` | Epoch seconds for last benchmark attempt. |
| `benchmark_last_status` | `string \| null` | Benchmark status code (`success`, `failed`, `skipped_disabled`, `skipped_sampled_out`). |
| `degraded_reasons` | `list[string]` | Bounded degraded reason list for current snapshot. |
| `active_model_version` | `string` | Alias of active version for backward-compatible consumers. |
| `feature_coverage_warning_active` | `bool` | Whether low feature coverage warning is currently active. |
| `feature_coverage_warning_last_seen_ts` | `float \| null` | Last warning observation timestamp. |
| `ml.training.run_id` | `string \| null` | Training run correlation id when available. |
| `ml.model.version` | `string \| null` | Training-side model version when available. |
| `ml.feature.schema_hash` | `string \| null` | Training-side schema hash when available. |

### Optional / nullable keys

| Field | Type | Null when |
|---|---|---|
| `last_reload_ts` | `float \| null` | No successful bundle has been loaded yet. |
| `last_error` | `string \| null` | No terminal failure has occurred. |
| `last_reload_reason` | `string \| null` | Last reload status is not `failed`. |
| `benchmark_last_run_ts` | `float \| null` | Benchmark has never run/skipped. |
| `benchmark_last_status` | `string \| null` | Benchmark has never run/skipped. |
| `feature_coverage_warning_last_seen_ts` | `float \| null` | No warning has been observed. |
| `ml.training.run_id` / `ml.model.version` / `ml.feature.schema_hash` | `string \| null` | Training identity artifact unavailable. |

## Degraded Reason Vocabulary

`degraded_reasons` may include:
- `reload_failed`
- `schema_mismatch`
- `feature_coverage_warning`

This field is additive and may contain multiple reasons in one snapshot.

## Example

```json
{
  "state": "ready",
  "model_version": "v42",
  "model_source": "mlflow",
  "last_error": null,
  "schema_mismatch_detected": false,
  "calibrator_loaded": true,
  "has_bundle": true,
  "last_reload_ts": 1769155481.222,
  "last_reload_status": "success",
  "last_reload_reason": null,
  "benchmark_last_run_ts": 1769155481.541,
  "benchmark_last_status": "success",
  "degraded_reasons": [],
  "active_model_version": "v42",
  "feature_coverage_warning_active": false,
  "feature_coverage_warning_last_seen_ts": null,
  "ml.training.run_id": "0f2a0c3f64f7490c9fd0d2f5a6fcb77d",
  "ml.model.version": "42",
  "ml.feature.schema_hash": "9a7f4c..."
}
```
