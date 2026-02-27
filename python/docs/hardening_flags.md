# ML Hardening Flags and Defaults

Single-source operational reference for Tranches 4-6 hardening controls.

## Runtime Flags

| Flag | Default | Scope | Where read | Effect | Related metrics / diagnostics | When to enable |
|---|---|---|---|---|---|---|
| `ENFORCE_MODEL_FEATURES` | `false` | Inference reload | `python/src/forecast/model_manager.py` (`_load_from_mlflow`) | If `true`, missing registry features fail reload. If `false`, warn-only and continue. | `forecast_model_reload_failure_total{reason=...}`, `last_reload_status`, `last_error` | Enable in prod when model/feature registry sync is reliable and reload correctness is preferred over availability. |
| `INFERENCE_BENCHMARK_ENABLED` | `true` | Inference reload | `python/src/forecast/model_manager.py` (`_load_benchmark_enabled`) | Enables/disables load-time benchmark entirely. | `benchmark_last_status`, `benchmark_last_run_ts`, benchmark Prometheus metrics | Keep enabled by default; disable only when reload-time overhead must be minimized. |
| `INFERENCE_BENCHMARK_SAMPLE_RATE` | `1.0` | Inference reload | `python/src/forecast/model_manager.py` (`_load_benchmark_sample_rate`) | Probabilistic benchmark gating per version load (`0.0-1.0`). | `benchmark_last_status` (`skipped_sampled_out`/`success`) | Reduce below `1.0` for frequent reload environments to cap overhead. |
| `INFERENCE_MODEL_RELOAD_SPAN_ENABLED` | `false` | Inference observability | `python/src/forecast/model_manager.py` (`_reload_span_context`) | Enables MLflow span creation for reload lifecycle. | Reload span attrs (`model.reload.*`, `ml.training.run_id`, etc.) | Enable during incident debugging or reload latency investigations. |
| `STRICT_TUNING_RESUME_VALIDATION` | `false` | Training/tuning | `python/src/model/tuning.py` (`run_tuning_study`) | Fails resume on invariant mismatch when enabled; warn-only when disabled. | Resume mismatch warnings and strict failure error path | Enable when reproducibility/comparability is required across resumed studies. |
| `STRICT_SPLIT_STRATEGY_VALIDATION` | `false` | Training request validation | `python/src/training/service.py` (`_apply_split_strategy_validation_policy`) | Rejects unsupported split strategies when enabled; warn-only compatibility mode when disabled. | `ValidateTrainRequest` warnings, gRPC invalid-argument failures | Enable after clients are migrated off unsupported split strategies. |
| `DRIFT_PSI_WARN_THRESHOLD` | `0.1` | Drift monitoring | `python/src/training/detect_drift.py` (`_load_drift_thresholds`) | Warning threshold for PSI alerts. | Drift status payload (`alerts`, per-feature `status`) | Raise/lower based on desired sensitivity to moderate drift. |
| `DRIFT_PSI_CRIT_THRESHOLD` | `0.25` | Drift monitoring | `python/src/training/detect_drift.py` (`_load_drift_thresholds`) | Critical threshold for PSI alerts. | Drift status payload (`drift_detected`, critical alerts) | Tune with incident history; keep stricter for high-risk models. |
| `DRIFT_PSI_MIN_EXPECTED_PER_BUCKET` | `5.0` | Drift PSI robustness | `python/src/training/detect_drift.py` (`calculate_psi`) | Guardrail for minimum expected mass per bucket before PSI is trusted. | `drift_error=insufficient_bucket_mass`, bucketing metadata | Increase for noisy sparse data; lower only with explicit review. |
| `DRIFT_PSI_MIN_NONEMPTY_BUCKETS_RATIO` | `0.6` | Drift PSI robustness | `python/src/training/detect_drift.py` (`calculate_psi`) | Minimum non-empty bucket ratio guardrail before PSI is trusted. | `drift_error=insufficient_bucket_mass`, bucketing metadata | Raise when wanting stricter PSI quality checks. |
| `DRIFT_REFERENCE_MODEL_ALIAS` | `""` | Drift reference selection | `python/src/training/detect_drift.py` (`_select_reference_model_version`) | Selects reference model by alias before stage/latest fallback. | `reference_resolution` metadata in drift response | Set in production for deterministic reference resolution. |
| `TUNING_MLFLOW_NESTED_RUNS` | `false` | Tuning observability | `python/src/model/tuning.py` (`JobProgressCallback`) | Enables per-trial nested MLflow runs for completed trials. | Nested run logs/metrics in MLflow | Enable for deep trial-level debugging; keep off to reduce tracking volume. |

## Calibration Guard Thresholds (Request Fields)

These are not env vars. They are request-level controls passed into `train_model`.

| Field | Default | Scope | Where read | Effect | Related metrics / diagnostics | When to enable / tune |
|---|---|---|---|---|---|---|
| `min_cal_samples` | `100` in `TrainRequest` defaults | Training calibration | `python/src/training/service.py` -> `python/src/model/train.py` | Minimum calibration slice size required before calibrator fit. | MLflow params: `calibration_enabled`, `calibration_skip_reason`, `calibration_samples` | Raise for stricter calibration quality; lower only for very small datasets. |
| `min_cal_pos` | `10` in `TrainRequest` defaults | Training calibration | `python/src/training/service.py` -> `python/src/model/train.py` | Minimum positive labels in calibration slice. | Same as above; skip reason code if not met | Raise when calibration on rare positives is unstable. |
| `min_cal_neg` | `10` in `TrainRequest` defaults | Training calibration | `python/src/training/service.py` -> `python/src/model/train.py` | Minimum negative labels in calibration slice. | Same as above; skip reason code if not met | Raise when negative-class calibration is noisy. |

Notes:
- `train_model` function signature contains internal fallback defaults (`200/5/5`), but service/RPC default behavior is governed by `TrainRequest` defaults (`100/10/10`).
- Calibration skip behavior is intentionally warn-only and emits explicit reason codes for observability.
