# Resilience & Operability Guide

This guide describes the key resilience signals exported by Label Lag services and how to use them for troubleshooting and alerting.

## 1. Metrics Reference

### Orchestrator (Go)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `orchestrator_breaker_state` | Gauge | `name` | Current state of a circuit breaker: `0` (Closed), `1` (Open), `2` (Half-Open). |
| `orchestrator_breaker_transitions_total` | Counter | `name`, `from`, `to` | Total state transitions for a circuit breaker. |
| `orchestrator_log_events_dropped_total` | Counter | `queue`, `reason` | Events dropped by async loggers. Queues: `analytics`. Reasons: `send_error`, `shutdown`. |
| `orchestrator_rate_limited_total` | Counter | `tenant_present` | Requests rejected by the rate limiter. |
| `orchestrator_rate_limit_tenants_total` | Gauge | - | Number of active per-tenant rate limiters in memory. |

### Forecast Service (Python)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `forecast_model_fallback_total` | Counter | `reason` | Times the service fell back from MLflow to a local pickle model. |
| `forecast_heuristic_fallback_total` | Counter | `reason` | Times the service used heuristic scoring due to model absence or missing features. |
| `forecast_zero_fallback_total` | Counter | `reason` | Times the service returned `0.0` probability due to critical fallback mode. |

---

## 2. Helpful PromQL Snippets

### Detection of Service Degradation
```promql
# Alert if any circuit breaker stays open for more than 5 minutes
max_over_time(orchestrator_breaker_state[5m]) == 1
```

### Queue Saturation Warning
```promql
# Rate of log drops per second
sum(rate(orchestrator_log_events_dropped_total[5m])) by (queue, reason) > 0
```

### Rate Limiting Pressure
```promql
# Compare rate-limited requests vs successful ones
sum(rate(orchestrator_rate_limited_total[1m])) / sum(rate(http_request_duration_seconds_count[1m]))
```

### Forecast Model Health
```promql
# Alert if model fallback rate is high
rate(forecast_model_fallback_total[5m]) > 0.1
```

---

## 3. Troubleshooting Playbook

### Scenario A: `orchestrator_breaker_state` is `1` (Open)
1. **Identify the target**: Check the `name` label (e.g., `analytics`).
2. **Check downstream health**: Verify if the target service (e.g., `analytics-crud`) is reachable and not returning errors.
3. **Check logs**: Look for `circuit breaker: open` in Orchestrator logs for the failure reason.
4. **Resolution**: If downstream is healthy, the breaker will transition to `Half-Open` (State `2`) after the reset timeout and eventually close.

### Scenario B: High `forecast_heuristic_fallback_total`
1. **Check Feature Coverage**: If `reason="no_history"`, the user might not have transactions in the database.
2. **Check Model Availability**: If `reason="model_not_loaded"`, verify the Python Inference service successfully loaded a model on startup.
3. **MLflow Connectivity**: Ensure the Inference service can reach the MLflow tracking server.

### Scenario C: Large `orchestrator_rate_limit_tenants_total`
1. **Memory Pressure**: High tenant counts may indicate a "DDoS" attempt or a very large number of active users.
2. **Cleanup Check**: The Orchestrator automatically cleans up idle limiters after 10 minutes. If this gauge doesn't decrease, check if cleanup is failing.
