# Canonical Inference & Governance Contracts

This document defines the shared contracts for inference and rule governance between the Python API and the Go Inference Gateway.

## 1. Inference Response

The Gateway (`POST /evaluate/signal` - **CANONICAL**) must return this structure:

### Forecaster Boundary (Python)
Python API serves as a **pure forecaster**.
*   **CANONICAL**: `POST /predict/signal` returns calibrated model scores only.
*   **OWNER**: Go Inference Gateway owns all rule evaluation and final decisioning.

```json
{
  "request_id": "req_...",
  "score": 85,
  "risk_label": "HIGH",
  "latency_ms": 45.2,
  "risk_components": [
    {
      "key": "velocity",
      "label": "high_transaction_velocity"
    }
  ],
  "model_version": "v1.0.0",
  "matched_rules": [
    {
      "rule_id": "high_velocity",
      "severity": "medium",
      "reason": "Velocity > 5",
      "explanation": "Velocity 10 exceeds threshold 5"
    }
  ],
  "shadow_matched_rules": [],
  "rules_version": "v1_20240101",
  "debug": { ... } // Optional, behind flag
}
```

### Semantics
*   **score**: Integer 1-99.
*   **risk_label**: LOW (<30), MEDIUM (30-79), HIGH (>=80).
*   **latency_ms**: End-to-end processing time in milliseconds.
*   **matched_rules**: Rules that actively modified the score or triggered a reject.
*   **shadow_matched_rules**: Rules evaluated in **Shadow Mode** (simulation/preview) that matched but did not affect the final score.

## 2. Rule Governance

### Publish Rule Request
```json
{
  "actor": "user@example.com",
  "reason": "Fixing false negatives in region X"
}
```

### Deploy Model Request
```json
{
  "model_version": "v2.0.0",
  "run_id": "run_123",
  "actor": "user@example.com",
  "reason": "Weekly retrain"
}
```

### Readiness & Approval
*   **Readiness**: Hard-gate checks (stability, volume, performance) must PASS or WARN. FAIL blocks publication.
*   **Approval Signals**: Heuristic signals (e.g., "High False Positive Risk") displayed to approvers.
