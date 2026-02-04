# Legacy Decisioning Deletion Readiness Checklist

This document tracks the readiness to delete the legacy Python rules engine and decisioning code.

## Checklist

- [x] Sandbox no longer calls Python rule evaluation (`api.rules.evaluate_rules`)
- [x] Backtest no longer calls Python rule evaluation (`api.rules.evaluate_rules`)
- [x] Go gateway is the only decisioning engine used by:
    - [x] BFF/Web canonical path (verified via existing routing)
    - [x] Sandbox evaluation (verified via `test_sandbox_no_python_decisioning`)
    - [x] Backtests (verified via `test_backtest_no_python_decisioning`)
- [x] Remaining Python decisioning code is only needed for:
    - [x] BFF rollback via `/evaluate/signal` (temporarily retained)
- [x] Tests added cover:
    - [x] Sandbox gateway calls and mapping (`tests/test_rule_inspector.py`)
    - [x] Backtest gateway calls and metrics (`tests/test_backtest.py`)
    - [x] Guardrails against re-importing Python rules engine in sandbox/backtest (`tests/test_migration_guardrails.py`)

## Verified Parity

- [x] **Sandbox Parity**: Go gateway returns full `Explanation` (Action, Score) enabling 100% parity with legacy sandbox outputs.
- [x] **Backtest Parity**: Backtest metrics computation remains unchanged, only the engine per-record has changed.
- [x] **Contract Parity**: `rules_version` vs `ruleset_version` consolidated.

## Ready for Deletion
The following symbols can be safely deleted once BFF rollback is no longer required:
- `api.rules.evaluate_rules()`
- `SignalEvaluator.evaluate()`
- `SignalEvaluator._apply_rules()`
- `POST /evaluate/signal` route in `api/main.py`
