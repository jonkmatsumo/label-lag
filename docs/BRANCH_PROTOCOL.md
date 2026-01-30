# Branch Protocol: feature/stabilize-persistence-features-inference-v2

This document defines the rules for working on this stabilization branch.

## Baseline

- **Branch created from**: `main` at `8102f468d9ec1e9e8271ac1b81be78e366c77d71`
- **Date**: 2026-01-30
- **Baseline test status**: All fast checks pass (725 passed, 5 skipped)

## Absolute Rules

### 1. New Branch Only (No Reuse)

- Work only on `feature/stabilize-persistence-features-inference-v2`
- **DO NOT** merge, rebase, cherry-pick, or copy/paste from `feature/fix-persistence-auth-features-inference`
- If similar functionality is needed, re-implement from first principles

### 2. Per-Commit Testing

After **every commit**, run the fast verification:

```bash
./scripts/verify.sh
```

After **every phase**, run the full verification:

```bash
./scripts/verify.sh --full
```

### 3. Immediate Revert on Regression

If any commit increases test failures meaningfully:

1. Stop immediately
2. Revert the commit: `git revert HEAD`
3. Re-implement behind a default-off feature flag or isolated module

### 4. Feature Flags Default to Legacy Behavior

All new features must:

- Be disabled by default
- Preserve existing behavior when disabled
- Use explicit opt-in via environment variables

### 5. No Auth / No Async Contract Changes

- DO NOT add JWT auth or RBAC enforcement
- DO NOT convert existing endpoints to async jobs
- If scaffolding is added, it must be:
  - Fully disabled by default
  - Non-invasive
  - Protected by tests ensuring it stays disabled

## Feature Flags Convention

| Flag | Default | Options | Purpose |
|------|---------|---------|---------|
| `RULE_STORE_BACKEND` | `inmemory` | inmemory, postgres | Ruleset persistence backend |
| `INFERENCE_EVENT_SINK` | `jsonl` | jsonl, stdout, postgres, none | Inference logging destination |
| `FEATURE_MATERIALIZATION_MODE` | `legacy` | legacy, cursor | Feature materialization strategy |
| `INFERENCE_BACKEND` | `python` | python, go, go_with_fallback | Inference routing backend |

## Verification Commands

```bash
# Fast check (every commit) - ~15 seconds
./scripts/verify.sh

# Full check (end of phase) - includes BFF, web, integration tests
./scripts/verify.sh --full
```

## Phase Checklist

- [x] Phase 0: Baseline lock + branch protocol
- [x] Phase 1: Ruleset persistence v2
- [x] Phase 2: Durable inference logging v2
- [ ] Phase 3: Feature materialization cursor mode v2
- [ ] Phase 4: Go inference routing v2
- [ ] Phase 5: Regression guardrails
