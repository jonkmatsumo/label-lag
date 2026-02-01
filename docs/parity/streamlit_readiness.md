# Streamlit Decommission Readiness

This document tracks the parity between the legacy Streamlit UI and the React full-stack application.

## P0 Blockers (Parity Gap)

| Feature | Status | Fixed In |
|---------|--------|----------|
| Live Scoring schema alignment | ✅ Fixed | Phase 1 |
| Dataset Outliers (IQR) | ✅ Fixed | Phase 4 |
| Dataset Relationships | ✅ Fixed | Phase 4 |
| TrainRequest Alignment | ✅ Fixed | Phase 2 |
| MLflow Promotions (Staging/Prod) | ✅ Fixed | Phase 3 |

## P1 Enhancements

| Feature | Status | Fixed In |
|---------|--------|----------|
| Custom Ruleset in Sandbox | ✅ Supported | Phase 5 |
| Visual Artifacts (PNG/MD) | ✅ Supported | Phase 6 |

## Readiness Verdict

The React full-stack application now supports all core workflows previously restricted to Streamlit.
- [x] Live Scoring renders canonical risk components and rules.
- [x] Model Lab supports feature selection, training window, and registry promotions.
- [x] Dataset page provides parity diagnostics (Distributions, Missingness, Outliers, Relationships).
- [x] Model Lab renders visual artifacts (Confusion Matrix, Feature Importance).

**Verdict:** Ready for decommission of legacy Streamlit UI in the next phase.
