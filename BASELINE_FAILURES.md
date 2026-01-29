# Baseline Known Failing Behavior

Date: 2026-01-29

## Summary of Current System Risks

1. **Persistence Gap**: Published rules are in-memory only and lost on API restart.
2. **Scalability**: Feature materialization is O(N^2) because it re-fetches all historical records for every batch.
3. **Security**: No authentication/authorization on API, BFF, or UI.
4. **Data Freshness**: Live inference uses stale features from `feature_snapshots` instead of real-time state.
5. **Observability**: Inference logs are written to volatile container storage.

## Failing Tests (Baseline)

The following tests are currently failing:

- `tests/integration/test_inference_gateway_parity.py`: Gateway vs FastAPI parity mismatches.
- `tests/test_ui_app.py`: Multiple failures related to `st.cache_data.clear()` call counts and timing.

## Validation Targets

- **Rules**: Must survive `docker compose restart api`.
- **Materializer**: Second run must be linear/incremental, not re-processing old rows.
- **Auth**: Admin/Publish endpoints must return 401/403 without JWT.
- **Inference**: Must detect when a user has a new transaction and re-compute features if snapshot is stale.
- **Logs**: `data/inference_events.jsonl` must persist across container recreation.
