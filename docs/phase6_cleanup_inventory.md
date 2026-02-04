# Phase 6 Cleanup Inventory and Guardrails

## Scope
This document records evidence for legacy UI removal (Streamlit) and FastAPI
endpoint pruning so cleanup stays safe and reversible.

## Streamlit inventory (where it exists today)
- Source: `src/ui/` (Streamlit app + Dockerfile)
- Python deps: `pyproject.toml` includes `streamlit>=1.32.0`
- Docker compose: `docker-compose.app.yml` uses Streamlit via profile `streamlit`
- Docs:
  - `README.md` sections mention Streamlit UI and ports
  - `docs/parity/streamlit_readiness.md` parity checklist
  - `CLAUDE.md` references `DASHBOARD_PORT`

## FastAPI endpoints still in use (as of Phase 5)
Gateway:
- Monitoring proxy uses FastAPI today:
  - `GET /monitoring/drift` -> FastAPI (proxy in gateway)
  - `GET /metrics/shadow/comparison` -> FastAPI (proxy in gateway)

BFF:
- All Phase 4/5 core read endpoints route to gateway.
- FastAPI remains for:
  - `GET /bff/v1/analytics/attribution`
  - `POST /bff/v1/backtest/compare`
- MLflow is routed directly (BFF uses MLflow base URL).

Web:
- React UI calls BFF only (no direct FastAPI).

## Evidence checks (grep / code pointers)
- Streamlit references:
  - `rg -n "streamlit|Streamlit" -S`
- BFF gateway routing for core reads:
  - `bff/src/routes/analytics.ts` (target gateway)
  - `bff/src/routes/monitoring.ts` (target gateway)
  - `bff/src/routes/backtest.ts` (results target gateway)
  - `bff/src/routes/dataset.ts` (overview/schema/sample target gateway)
- FastAPI-only paths:
  - `bff/src/routes/analytics.ts` attribution still FastAPI
  - `bff/src/routes/backtest.ts` compare still FastAPI

## Stop conditions (do not delete yet)
- If a dependency is unclear (scripts/tests/CI hitting FastAPI or Streamlit),
  keep the endpoint/service and document the dependency here.
- If gateway still proxies a FastAPI endpoint, do not delete that endpoint until
  the proxy is removed or migrated.

## Notes for Phase 7 follow-ups
- Monitoring drift + shadow comparison still live in FastAPI via gateway proxy.
- Backtest compare still FastAPI (compute-only path).
