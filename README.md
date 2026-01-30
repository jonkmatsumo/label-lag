# Label Lag

## Overview

Label Lag is an end-to-end fraud detection system that pairs realistic label-delay simulation with hybrid model-and-rules scoring. It generates synthetic transaction data, trains and registers models, serves live inference through an API, and provides a dashboard for analysis and rule authoring.

## Diagrams

### System Design Diagram

This diagram shows the publish/deploy path across UI, API, and storage, emphasizing how approval and deployment are separated for both rules and models.

```mermaid
flowchart TB
    subgraph UI[Streamlit UI]
        UI1[Rule Inspector]
        UI2[Model Lab]
    end
    
    subgraph API[FastAPI]
        API1[POST /rules/id/publish]
        API2[POST /models/deploy]
        API3[Audit Logger]
    end
    
    subgraph Storage
        S1[DraftRuleStore]
        S2[ModelManager]
        S3[MLflow Registry]
    end
    
    UI1 --> API1
    UI2 --> API2
    API1 --> S1
    API1 --> S2
    API1 --> API3
    API2 --> S3
    API2 --> S2
    API2 --> API3
```

### ML / Data Pipeline Diagram

This diagram summarizes the end-to-end training and deployment path, from model training and registration to live inference.

```mermaid
flowchart LR
    subgraph Training
        A[Train Model] --> B[Register in MLflow]
    end
    subgraph Promotion
        B --> C[Promote to Staging]
        C --> D[Approve for Production]
    end
    subgraph Deployment
        D --> E[Deploy to Production]
        E --> F[Reload API Model]
    end
    subgraph Runtime
        F --> G[Live Inference]
    end
```

### State Machine Diagram

This diagram captures the rule lifecycle, highlighting approval versus deployment and the transitions that keep rule changes auditable.

```mermaid
stateDiagram-v2
    [*] --> draft: create
    draft --> pending_review: submit
    pending_review --> approved: approve
    pending_review --> draft: reject
    approved --> active: publish
    approved --> draft: revoke
    active --> shadow: shadow
    active --> disabled: disable
    shadow --> active: activate
    shadow --> disabled: disable
    disabled --> active: activate
    disabled --> archived: archive
    archived --> [*]
```

## Quick Start

1) Copy `.env.example` to `.env` and adjust ports or credentials as needed.  
2) Start the stack with `docker compose up -d`.  
3) Open the dashboard at `http://localhost:8601` and verify Live Scoring renders.

## Detailed Architecture Breakdown

Label Lag separates infrastructure, application runtime, and lifecycle workflows so that training and deployment are explicit and observable. The publish/deploy diagram above illustrates how UI actions flow through the API into storage and registry services, while the pipeline diagram shows how models move from training to production inference. The rule state machine anchors governance, ensuring changes pass review before affecting live scoring.

Core flows:
- **Data generation and feature materialization** feed training and historical analytics while preserving point-in-time correctness.
- **Training and registry** capture metrics and artifacts in MLflow, enabling explicit promotion and deployment.
- **Inference and rule evaluation** combine model predictions with a rule engine that supports shadow testing and auditing.
- **Dashboard-driven workflows** expose model and rule lifecycle actions without bypassing API controls.

## Ports & Services Table

All ports are configurable via `.env`.

| Service | Port | Purpose |
|---------|------|---------|
| Dashboard (Streamlit) | 8601 | Streamlit UI for scoring, analytics, model training, and rule authoring |
| Web (React) | 5180 | React UI - modern alternative to Streamlit (runs in parallel) |
| BFF | 3210 | Backend for Frontend - Node.js proxy layer for React UI |
| API | 8100 | FastAPI fraud scoring and training endpoints |
| API Docs | 8100 | Swagger UI served by the API |
| MLflow | 5005 | Experiment tracking and model registry |
| MinIO API | 9100 | Object storage API for artifacts |
| MinIO Console | 9101 | Object storage console (minioadmin/minioadmin) |
| PostgreSQL | 5542 | Transaction and feature storage |
| Inference Gateway | 8181 | Go-based high-throughput inference gateway |

### Parallel UI Operation

Both Streamlit (port 8601) and React (port 5180) UIs run simultaneously. The React UI communicates with FastAPI through the BFF proxy layer (port 3210), while Streamlit connects directly to FastAPI. This allows safe migration without disrupting existing workflows.

The React UI now supports:
- **Synthetic Dataset Management**: Generate data, view distributions, and analyze correlations.
- **Model Registry**: View MLflow models, CV metrics, and tuning trials.
- **Rule Inspector**: Full rule lifecycle management including Shadow Mode and Backtesting.
- **Analytics**: Historical trends and alert monitoring.

## Go Inference Cutover Readiness

The system includes a Go-based `inference-gateway` designed to replace the FastAPI `/evaluate/signal` endpoint for high-throughput inference.

### Switching Inference Modes

The BFF supports toggling between FastAPI and Go Gateway via environment variable:

- **FastAPI Mode (Default)**: `BFF_INFERENCE_MODE=fastapi`
- **Go Gateway Mode**: `BFF_INFERENCE_MODE=gateway`

To switch:
1. Update `.env`: `BFF_INFERENCE_MODE=gateway`
2. Restart BFF: `docker compose -f docker-compose.infra.yml -f docker-compose.app.yml restart bff`

### Verifying Parity

A parity test suite is available to compare outputs from both engines:

```bash
# Run parity integration tests (requires stack running)
export RUN_PARITY_TESTS=1
export BFF_FASTAPI_BASE_URL=http://localhost:8100
export BFF_GATEWAY_BASE_URL=http://localhost:8181
cd bff && npm test tests/parity.test.ts
```

### UI Modes

You can control which UIs are started using Docker Compose profiles:

- `UI_MODE=streamlit` (Starts only Streamlit)
- `UI_MODE=react` (Starts React + BFF)
- `UI_MODE=both` (Starts all - default if unset)

Example:
```bash
COMPOSE_PROFILES=react docker compose ... up -d
```

## Repository / File Structure

The repo is organized around data flow and runtime boundaries so services can evolve independently while sharing a common domain model.

```
src/
├── api/                 # FastAPI app, rule engine, evaluation services
├── model/               # XGBoost training, evaluation, tuning
├── monitor/             # Feature distribution monitoring and drift reporting
├── pipeline/            # Point-in-time feature materialization (SQL window functions)
├── generator/           # Stateful fraud profile simulation
├── synthetic_pipeline/  # Core data generation, DB models
└── ui/                  # Streamlit dashboard
bff/                     # Node.js BFF (Backend for Frontend) for React UI
web/                     # React + TypeScript frontend
```

Key folders:
- **`api/`**: Orchestrates scoring, rule lifecycle, validation, audit logging, and deployment actions.
- **`model/`**: Training workflows, evaluation metrics, and registry interactions.
- **`pipeline/`**: Feature materialization and data correctness safeguards.
- **`generator/`** and **`synthetic_pipeline/`**: Synthetic data creation, fraud patterns, and persistence.
- **`ui/`**: Operator-facing workflows for training, evaluation, and rule management.

## Service-Level Breakdown

### API Service

Responsible for live scoring, training triggers, rule lifecycle actions, and model deployment. It exposes evaluation and lifecycle endpoints (`/evaluate/signal`, `/train`, `/rules/{id}/publish`, `/models/deploy`) and serves Swagger docs at `/docs`.

### Dashboard (Streamlit)

The UI consolidates operational workflows: live scoring, historical analytics, dataset exploration, model training and registry promotion, and rule authoring. It is the primary entry point for rule publishing, model deployment, and sandbox evaluation.

### Model Training & Registry (MLflow)

Training runs are tracked with metrics and artifacts, then promoted through stages before deployment. The deploy action reloads the production model into the API, keeping approval and activation separate.

### Rule Engine

Rules evaluate transaction features using operators (`>`, `>=`, `<`, `<=`, `==`, `in`, `not_in`) and actions (`override_score`, `clamp_min`, `clamp_max`, `reject`). The lifecycle enforces draft → review → approval → publish transitions, and supports shadow evaluation and sandbox testing for safe iteration.

### Synthetic Data Generator

Generates labeled transaction streams with controlled fraud patterns and label delay to support realistic training and backtesting. It can create data via the dashboard or CLI entrypoints.

Fraud patterns used by the generator:

| Pattern | Description | Key Indicators |
|---------|-------------|----------------|
| Liquidity Crunch | Overdraft attempt | balance z-score < -2.5, returned=True |
| Link Burst | Rapid bank linking | 5-15 connections in 24h |
| ATO (Account Takeover) | Compromised account | amount_ratio > 5.0, off-hours, recent identity change |
| Bust-Out | Build trust then fraud | 20-50 legit transactions, then >500% spike |
| Sleeper ATO | Dormant then active | 30+ days dormancy, link burst, high-value withdrawal |

## Ops Runbook

### Authentication
The system uses JWT-based authentication. In development, you can obtain an admin token using:
```bash
curl -X POST http://localhost:3210/bff/v1/auth/dev-login -H "Content-Type: application/json" -d '{"role": "admin"}'
```
Include this token in the `Authorization: Bearer <token>` header for all protected API/BFF calls.

### Rule Lifecycle & Persistence
Rules are persisted in Postgres. To verify state or recover:
- **Draft Store**: Rehydrates from the `rules` and `rule_versions` tables on startup.
- **Production Ruleset**: Loaded from the `published_rulesets` table (latest snapshot).
- **Restarting**: `docker compose restart api` will reload the rules from DB, preserving any published or draft state.

### Monitoring Inference
Inference events are durably logged to the `inference_events` table and backed up to `data/inference_events.jsonl` (mounted volume).
Query recent events:
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8100/inference/events?limit=50
```

### Background Jobs
Data generation and training are async. Use the Jobs API to monitor:
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8100/jobs/<job_id>
```

### Feature Materialization
Materialization is now incremental and cursor-based (table `feature_materialization_state`). It runs automatically during data generation and on-demand during inference if stale features are detected.

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed.

### Database

```
POSTGRES_USER=synthetic
POSTGRES_PASSWORD=synthetic_dev_password
POSTGRES_DB=synthetic_data
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5542/${POSTGRES_DB}
```

### Service Ports

```
DB_PORT=5542
API_PORT=8100
INFERENCE_GATEWAY_PORT=8181
DASHBOARD_PORT=8601
WEB_PORT=5180
BFF_PORT=3210
MLFLOW_PORT=5005
MINIO_API_PORT=9100
MINIO_CONSOLE_PORT=9101
```

### BFF Configuration

```
BFF_FASTAPI_BASE_URL=http://api:8000
BFF_MLFLOW_TRACKING_URI=http://mlflow:5000
BFF_INFERENCE_MODE=fastapi  # or 'gateway' to use inference-gateway
BFF_GATEWAY_BASE_URL=http://inference-gateway:8081
BFF_REQUEST_TIMEOUT=30000
BFF_LOG_LEVEL=info
```

### MLflow / MinIO

```
MLFLOW_TRACKING_URI=http://localhost:5005
MLFLOW_S3_ENDPOINT_URL=http://localhost:9100
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```
