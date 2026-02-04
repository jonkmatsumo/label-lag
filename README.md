# Label Lag

## Overview

Label Lag is an end-to-end fraud detection system that pairs realistic label-delay simulation with hybrid model-and-rules scoring. It generates synthetic transaction data, trains and registers models, serves live inference through an API, and provides a React dashboard for analysis and rule authoring.

## Diagrams

### System Design Diagram

This diagram shows the current runtime path for inference, analytics, and model registry.

```mermaid
flowchart TB
    subgraph UI[User Interfaces]
        UI_REACT[React UI]
    end

    subgraph Edge[Edge Layer]
        BFF[Node BFF]
    end

    subgraph Inference[Inference & ML Compute]
        GO_INF[Go Inference Gateway]
        PY_API[Python API / ML Compute]
    end

    subgraph Analytics[Analytics Data Access]
        GO_CRUD[Go Analytics CRUD]
        DB[(Postgres)]
    end

    subgraph Registry[Model Registry]
        MLFLOW[MLflow Registry]
        MINIO[MinIO Artifacts]
    end

    UI_REACT --> BFF --> GO_INF --> PY_API --> GO_CRUD --> DB
    PY_API --> MLFLOW --> MINIO
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
3) Open the dashboard at `http://localhost:5180` and verify Live Scoring renders.

## Detailed Architecture Breakdown

Label Lag separates infrastructure, application runtime, and lifecycle workflows so that training and deployment are explicit and observable. The system design diagram above shows the runtime path (React → BFF → Go Inference → Python ML → Go Analytics → DB). The pipeline diagram shows how models move from training to production inference. The rule state machine anchors governance, ensuring changes pass review before affecting live scoring.

Core flows:
- **Data generation and feature materialization** feed training and historical analytics while preserving point-in-time correctness.
- **Training and registry** capture metrics and artifacts in MLflow, enabling explicit promotion and deployment.
- **Inference and rule evaluation** combine model predictions with a rule engine that supports shadow testing and auditing.
- **Dashboard-driven workflows** expose model and rule lifecycle actions without bypassing API controls.

## Ports & Services Table

All ports are configurable via `.env`.

| Service | Port | Purpose |
|---------|------|---------|
| Web (React) | 5180 | React UI for scoring, analytics, model training, and rule authoring |
| BFF | 3210 | Backend for Frontend - Node.js proxy layer for React UI |
| API | 8100 | FastAPI fraud scoring and training endpoints |
| API Docs | 8100 | Swagger UI served by the API |
| MLflow | 5005 | Experiment tracking and model registry |
| MinIO API | 9100 | Object storage API for artifacts |
| MinIO Console | 9101 | Object storage console (minioadmin/minioadmin) |
| PostgreSQL | 5542 | Transaction and feature storage |
| Inference Gateway | 8181 | Go-based high-throughput inference gateway |
| Analytics CRUD | 50051 | Go analytics service (gRPC) backing compute-only data access |

The React UI now supports:
- **Synthetic Dataset Management**: Generate data, view distributions, and analyze correlations.
- **Model Registry**: View MLflow models, CV metrics, and tuning trials.
- **Rule Inspector**: Full rule lifecycle management including Shadow Mode and Backtesting.
- **Analytics**: Historical trends and alert monitoring.

## Go Inference Cutover Readiness

The system includes a Go-based `inference-gateway` designed to front the Python `/evaluate/signal` compute path for high-throughput inference.

The gateway exposes `GET /ready` to verify rules loading and Python backend connectivity.

### Gateway Default Routing

The BFF routes inference and core UI reads through the gateway by default. The Python
API remains for ML-only operations (training, rule lifecycle, backtest compare, and
rule attribution).

### Verifying Parity

A parity test suite is available to compare outputs from both engines:

```bash
# Run parity integration tests (requires stack running)
export RUN_PARITY_TESTS=1
export BFF_PYTHON_API_BASE_URL=http://localhost:8100
export BFF_GATEWAY_BASE_URL=http://localhost:8181
cd bff && npm test tests/parity.test.ts
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
└── synthetic_pipeline/  # Core data generation, DB models
bff/                     # Node.js BFF (Backend for Frontend) for React UI
web/                     # React + TypeScript frontend
```

Key folders:
- **`api/`**: Orchestrates scoring, rule lifecycle, validation, audit logging, and deployment actions.
- **`model/`**: Training workflows, evaluation metrics, and registry interactions.
- **`pipeline/`**: Feature materialization and data correctness safeguards.
- **`generator/`** and **`synthetic_pipeline/`**: Synthetic data creation, fraud patterns, and persistence.

## Service-Level Breakdown

### API Service

Responsible for live scoring, training triggers, rule lifecycle actions, and model deployment. It exposes evaluation and lifecycle endpoints (`/evaluate/signal`, `/train`, `/rules/{id}/publish`, `/models/deploy`) and serves Swagger docs at `/docs`. The API is compute-only and relies on the Go Analytics CRUD service for data access.

### Model Training & Registry (MLflow)

Training runs are tracked with metrics and artifacts, then promoted through stages before deployment. The deploy action reloads the production model into the API, keeping approval and activation separate.

### Rule Engine

Rules evaluate transaction features using operators (`>`, `>=`, `<`, `<=`, `==`, `in`, `not_in`) and actions (`override_score`, `clamp_min`, `clamp_max`, `reject`). The lifecycle enforces draft → review → approval → publish transitions, and supports shadow evaluation and sandbox testing for safe iteration.

### Analytics CRUD (Go)

Provides the gRPC data access layer for compute-only services. The Python API, model training loaders, and analytics endpoints rely on this service for reads and writes, keeping direct database I/O out of ML/compute surfaces.

### Synthetic Data Generator

Generates labeled transaction streams with controlled fraud patterns and label delay to support realistic training and backtesting. It can create data via CLI entrypoints.

Fraud patterns used by the generator:

| Pattern | Description | Key Indicators |
|---------|-------------|----------------|
| Liquidity Crunch | Overdraft attempt | balance z-score < -2.5, returned=True |
| Link Burst | Rapid bank linking | 5-15 connections in 24h |
| ATO (Account Takeover) | Compromised account | amount_ratio > 5.0, off-hours, recent identity change |
| Bust-Out | Build trust then fraud | 20-50 legit transactions, then >500% spike |
| Sleeper ATO | Dormant then active | 30+ days dormancy, link burst, high-value withdrawal |

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
WEB_PORT=5180
BFF_PORT=3210
MLFLOW_PORT=5005
MINIO_API_PORT=9100
MINIO_CONSOLE_PORT=9101
```

### BFF Configuration

```
BFF_PYTHON_API_BASE_URL=http://api:8000
BFF_MLFLOW_TRACKING_URI=http://mlflow:5000
BFF_GATEWAY_BASE_URL=http://inference-gateway:8081
BFF_REQUEST_TIMEOUT=30000
BFF_LOG_LEVEL=info
```

### Inference Gateway (Go)

```
INFERENCE_GATEWAY_MAX_BODY_BYTES=1048576
INFERENCE_GATEWAY_READ_TIMEOUT=10s
INFERENCE_GATEWAY_WRITE_TIMEOUT=30s
INFERENCE_GATEWAY_IDLE_TIMEOUT=60s
```

### Analytics CRUD (Go)

```
ANALYTICS_CRUD_PORT=50051
ANALYTICS_CRUD_TARGET=analytics-crud:50051
ANALYTICS_CRUD_TIMEOUT_SECONDS=15
ANALYTICS_CRUD_ALLOW_INSECURE_DEFAULTS=false
```

### MLflow / MinIO

```
MLFLOW_TRACKING_URI=http://localhost:5005
MLFLOW_S3_ENDPOINT_URL=http://localhost:9100
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```
