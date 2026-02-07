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

    subgraph Orchestration[Orchestration & Rules]
        ORCH[Go Orchestrator]
    end

    subgraph Compute[ML Compute]
        PY_INF[Python Inference Service]
        PY_TRAIN[Python Training Service]
    end

    subgraph Analytics[Analytics Data Access]
        GO_CRUD[Go Analytics Service]
        DB[(Postgres)]
    end

    subgraph Registry[Model Registry]
        MLFLOW[MLflow Registry]
        MINIO[MinIO Artifacts]
    end

    UI_REACT --> BFF --> ORCH
    ORCH --> PY_INF
    ORCH --> PY_TRAIN
    ORCH --> GO_CRUD
    PY_INF --> GO_CRUD
    PY_TRAIN --> GO_CRUD
    PY_TRAIN --> MLFLOW --> MINIO
    PY_INF --> MLFLOW
    GO_CRUD --> DB
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
2) Start the stack with `docker compose -f docker-compose.infra.yml -f docker-compose.app.yml --profile react up -d`.
3) Open the dashboard at `http://localhost:5180` and verify Live Scoring renders.

## Detailed Architecture Breakdown

Label Lag separates infrastructure, application runtime, and lifecycle workflows so that training and deployment are explicit and observable. The system design diagram above shows the runtime path (React → BFF → Go Orchestrator → Python Services → Go Analytics → DB). The pipeline diagram shows how models move from training to production inference. The rule state machine anchors governance, with all rule management and analysis now handled by the Go-based control plane.

Core flows:
- **Data generation and feature materialization** feed training and historical analytics while preserving point-in-time correctness.
- **Training and registry** capture metrics and artifacts in MLflow, enabling explicit promotion and deployment.
- **Inference and rule evaluation** combine model predictions with a high-performance Go rule engine (Orchestrator) that supports shadow testing and auditing.
- **Dashboard-driven workflows** expose model and rule lifecycle actions managed through the Go Orchestrator Service.

## Ports & Services Table

All ports are configurable via `.env`.

| Service | Port | Purpose |
|---------|------|---------|
| Service | Port | Purpose |
|---------|------|---------|
| Web (React) | 5180 | React UI for scoring, analytics, model training, and rule authoring |
| BFF | 3210 | Backend for Frontend - Node.js proxy layer for React UI |
| Orchestrator | 8081 | Go Orchestrator (HTTP Gateway + Rule Engine) |
| Training | 50053 | Python Training Service (gRPC) |
| Inference | 50052 | Python Inference Service (gRPC) |
| Analytics | 50051 | Go Analytics Service (gRPC) |
| MLflow | 5005 | Experiment tracking and model registry |
| MinIO API | 9100 | Object storage API for artifacts |
| MinIO Console | 9101 | Object storage console (minioadmin/minioadmin) |
| PostgreSQL | 5542 | Transaction and feature storage |

The React UI now supports:
- **Synthetic Dataset Management**: Generate data, view distributions, and analyze correlations.
- **Model Registry**: View MLflow models, CV metrics, and tuning trials.
- **Rule Inspector**: Full rule lifecycle management including Shadow Mode and Backtesting.
- **Analytics**: Historical trends and alert monitoring.

## Repository / File Structure

The repo is organized around data flow and runtime boundaries so services can evolve independently while sharing a common domain model.

```
go/
├── analytics/           # Go gRPC services
└── orchestrator/        # Go rule engine & gateway
python/
└── src/
    ├── forecast/        # Forecasting logic
    ├── inference/       # gRPC Inference Service
    ├── training/        # gRPC Training Service
    └── model/           # Shared ML logic
typescript/
├── bff/                 # Node.js Backend for Frontend
└── ui/                  # React + TypeScript frontend
```

## Service-Level Breakdown

### Orchestrator Service (Go)

(Formerly Inference Service/Gateway). The central gateway for the Label Lag system. It serves HTTP traffic from the BFF/UI and orchestrates calls to backend services (Training, Inference, Analytics). It also hosts the high-performance **Rule Engine**.

Key features:
- **Rule Engine**: Evaluates transactions against fraud rules using operators (`>`, `>=`, `in`, etc.).
- **Gateway**: Proxies administrative actions (training, deployment) to Python services.
- **Shadow Mode**: Supports shadow evaluation and auditing.

### Inference Service (Python)

A specialized gRPC service for low-latency model inference. It loads registered models from MLflow and serves prediction requests from the Orchestrator. It is compute-only and fetches features from the Analytics Service.

### Training Service (Python)

A gRPC service responsible for:
- Model training (XGBoost)
- Triggering data generation
- Handling model deployment (loading models into memory/registry)
- Forecasting / Drift Monitoring

### Analytics Service (Go)

Provides the gRPC data access layer for all compute services. It manages PostgreSQL interactions, ensuring that Python services remain stateless and compute-focused.

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
BFF_PYTHON_API_BASE_URL=http://training-server:8000
BFF_MLFLOW_TRACKING_URI=http://mlflow:5000
BFF_GATEWAY_BASE_URL=http://inference:8081
BFF_REQUEST_TIMEOUT=30000
BFF_LOG_LEVEL=info
```

### Inference Service (Go)

```
INFERENCE_GATEWAY_MAX_BODY_BYTES=1048576
INFERENCE_GATEWAY_READ_TIMEOUT=10s
INFERENCE_GATEWAY_WRITE_TIMEOUT=30s
INFERENCE_GATEWAY_IDLE_TIMEOUT=60s
INFERENCE_GATEWAY_RULES_PATH=go/orchestrator/config/default_rules.json
INFERENCE_GATEWAY_RULES_WATCH=true
ENABLE_ADMIN_RPCS=false
```

The Inference Service provides high-throughput rule evaluation and supports several advanced features:
- **Rule Conflict Detection**: Automatically identifies overlapping or clashing rules (e.g., same-field same-op, range overlaps, reject vs override clashes). Conflicts are returned as `warnings` in the `/evaluate/rules` and `/evaluate/rules/diff` endpoints.
- **Diff Severity Categorization**: The `/evaluate/rules/diff` endpoint classifies ruleset changes as `breaking`, `behavioral`, or `cosmetic` based on score deltas and action changes.
- **Performance Metrics**: Evaluation responses include `evaluation_time_ms`. Enabling `debug=true` query parameter provides granular `per_rule_timings_ms`.
- **Hot-Reload (Dev)**: When `INFERENCE_GATEWAY_RULES_WATCH=true`, the gateway watches the rules file for changes and reloads it automatically with a last-known-good fallback.
- **Rules CLI**: A native Go CLI for offline rule evaluation and diffing.
  - `rules-cli evaluate --features features.json --base-score 50 --rules rules.json`
  - `rules-cli diff --features features.json --base-score 50 --a rules_a.json --b rules_b.json`
- **Shadow Mode**: In all evaluation paths, `shadow_mode=true` performs rule simulation (dry-run) to preview effects without affecting the final production decision.

### Analytics Service (Go)

```
ANALYTICS_CRUD_PORT=50051
ANALYTICS_CRUD_TARGET=analytics:50051
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
