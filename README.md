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
3) Open the dashboard at `http://localhost:8501` and verify Live Scoring renders.

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
| Dashboard | 8501 | Streamlit UI for scoring, analytics, model training, and rule authoring |
| API | 8000 | FastAPI fraud scoring and training endpoints |
| API Docs | 8000 | Swagger UI served by the API |
| MLflow | 5005 | Experiment tracking and model registry |
| MinIO API | 9000 | Object storage API for artifacts |
| MinIO Console | 9001 | Object storage console (minioadmin/minioadmin) |
| PostgreSQL | 5432 | Transaction and feature storage |

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

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed.

### Database

```
POSTGRES_USER=synthetic
POSTGRES_PASSWORD=synthetic_dev_password
POSTGRES_DB=synthetic_data
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}
```

### Service Ports

```
DB_PORT=5432
API_PORT=8000
DASHBOARD_PORT=8501
MLFLOW_PORT=5005
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
```

### MLflow / MinIO

```
MLFLOW_TRACKING_URI=http://localhost:5005
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```
