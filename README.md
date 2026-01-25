# Label Lag

End-to-end ML system for fraud detection with realistic label delay simulation. Combines XGBoost models and rule-based decision engine for hybrid fraud scoring. Generates synthetic transaction data, trains models with MLflow tracking, serves predictions via API, and provides a dashboard for analysis, model management, and rule authoring.

## Quick Start

```bash
# Copy .env (see Environment Variables)
cp .env.example .env

# Start all services (convenience wrapper includes infra + app)
docker compose up -d

# Open the dashboard
open http://localhost:8501

# Generate data via dashboard: Model Lab > Generate Data
# Or via CLI:
docker compose exec generator uv run python src/main.py seed --users 1000 --fraud-rate 0.05
```

### Split infra vs app (recommended for development)

Start infrastructure once, then run app separately so you can rebuild app without touching infra:

```bash
# 1. Start infra (db, minio, mlflow). Keep running.
docker compose -f docker-compose.infra.yml up -d

# 2. Start app (api, dashboard, generator). Rebuild frequently.
docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d

# Rebuild only API after source change:
docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build api
docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d api
```

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | Streamlit UI for scoring, analytics, model training, and rule authoring |
| API | http://localhost:8000 | FastAPI fraud scoring and training endpoints |
| API Docs | http://localhost:8000/docs | Swagger UI |
| MLflow | http://localhost:5005 | Experiment tracking and model registry |
| MinIO | http://localhost:9001 | Object storage console (minioadmin/minioadmin) |
| PostgreSQL | localhost:5432 | Database |

All ports are configurable via `.env` file.

## Dashboard

The Streamlit dashboard provides five main views:

### Live Scoring
- Submit transactions for real-time fraud risk evaluation
- Displays current model status (ML model or rule-based fallback)
- Shows risk score (1-99), risk level, and contributing factors
- API latency monitoring

### Historical Analytics
- Model version selector with PRODUCTION/LIVE indicators
- Global metrics: transaction volume, fraud rate, false positive rate
- Time series visualization of daily transaction volume and fraud trends
- Transaction amount distribution by fraud status
- Recent high-risk alerts table

### Synthetic Dataset
- **Data Generation**: Generate synthetic transaction data with configurable user count and fraud rate
- **Dataset Overview**: Total records, fraud statistics, timestamp ranges, schema summary
- **Feature Analysis**: Histograms, box plots, correlation matrices (Pearson, Spearman, Cramér's V)
- **Data Sampling**: Stratified sampling with fraud status preservation

### Model Lab
- **Train Models**: Configure feature columns, hyperparameters, and training window
- **Model Registry**: View experiment runs sorted by PR-AUC, promote models to production
- **Training Progress**: Real-time training metrics and completion status

### Rule Inspector
- **Sandbox**: Deterministic rule testing with custom features and rulesets (no side effects, no DB writes)
- **Shadow Metrics**: Compare production rules vs shadow rules with match counts and overlap statistics
- **Backtest Results**: View historical rule performance metrics and score distributions
- **Suggestions**: Browse heuristic rule recommendations based on feature distributions
- **Draft Rules**: Author, validate, and submit rules through the lifecycle workflow

## API

The API provides endpoints for:

- **Evaluation**: Real-time fraud scoring with hybrid ML + rules (`POST /evaluate/signal`)
- **Training**: Model training with MLflow tracking (`POST /train`)
- **Data**: Synthetic data generation and management (`POST /data/generate`, `DELETE /data/clear`)
- **Rules**: Sandbox testing, draft rule management, heuristic suggestions, shadow metrics, and backtest results

Full API documentation with request/response schemas available at http://localhost:8000/docs (Swagger UI).

## CLI Commands

```bash
# Generate synthetic data
docker compose exec generator uv run python src/main.py seed --users 1000 --fraud-rate 0.05

# View database statistics
docker compose exec generator uv run python src/main.py stats

# Train a model directly
docker compose exec generator uv run python src/model/train.py 30

# Run drift detection
docker compose exec generator uv run python src/monitor/detect_drift.py --hours 24
```

## Development

```bash
# Install dependencies locally
make install

# Run tests (Python 3.12)
make test

# Linting
make lint         # Check only
make lint-fix     # Auto-fix issues
```

### Docker workflow (split compose)

| When | Command |
|------|---------|
| **Once per machine** | `cp .env.example .env` |
| **Start infra** (long-lived) | `docker compose -f docker-compose.infra.yml up -d` |
| **Start app** (daily) | `docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d` |
| **After source change** | `docker compose -f ... -f ... build api` then `up -d api` (or restart) |
| **After dependency change** | `docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build --no-cache` then `up -d` |
| **App down only** | `docker compose -f docker-compose.app.yml down` |
| **Infra down only** | `docker compose -f docker-compose.infra.yml down` |

**Always start infra before app.** The app compose file references the shared network; use both `-f docker-compose.infra.yml -f docker-compose.app.yml` when running app.

### Reset commands

| Scope | Command |
|-------|---------|
| **Wipe app only** | `docker compose -f docker-compose.app.yml down` |
| **Wipe infra only** | `docker compose -f docker-compose.infra.yml down` |
| **Full reset (all data)** | `docker compose -f docker-compose.infra.yml down -v` then `docker compose -f docker-compose.app.yml down` |
| **Reset DB only** | `docker compose -f docker-compose.infra.yml stop db` → `docker compose -f docker-compose.infra.yml rm -f db` → `docker volume rm labellag_postgres_data` → `docker compose -f docker-compose.infra.yml up -d db` |
| **Reset MinIO only** | `docker compose -f docker-compose.infra.yml stop minio` → `docker compose -f docker-compose.infra.yml rm -f minio create-buckets` → `docker volume rm labellag_minio_data` → `docker compose -f docker-compose.infra.yml up -d minio create-buckets` |

**Volume migration:** If you previously used the legacy setup, volumes `postgres_data` and `minio_data` are orphaned. Remove them after confirming the new setup works: `docker volume rm postgres_data minio_data`.

## Architecture

```
src/
├── api/              # FastAPI app, rule engine, evaluation services
├── model/            # XGBoost training, evaluation, tuning
├── monitor/          # Drift detection (PSI)
├── pipeline/         # Feature materialization (SQL window functions)
├── generator/        # Stateful fraud profile simulation
├── synthetic_pipeline/  # Core data generation, DB models
└── ui/               # Streamlit dashboard
```

### Package Responsibilities

- **`api/`**: FastAPI application with hybrid ML + rule evaluation, rule lifecycle management (draft → active → shadow), validation, audit logging, versioning, backtesting, and heuristic suggestions
- **`model/`**: XGBoost training pipeline with MLflow integration, hyperparameter tuning (Optuna), temporal train/test splitting, and score calibration
- **`monitor/`**: Population Stability Index (PSI) drift detection comparing production model reference data with live feature distributions
- **`pipeline/`**: Point-in-time correct feature engineering using SQL window functions to prevent data leakage
- **`generator/`**: Stateful user simulation with fraud profiles (BustOut, Sleeper ATO) and label delay modeling
- **`synthetic_pipeline/`**: Core synthetic data generation with graph network relationships, database persistence, and Pydantic domain models
- **`ui/`**: Streamlit dashboard with five pages: Live Scoring, Historical Analytics, Synthetic Dataset, Model Lab, and Rule Inspector

### Key Components

- **SignalEvaluator**: Hybrid scoring combining ML model predictions with rule-based adjustments, queries features from `feature_snapshots` table
- **ModelManager**: Loads production models from MLflow registry with hot-reload support on promotion
- **RuleEvaluator**: Matches transaction features against rule conditions, applies actions (override_score, clamp, reject) with precedence
- **DraftRuleStore**: Manages draft rule authoring workflow with validation, conflict detection, and state transitions
- **FeatureMaterializer**: SQL window functions compute point-in-time correct features without future data leakage
- **DataLoader**: Temporal train/test split respecting label maturity (fraud confirmation dates)
- **DriftDetector**: PSI calculation comparing reference data (from model artifacts) with live feature distributions

## Rule Engine

The system includes a comprehensive rule-based decision engine that works alongside ML models for hybrid fraud scoring.

### Rule Structure

Rules consist of:
- **Field**: Feature to evaluate (e.g., `velocity_24h`, `amount_to_avg_ratio_30d`)
- **Operator**: Comparison operator (`>`, `>=`, `<`, `<=`, `==`, `in`, `not_in`)
- **Value**: Threshold or comparison value
- **Action**: What to do when matched (`override_score`, `clamp_min`, `clamp_max`, `reject`)
- **Severity**: Risk level (`low`, `medium`, `high`)
- **Status**: Lifecycle state (see below)

### Rule Lifecycle

Rules progress through states with controlled transitions:

```mermaid
stateDiagram-v2
    [*] --> draft
    draft --> pending_review
    pending_review --> active: approved
    pending_review --> draft: rejected
    active --> shadow: test mode
    active --> disabled: deactivate
    shadow --> active: promote
    shadow --> disabled: deactivate
    disabled --> active: reactivate
    disabled --> archived: permanent removal
    archived --> [*]
```

- **draft**: Initial creation, can be edited and validated
- **pending_review**: Submitted for approval, cannot be edited
- **active**: Production-active, affects scoring
- **shadow**: Evaluated but not applied (for A/B testing)
- **disabled**: Temporarily inactive
- **archived**: Terminal state, historical record only

### Shadow Mode

Shadow rules are evaluated alongside production rules but don't affect the final score. This enables:
- Safe testing of new rules without production impact
- Comparison metrics between production and shadow rule performance
- Gradual rollout of rule changes

### Sandbox Testing

The sandbox provides deterministic rule evaluation with no side effects:
- Test rules against arbitrary feature inputs
- Use custom rulesets (ephemeral, not saved)
- Inspect matched rules and explanations
- No database writes or production impact

## Fraud Patterns

| Pattern | Description | Key Indicators |
|---------|-------------|----------------|
| Liquidity Crunch | Overdraft attempt | balance z-score < -2.5, returned=True |
| Link Burst | Rapid bank linking | 5-15 connections in 24h |
| ATO (Account Takeover) | Compromised account | amount_ratio > 5.0, off-hours, recent identity change |
| Bust-Out | Build trust then fraud | 20-50 legit transactions, then >500% spike |
| Sleeper ATO | Dormant then active | 30+ days dormancy, link burst, high-value withdrawal |

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
# Database
POSTGRES_USER=synthetic
POSTGRES_PASSWORD=synthetic_dev_password
POSTGRES_DB=synthetic_data
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}

# Service ports
DB_PORT=5432
API_PORT=8000
DASHBOARD_PORT=8501
MLFLOW_PORT=5005
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001

# MLflow (set automatically in Docker)
MLFLOW_TRACKING_URI=http://localhost:5005
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

## Model Workflow

1. **Generate Data**: Dashboard > Model Lab > Generate Data (or CLI)
2. **Train Model**: Dashboard > Model Lab > Start Training
3. **Review Metrics**: View experiment runs sorted by PR-AUC
4. **Promote**: Select best run and click "Promote to Production"
5. **Verify**: Check Live Scoring page shows new model version

The API automatically reloads the production model when promoted via the dashboard.

## Drift Detection

Monitor feature distributions for drift using Population Stability Index (PSI):

```bash
# Check last 24 hours
docker compose exec generator uv run python src/monitor/detect_drift.py

# Custom window and threshold
docker compose exec generator uv run python src/monitor/detect_drift.py --hours 48 --threshold 0.25

# JSON output for automation
docker compose exec generator uv run python src/monitor/detect_drift.py --json
```

PSI thresholds:
- < 0.1: No significant drift
- 0.1 - 0.2: Warning, monitor closely
- >= 0.2: Critical, action required
