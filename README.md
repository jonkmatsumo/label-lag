# Label Lag

End-to-end ML system for fraud detection with realistic label delay simulation. Generates synthetic transaction data, trains XGBoost models with MLflow tracking, serves predictions via API, and provides a dashboard for analysis and model management.

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
| Dashboard | http://localhost:8501 | Streamlit UI for scoring, analytics, and model training |
| API | http://localhost:8000 | FastAPI fraud scoring and training endpoints |
| API Docs | http://localhost:8000/docs | Swagger UI |
| MLflow | http://localhost:5005 | Experiment tracking and model registry |
| MinIO | http://localhost:9001 | Object storage console (minioadmin/minioadmin) |
| PostgreSQL | localhost:5432 | Database |

All ports are configurable via `.env` file.

## Dashboard

The Streamlit dashboard provides three main views:

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

### Model Lab
- **Train Models**: Configure max depth and training window, train XGBoost models
- **Data Management**: Generate synthetic data or clear existing data
- **Model Registry**: View experiment runs sorted by PR-AUC, promote models to production

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/signal` | POST | Evaluate transaction fraud risk |
| `/train` | POST | Train a new model with MLflow tracking |
| `/reload-model` | POST | Reload production model from registry |
| `/data/generate` | POST | Generate synthetic transaction data |
| `/data/clear` | DELETE | Clear all transaction data |
| `/health` | GET | Health check with model status |

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
├── api/              # FastAPI application
│   ├── main.py       # API endpoints
│   ├── services.py   # SignalEvaluator with ML model integration
│   ├── model_manager.py  # MLflow model loading
│   └── schemas.py    # Pydantic request/response models
├── model/            # ML training pipeline
│   ├── train.py      # XGBoost training with MLflow
│   └── loader.py     # Point-in-time correct data loading
├── monitor/          # Production monitoring
│   └── detect_drift.py   # PSI-based drift detection
├── pipeline/         # Feature engineering
│   └── materialize_features.py  # SQL window functions
├── generator/        # Stateful user simulation
│   └── core.py       # BustOut, Sleeper, and other fraud profiles
├── synthetic_pipeline/   # Core data generation
│   ├── generator.py  # DataGenerator with sequences
│   ├── graph.py      # Graph network generation
│   ├── db/           # Database models and session
│   └── models/       # Pydantic domain models
└── ui/               # Streamlit dashboard
    ├── app.py        # Dashboard application
    ├── data_service.py   # API and DB queries
    └── mlflow_utils.py   # MLflow client utilities
```

### Key Components

- **SignalEvaluator**: Queries real features from `feature_snapshots` table, uses ML model when available, falls back to rule-based scoring for unknown users
- **ModelManager**: Loads production models from MLflow registry, supports hot-reloading on promotion
- **FeatureMaterializer**: SQL window functions compute point-in-time correct features without data leakage
- **DataLoader**: Temporal train/test split respecting label maturity (fraud confirmation dates)
- **DriftDetector**: PSI calculation comparing reference data (from model artifacts) with live feature distributions

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
