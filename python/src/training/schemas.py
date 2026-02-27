"""Pydantic schemas for API request/response models."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Currency(str, Enum):
    """Supported currency codes."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"


class ErrorCategory(str, Enum):
    """Canonical error categories for inference, fallback, and hardening."""

    # Core Prediction Failures
    MODEL_NOT_LOADED = "model_not_loaded"
    NO_HISTORY = "no_history"
    MISSING_FEATURES = "missing_features"
    MODEL_PREDICTION_ERROR = "model_prediction_error"
    HEURISTIC_DISABLED = "heuristic_disabled"

    # Infrastructure / External Services
    MLFLOW_UNAVAILABLE = "mlflow_unavailable"
    MLFLOW_FETCH_ERROR = "mlflow_fetch"
    ARTIFACT_MISSING = "artifact_missing"
    REGISTRY_SYNC_FAILURE = "registry_sync_failure"

    # Validation & Hardening
    SCHEMA_MISMATCH = "schema_mismatch"
    RESUME_INVARIANT_MISMATCH = "resume_invariant_mismatch"

    # Drift Detection
    TIED_QUANTILES = "tied_quantiles"
    INSUFFICIENT_DATA = "insufficient_data"
    INSUFFICIENT_BUCKET_MASS = "insufficient_bucket_mass"

    # Generic
    UNKNOWN = "unknown"


class SignalRequest(BaseModel):
    """Request schema for signal evaluation endpoint."""

    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        examples=["user_abc123"],
    )
    amount: Decimal = Field(
        ...,
        gt=0,
        description="Transaction amount",
        examples=[150.00],
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Currency code",
    )
    client_transaction_id: str = Field(
        ...,
        description="Client-provided transaction identifier for idempotency",
        examples=["txn_xyz789"],
    )
    fallback_mode: Literal["probability", "error", "zero"] | None = Field(
        default=None,
        description="Override default fallback behavior for this request",
    )
    include_importance: bool = Field(
        default=False,
        description="Whether to include feature importance in the response",
    )


class RiskComponent(BaseModel):
    """Individual risk factor contributing to the score."""

    key: str = Field(
        ...,
        description="Machine-readable identifier for the risk factor",
        examples=["velocity", "history", "amount_ratio"],
    )
    label: str = Field(
        ...,
        description="Human-readable description of the risk factor",
        examples=["high_transaction_velocity", "insufficient_history"],
    )


class MatchedRule(BaseModel):
    """Rule that matched during evaluation."""

    rule_id: str = Field(
        ...,
        description="Rule identifier",
        examples=["high_velocity", "reject_large_amount"],
    )
    severity: str = Field(
        default="medium",
        description="Rule severity level",
        examples=["low", "medium", "high"],
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of why the rule matched",
        examples=["high transaction velocity", "transaction amount exceeds threshold"],
    )
    explanation: str | None = Field(
        default=None,
        description="Detailed explanation of the rule match",
    )


class SignalResponse(BaseModel):
    """Response schema for signal evaluation endpoint."""

    request_id: str = Field(
        ...,
        description="Unique identifier for this evaluation request",
        examples=["req_123xyz"],
    )
    score: int = Field(
        ...,
        ge=1,
        le=99,
        description="Risk score from 1 (lowest risk) to 99 (highest risk)",
        examples=[85],
    )
    risk_label: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ...,
        description="Risk category label",
        examples=["HIGH"],
    )
    latency_ms: float = Field(
        ...,
        description="Inference latency in milliseconds",
        examples=[45.2],
    )
    risk_components: list[RiskComponent] = Field(
        default_factory=list,
        description="List of risk factors contributing to the score",
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for evaluation",
        examples=["v1.0.0"],
    )
    matched_rules: list[MatchedRule] = Field(
        default_factory=list,
        description="Decision rules that matched this request",
    )
    model_score: int | None = Field(
        default=None,
        description="Raw model score before rule adjustments (if rules applied)",
        examples=[75],
    )
    rules_version: str | None = Field(
        default=None,
        description="Version of decision rules applied",
        examples=["v1"],
    )
    shadow_matched_rules: list[MatchedRule] = Field(
        default_factory=list,
        description="Shadow rules that matched (evaluated but not applied to score)",
    )
    debug: dict[str, Any] | None = Field(
        default=None,
        description="Optional debug information (behind flag)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "req_123xyz",
                    "score": 85,
                    "risk_components": [
                        {"key": "velocity", "label": "high_transaction_velocity"},
                        {"key": "history", "label": "insufficient_history"},
                    ],
                    "model_version": "v1.0.0",
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Response schema for prediction-only endpoint."""

    request_id: str = Field(..., description="Request identifier")
    model_score: int = Field(..., ge=1, le=99, description="Calibrated model score")
    model_version: str = Field(..., description="Model version")
    model_loaded: bool = Field(..., description="True if a custom model was used")
    fallback_used: bool = Field(
        default=False, description="True if heuristic fallback was used"
    )
    latency_ms: float = Field(..., description="Prediction latency")
    feature_importance: dict[str, float] | None = Field(
        None, description="Feature importance scores"
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict, description="Diagnostic info"
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    version: str = Field(default="0.1.0")


class SplitStrategy(str, Enum):
    """Supported train/test split strategies."""

    TEMPORAL = "temporal"
    TEMPORAL_STRATIFIED = "temporal_stratified"
    GROUP_TEMPORAL = "group_temporal"
    KFOLD_TEMPORAL = "kfold_temporal"
    EXPANDING_WINDOW = "expanding_window"


class SplitConfig(BaseModel):
    """Configuration for train/test split and optional CV."""

    strategy: SplitStrategy = Field(
        default=SplitStrategy.TEMPORAL,
        description="Split strategy",
    )
    n_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of folds for CV strategies",
    )
    stratify_column: str | None = Field(
        default=None,
        description="Column to stratify on (e.g. is_fraudulent)",
    )
    group_column: str | None = Field(
        default="user_id",
        description="Column for group-based splits",
    )
    validation_fraction: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Validation fraction when using validation split",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")


class TuningStrategy(str, Enum):
    """Hyperparameter tuning strategy."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


class TuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""

    enabled: bool = Field(default=False, description="Enable tuning")
    strategy: TuningStrategy = Field(
        default=TuningStrategy.BAYESIAN,
        description="Tuning strategy",
    )
    n_trials: int = Field(default=20, ge=5, le=100, description="Number of trials")
    timeout_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Max tuning time in minutes",
    )
    metric: str = Field(default="pr_auc", description="Metric to optimize")
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize",
        description="Optimization direction",
    )
    selected_trial_number: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional trial number to use instead of best trial. "
            "If None, uses best trial automatically."
        ),
    )
    search_space: dict[str, str] | None = Field(
        default=None,
        description="Optional overrides for search space (JSON strings).",
    )


class TrainRequest(BaseModel):
    """Request schema for model training endpoint."""

    max_depth: int = Field(
        default=6,
        ge=2,
        le=12,
        description="Maximum depth of XGBoost trees",
    )
    training_window_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Number of days for training window",
    )
    selected_feature_columns: list[str] | None = Field(
        default=None,
        description="Feature columns for training. If None, uses defaults.",
    )
    split_config: SplitConfig = Field(
        default_factory=SplitConfig,
        description="Split and optional CV configuration",
    )
    tuning_config: TuningConfig = Field(
        default_factory=TuningConfig,
        description="Hyperparameter tuning configuration",
    )
    n_estimators: int = Field(default=100, ge=50, le=500)
    learning_rate: float = Field(default=0.1, ge=0.01, le=0.3)
    min_child_weight: int = Field(default=1, ge=1, le=10)
    subsample: float = Field(default=1.0, ge=0.5, le=1.0)
    colsample_bytree: float = Field(default=1.0, ge=0.5, le=1.0)
    gamma: float = Field(default=0.0, ge=0.0, le=5.0)
    reg_alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    reg_lambda: float = Field(default=1.0, ge=0.0, le=10.0)
    random_state: int = Field(default=42)
    early_stopping_rounds: int | None = Field(default=None, ge=5, le=50)
    min_cal_samples: int = Field(
        default=100, ge=0, description="Min samples required for calibration"
    )
    min_cal_pos: int = Field(
        default=10, ge=0, description="Min positives required for calibration"
    )
    min_cal_neg: int = Field(
        default=10, ge=0, description="Min negatives required for calibration"
    )
    feature_groups: list[str] | None = Field(
        default=None,
        description="List of feature groups to include (e.g. 'transaction')",
    )
    feature_resolution_mode: Literal["strict", "best_effort"] = Field(
        default="strict",
        description="How to handle missing features from groups",
    )

    def model_post_init(self, __context) -> None:
        """Validate selected_feature_columns if provided."""
        if self.selected_feature_columns is not None:
            if len(self.selected_feature_columns) == 0:
                raise ValueError("selected_feature_columns cannot be empty if provided")


class TrainResponse(BaseModel):
    """Response schema for model training endpoint."""

    success: bool = Field(..., description="Whether training completed successfully")
    run_id: str | None = Field(None, description="MLflow run ID if successful")
    error: str | None = Field(None, description="Error message if training failed")


class DeployModelRequest(BaseModel):
    """Request schema for deploying a model to production."""

    actor: str = Field(..., description="Who is deploying this model")
    reason: str = Field(..., description="Reason for deployment")


class DeployModelResponse(BaseModel):
    """Response schema for deploying a model."""

    success: bool = Field(..., description="Whether deployment completed successfully")
    model_version: str = Field(..., description="Deployed model version")
    deployed_at: str = Field(..., description="Deployment timestamp (ISO format)")
    previous_version: str | None = Field(
        None, description="Previous model version if replaced"
    )


class GenerateDataRequest(BaseModel):
    """Request schema for data generation endpoint."""

    num_users: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Number of unique users to generate",
    )
    fraud_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Fraction of users with fraud events (0.0-1.0)",
    )
    drop_existing: bool = Field(
        default=False,
        description="Drop existing tables before generating new data",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    idempotency_key: str = Field(
        default="",
        description="Optional key to prevent duplicate generation",
    )


class GenerateDataResponse(BaseModel):
    """Response schema for data generation endpoint."""

    success: bool = Field(..., description="Whether generation completed successfully")
    total_records: int = Field(..., description="Total records generated")
    fraud_records: int = Field(..., description="Number of fraud records")
    features_materialized: int = Field(
        ..., description="Number of feature snapshots created"
    )
    error: str | None = Field(None, description="Error message if generation failed")


class ClearDataResponse(BaseModel):
    """Response schema for data clearing endpoint."""

    success: bool = Field(..., description="Whether clearing completed successfully")
    tables_cleared: list[str] = Field(
        default_factory=list,
        description="List of tables that were cleared",
    )
    error: str | None = Field(None, description="Error message if clearing failed")


class TrainingRunSpec(BaseModel):
    """Canonical specification for a training run, used for reproducibility."""

    schema_version: int = 1
    run_id: str
    model_name: str
    created_at: str
    training_config_hash: str
    feature_set_id: str | None = None
    feature_set_hash: str
    resolved_features: list[str]
    feature_resolution_mode: str
    requested_feature_groups: list[str] | None = None
    resolved_feature_groups: list[str] | None = None
    split_config: SplitConfig | None = None
    tuning_config: TuningConfig | None = None
    training_window_days: int


# =============================================================================
# Monitoring and Drift Schemas
# =============================================================================


class BucketingMetadata(BaseModel):
    """Metadata about the bucketing used for drift calculation."""

    bucket_type: str = Field(..., description="bins | quantiles")
    n_buckets: int = Field(..., description="Requested number of buckets")
    actual_buckets: int = Field(..., description="Actual number of buckets used")
    breakpoints: list[float] = Field(..., description="List of bucket boundaries")


class FeatureDriftDetail(BaseModel):
    """Per-feature drift information."""

    feature: str = Field(..., description="Feature name")
    psi: float = Field(..., ge=0.0, description="PSI value")
    status: str = Field(..., description="OK | WARNING | CRITICAL")
    bucketing: BucketingMetadata | None = Field(default=None)


class AlertItem(BaseModel):
    """Structured alert for drift or monitoring events."""

    severity: Literal["warning", "critical"] = Field(..., description="Alert severity")
    feature: str = Field(..., description="Feature name or component")
    psi: float = Field(..., description="PSI value")
    threshold: float = Field(..., description="Threshold exceeded")
    recommendation: str = Field(..., description="Recommended action")


class DriftStatusResponse(BaseModel):
    """Response schema for drift status endpoint."""

    status: str = Field(
        ...,
        description="Overall status: ok | warn | fail | unknown",
    )
    computed_at: str = Field(..., description="ISO timestamp of computation")
    cached: bool = Field(..., description="Whether result was from cache")
    reference_window: str = Field(
        ...,
        description="Reference data description (e.g., 'Production model v3')",
    )
    current_window: str = Field(
        ...,
        description="Current window (e.g., 'Last 24 hours')",
    )
    reference_size: int = Field(..., description="Reference sample count")
    live_size: int = Field(..., description="Live sample count")
    top_features: list[FeatureDriftDetail] = Field(
        default_factory=list,
        description="Features sorted by PSI descending",
    )
    alerts: list[AlertItem] = Field(
        default_factory=list,
        description="List of structured alerts based on drift thresholds",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Threshold values (warn, fail)",
    )
    error: str | None = Field(None, description="Error message if status=unknown")


class ScoreDistributionItem(BaseModel):
    """Per-bucket score distribution info."""

    bucket: list[int] = Field(..., description="Bucket range [min, max]")
    baseline_ratio: float = Field(..., description="Ratio in baseline data")
    observed_ratio: float = Field(..., description="Ratio in observed live data")
    observed_count: int = Field(..., description="Count in observed live data")


class ScoreDistributionResponse(BaseModel):
    """Response schema for score distribution monitoring."""

    computed_at: str = Field(..., description="ISO timestamp of computation")
    observed_size: int = Field(..., description="Number of live samples")
    baseline_size: int | None = Field(None, description="Number of baseline samples")
    divergence: float = Field(..., description="Divergence metric value")
    divergence_metric: str = Field(
        default="JS", description="Metric used (JS for Jensen-Shannon)"
    )
    distribution: list[ScoreDistributionItem] = Field(default_factory=list)
    shift_detected: bool = Field(
        ..., description="True if any bucket exceeded 2x baseline"
    )


# =============================================================================
# Analytics CRUD Schemas
# =============================================================================


class DailyStat(BaseModel):
    date: str
    total_transactions: int
    fraud_count: int
    fraud_rate: float
    total_amount: float
    avg_z_score: float


class DailyStatsResponse(BaseModel):
    stats: list[DailyStat]


class TransactionDetail(BaseModel):
    record_id: str
    user_id: str
    created_at: datetime
    is_train_eligible: bool
    is_pre_fraud: bool
    amount: float
    is_fraudulent: bool
    fraud_type: str | None = None
    is_off_hours_txn: bool
    merchant_risk_score: int
    velocity_24h: int
    amount_to_avg_ratio_30d: float
    balance_volatility_z_score: float


class TransactionDetailsResponse(BaseModel):
    transactions: list[TransactionDetail]


class Alert(BaseModel):
    record_id: str
    user_id: str
    created_at: datetime
    amount: float
    is_fraudulent: bool
    fraud_type: str
    merchant_risk_score: int
    velocity_24h: int
    amount_to_avg_ratio_30d: float
    balance_volatility_z_score: float
    computed_risk_score: int


class RecentAlertsResponse(BaseModel):
    alerts: list[Alert]


class AnalyticsOverviewResponse(BaseModel):
    total_records: int
    fraud_records: int
    fraud_rate: float
    unique_users: int
    min_transaction_timestamp: datetime | None = None
    max_transaction_timestamp: datetime | None = None
    min_created_at: datetime | None = None
    max_created_at: datetime | None = None
    total_amount: float
    fraud_amount: float


class TableFingerprint(BaseModel):
    count: int
    max_created_at: datetime | None = None
    max_timestamp: datetime | None = None
    max_id: int | None = None


class DatasetFingerprintResponse(BaseModel):
    generated_records: TableFingerprint
    feature_snapshots: TableFingerprint


class FeatureSample(BaseModel):
    record_id: str
    is_fraudulent: bool
    velocity_24h: float
    amount_to_avg_ratio_30d: float
    balance_volatility_z_score: float


class FeatureSampleResponse(BaseModel):
    samples: list[FeatureSample]


class ColumnInfo(BaseModel):
    table_name: str
    column_name: str
    data_type: str
    is_nullable: str
    ordinal_position: int


class SchemaSummaryResponse(BaseModel):
    columns: list[ColumnInfo]


class RelationshipMetric(BaseModel):
    feature_a: str
    feature_b: str
    metric_type: str  # 'pearson', 'cramers_v', 'eta'
    value: float


class CorrelationPair(BaseModel):
    feature_a: str
    feature_b: str
    value: float


class DatasetCorrelationsResponse(BaseModel):
    pearson: list[CorrelationPair]
    spearman: list[CorrelationPair]
    cramers_v: list[CorrelationPair]
    numeric_columns: list[str]
    categorical_columns: list[str]


class TransactionSearchRequest(BaseModel):
    user_id: str | None = None
    transaction_id: str | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    start_date: str | None = None
    end_date: str | None = None
    is_fraudulent: bool | None = None
    min_score: int | None = None
    max_score: int | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class TransactionSearchResponse(BaseModel):
    transactions: list[TransactionDetail]
    total: int


class DatasetRelationshipsResponse(BaseModel):
    relationships: list[RelationshipMetric]
    target_column: str
