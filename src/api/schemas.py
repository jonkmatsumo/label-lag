"""Pydantic schemas for API request/response models."""

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


class GenerateDataRequest(BaseModel):
    """Request schema for data generation endpoint."""

    num_users: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Number of unique users to generate",
    )
    fraud_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Fraction of users with fraud events (0.0-0.5)",
    )
    drop_existing: bool = Field(
        default=False,
        description="Drop existing tables before generating new data",
    )


class GenerateDataResponse(BaseModel):
    """Response schema for data generation endpoint."""

    success: bool = Field(..., description="Whether generation completed successfully")
    total_records: int = Field(0, description="Total records generated")
    fraud_records: int = Field(0, description="Number of fraud records")
    features_materialized: int = Field(
        0, description="Number of feature snapshots created"
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


# =============================================================================
# Rule Inspector Schemas (Phase 1)
# =============================================================================


class SandboxFeatures(BaseModel):
    """Feature input for sandbox evaluation."""

    velocity_24h: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Transaction count in last 24 hours",
    )
    amount_to_avg_ratio_30d: float = Field(
        default=1.0,
        ge=0.0,
        le=20.0,
        description="Transaction amount vs 30-day average ratio",
    )
    balance_volatility_z_score: float = Field(
        default=0.0,
        ge=-5.0,
        le=5.0,
        description="Balance volatility z-score",
    )
    bank_connections_24h: int = Field(
        default=0,
        ge=0,
        le=30,
        description="Bank connections in last 24 hours",
    )
    merchant_risk_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Merchant risk score (0-100)",
    )
    has_history: bool = Field(
        default=True,
        description="Whether user has transaction history",
    )
    transaction_amount: float = Field(
        default=100.0,
        ge=0.0,
        le=10000.0,
        description="Transaction amount",
    )


class RuleDefinition(BaseModel):
    """A single rule definition for sandbox testing."""

    id: str = Field(..., description="Rule identifier")
    field: str = Field(..., description="Feature field to check")
    op: str = Field(
        ...,
        description="Comparison operator (>, >=, <, <=, ==, in, not_in)",
    )
    value: int | float | list = Field(..., description="Value to compare against")
    action: str = Field(
        ...,
        description="Action to take (override_score, clamp_min, clamp_max, reject)",
    )
    score: int | None = Field(None, description="Score for score-modifying actions")
    severity: str = Field(
        default="medium", description="Rule severity (low/medium/high)"
    )
    reason: str = Field(default="", description="Human-readable reason")
    status: str = Field(default="active", description="Rule status")


class RuleSetDefinition(BaseModel):
    """RuleSet definition for sandbox testing."""

    version: str = Field(..., description="RuleSet version string")
    rules: list[RuleDefinition] = Field(
        default_factory=list, description="List of rules"
    )


class SandboxEvaluateRequest(BaseModel):
    """Request schema for sandbox rule evaluation."""

    features: SandboxFeatures = Field(
        default_factory=SandboxFeatures,
        description="Feature values for evaluation",
    )
    base_score: int = Field(
        default=50,
        ge=1,
        le=99,
        description="Base score before rule application",
    )
    ruleset: RuleSetDefinition | None = Field(
        default=None,
        description="Custom ruleset to evaluate. If None, uses production ruleset.",
    )


class SandboxMatchedRule(BaseModel):
    """Matched rule in sandbox evaluation result."""

    rule_id: str = Field(..., description="Rule identifier")
    severity: str = Field(..., description="Rule severity")
    reason: str = Field(..., description="Human-readable reason")
    action: str = Field(default="", description="Action taken")
    score: int | None = Field(None, description="Score value if applicable")


class SandboxEvaluateResponse(BaseModel):
    """Response schema for sandbox rule evaluation."""

    final_score: int = Field(..., description="Final score after rule application")
    matched_rules: list[SandboxMatchedRule] = Field(
        default_factory=list,
        description="Production rules that matched",
    )
    explanations: list[dict] = Field(
        default_factory=list,
        description="Explanations for matched rules",
    )
    shadow_matched_rules: list[SandboxMatchedRule] = Field(
        default_factory=list,
        description="Shadow rules that matched (not applied to score)",
    )
    rejected: bool = Field(
        default=False, description="Whether transaction was rejected"
    )
    ruleset_version: str = Field(..., description="Version of ruleset used")


class RuleMetricsItem(BaseModel):
    """Metrics for a single rule over a time period."""

    rule_id: str = Field(..., description="Rule identifier")
    period_start: str = Field(..., description="Period start date (ISO format)")
    period_end: str = Field(..., description="Period end date (ISO format)")
    production_matches: int = Field(default=0, description="Production match count")
    shadow_matches: int = Field(default=0, description="Shadow match count")
    overlap_count: int = Field(default=0, description="Overlap count")
    production_only_count: int = Field(default=0, description="Production-only matches")
    shadow_only_count: int = Field(default=0, description="Shadow-only matches")


class ShadowComparisonResponse(BaseModel):
    """Response schema for shadow mode comparison."""

    period_start: str = Field(..., description="Period start date (ISO format)")
    period_end: str = Field(..., description="Period end date (ISO format)")
    rule_metrics: list[RuleMetricsItem] = Field(
        default_factory=list,
        description="Per-rule metrics",
    )
    total_requests: int = Field(default=0, description="Total requests in period")


class BacktestMetricsResponse(BaseModel):
    """Backtest metrics in response."""

    total_records: int = Field(default=0, description="Total records processed")
    matched_count: int = Field(default=0, description="Records with matched rules")
    match_rate: float = Field(default=0.0, description="Match rate (0.0-1.0)")
    score_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Score distribution by range",
    )
    score_mean: float = Field(default=0.0, description="Mean score")
    score_std: float = Field(default=0.0, description="Score standard deviation")
    score_min: int = Field(default=0, description="Minimum score")
    score_max: int = Field(default=0, description="Maximum score")
    rejected_count: int = Field(default=0, description="Number of rejections")
    rejected_rate: float = Field(default=0.0, description="Rejection rate (0.0-1.0)")


class BacktestResultResponse(BaseModel):
    """Response schema for a single backtest result."""

    job_id: str = Field(..., description="Backtest job identifier")
    rule_id: str | None = Field(None, description="Rule ID if testing single rule")
    ruleset_version: str = Field(..., description="RuleSet version tested")
    start_date: str = Field(..., description="Backtest start date (ISO format)")
    end_date: str = Field(..., description="Backtest end date (ISO format)")
    metrics: BacktestMetricsResponse = Field(..., description="Computed metrics")
    completed_at: str = Field(..., description="Completion timestamp (ISO format)")
    error: str | None = Field(None, description="Error message if backtest failed")


class BacktestResultsListResponse(BaseModel):
    """Response schema for backtest results list."""

    results: list[BacktestResultResponse] = Field(
        default_factory=list,
        description="List of backtest results",
    )
    total: int = Field(default=0, description="Total number of results")


class SuggestionEvidence(BaseModel):
    """Evidence supporting a rule suggestion."""

    statistic: str = Field(..., description="Statistical measure used")
    value: float = Field(..., description="Threshold value")
    mean: float = Field(..., description="Mean of the distribution")
    std: float = Field(..., description="Standard deviation")
    sample_count: int = Field(default=0, description="Number of samples analyzed")


class RuleSuggestionResponse(BaseModel):
    """Response schema for a single rule suggestion."""

    field: str = Field(..., description="Feature field for the suggested rule")
    operator: str = Field(..., description="Comparison operator")
    threshold: float = Field(..., description="Threshold value")
    action: str = Field(..., description="Suggested action")
    suggested_score: int = Field(..., description="Suggested score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence (0.0-1.0)")
    evidence: SuggestionEvidence = Field(..., description="Supporting evidence")
    reason: str = Field(default="", description="Human-readable reason")


class SuggestionsListResponse(BaseModel):
    """Response schema for suggestions list."""

    suggestions: list[RuleSuggestionResponse] = Field(
        default_factory=list,
        description="List of rule suggestions",
    )
    total: int = Field(default=0, description="Total number of suggestions")


# =============================================================================
# Draft Rule Schemas
# =============================================================================


class DraftRuleCreateRequest(BaseModel):
    """Request schema for creating a draft rule."""

    id: str = Field(..., description="Rule identifier")
    field: str = Field(..., description="Feature field to check")
    op: str = Field(
        ...,
        description="Comparison operator (>, >=, <, <=, ==, in, not_in)",
    )
    value: int | float | list = Field(..., description="Value to compare against")
    action: str = Field(
        ...,
        description="Action to take (override_score, clamp_min, clamp_max, reject)",
    )
    score: int | None = Field(
        None,
        description="Score for score-modifying actions",
    )
    severity: str = Field(
        default="medium",
        description="Rule severity (low/medium/high)",
    )
    reason: str = Field(default="", description="Human-readable reason")
    actor: str = Field(..., description="Who is creating this rule")


class DraftRuleUpdateRequest(BaseModel):
    """Request schema for updating a draft rule."""

    field: str | None = Field(None, description="Feature field to check")
    op: str | None = Field(
        None,
        description="Comparison operator (>, >=, <, <=, ==, in, not_in)",
    )
    value: int | float | list | None = Field(
        None, description="Value to compare against"
    )
    action: str | None = Field(
        None,
        description="Action to take (override_score, clamp_min, clamp_max, reject)",
    )
    score: int | None = Field(
        None,
        description="Score for score-modifying actions",
    )
    severity: str | None = Field(None, description="Rule severity (low/medium/high)")
    reason: str | None = Field(None, description="Human-readable reason")
    actor: str = Field(..., description="Who is updating this rule")


class ConflictResponse(BaseModel):
    """Response schema for a rule conflict."""

    rule1_id: str = Field(..., description="First rule ID")
    rule2_id: str = Field(..., description="Second rule ID")
    conflict_type: str = Field(..., description="Type of conflict")
    description: str = Field(..., description="Human-readable description")


class RedundancyResponse(BaseModel):
    """Response schema for a rule redundancy."""

    rule_id: str = Field(..., description="Redundant rule ID")
    redundant_with: str = Field(..., description="Rule it's redundant with")
    redundancy_type: str = Field(..., description="Type of redundancy")
    description: str = Field(..., description="Human-readable description")


class ValidationResult(BaseModel):
    """Validation result for a draft rule."""

    conflicts: list[ConflictResponse] = Field(
        default_factory=list,
        description="List of conflicts",
    )
    redundancies: list[RedundancyResponse] = Field(
        default_factory=list,
        description="List of redundancies",
    )
    is_valid: bool = Field(..., description="Whether rule is valid (no conflicts)")


class DraftRuleResponse(BaseModel):
    """Response schema for a draft rule."""

    rule_id: str = Field(..., description="Rule identifier")
    field: str = Field(..., description="Feature field")
    op: str = Field(..., description="Comparison operator")
    value: int | float | list = Field(..., description="Comparison value")
    action: str = Field(..., description="Action to take")
    score: int | None = Field(None, description="Score value")
    severity: str = Field(..., description="Rule severity")
    reason: str = Field(..., description="Human-readable reason")
    status: str = Field(..., description="Rule status")
    created_at: str | None = Field(None, description="Creation timestamp (ISO format)")


class DraftRuleCreateResponse(BaseModel):
    """Response schema for creating a draft rule."""

    rule_id: str = Field(..., description="Rule identifier")
    rule: DraftRuleResponse = Field(..., description="Created rule")
    validation: ValidationResult = Field(..., description="Validation results")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")


class DraftRuleListResponse(BaseModel):
    """Response schema for listing draft rules."""

    rules: list[DraftRuleResponse] = Field(
        default_factory=list,
        description="List of draft rules",
    )
    total: int = Field(default=0, description="Total number of rules")


class DraftRuleUpdateResponse(BaseModel):
    """Response schema for updating a draft rule."""

    rule: DraftRuleResponse = Field(..., description="Updated rule")
    version_id: str = Field(..., description="Version ID of the update")
    validation: ValidationResult = Field(..., description="Validation results")


class DraftRuleValidateRequest(BaseModel):
    """Request schema for validating a draft rule."""

    include_existing_rules: bool = Field(
        default=True,
        description="Validate against production ruleset",
    )


class DraftRuleValidateResponse(BaseModel):
    """Response schema for draft rule validation."""

    schema_errors: list[str] = Field(
        default_factory=list,
        description="Schema validation errors",
    )
    conflicts: list[ConflictResponse] = Field(
        default_factory=list,
        description="List of conflicts",
    )
    redundancies: list[RedundancyResponse] = Field(
        default_factory=list,
        description="List of redundancies",
    )
    is_valid: bool = Field(
        ..., description="Whether rule is valid (no errors or conflicts)"
    )


class DraftRuleSubmitRequest(BaseModel):
    """Request schema for submitting a draft rule for review."""

    actor: str = Field(..., description="Who is submitting this rule")
    justification: str = Field(
        ...,
        min_length=10,
        description="Justification for submission (min 10 characters)",
    )


class DraftRuleSubmitResponse(BaseModel):
    """Response schema for submitting a draft rule."""

    rule: DraftRuleResponse = Field(..., description="Rule with pending_review status")
    submitted_at: str = Field(..., description="Submission timestamp (ISO format)")
    audit_id: str | None = Field(None, description="Audit record ID if available")


class AcceptSuggestionRequest(BaseModel):
    """Request schema for accepting a suggestion as a draft rule."""

    actor: str = Field(..., description="Who is accepting this suggestion")
    suggestion: RuleSuggestionResponse = Field(
        ...,
        description="The suggestion to accept (from suggestions list)",
    )
    custom_id: str | None = Field(
        None,
        description="Custom rule ID (overrides auto-generated)",
    )
    edits: dict[str, Any] | None = Field(
        None,
        description="Optional field overrides (field, op, value, action, score, etc.)",
    )


class AcceptSuggestionResponse(BaseModel):
    """Response schema for accepting a suggestion."""

    rule: DraftRuleResponse = Field(..., description="Created draft rule")
    rule_id: str = Field(..., description="Rule identifier")
    source_suggestion: dict[str, Any] = Field(
        ...,
        description=(
            "Source suggestion metadata (confidence, evidence, field, threshold)"
        ),
    )


# =============================================================================
# Approval and Activation Schemas
# =============================================================================


class ApproveRuleRequest(BaseModel):
    """Request schema for approving a pending rule."""

    approver: str = Field(..., description="Who is approving this rule")
    reason: str = Field(
        default="",
        description="Optional reason for approval",
    )


class ApproveRuleResponse(BaseModel):
    """Response schema for approving a rule."""

    rule: DraftRuleResponse = Field(..., description="Approved rule")
    approved_at: str = Field(..., description="Approval timestamp (ISO format)")


class RejectRuleRequest(BaseModel):
    """Request schema for rejecting a pending rule."""

    actor: str = Field(..., description="Who is rejecting this rule")
    reason: str = Field(
        ...,
        min_length=10,
        description="Reason for rejection (min 10 characters)",
    )


class RejectRuleResponse(BaseModel):
    """Response schema for rejecting a rule."""

    rule: DraftRuleResponse = Field(..., description="Rejected rule (back to draft)")
    rejected_at: str = Field(..., description="Rejection timestamp (ISO format)")


class ActivateRuleRequest(BaseModel):
    """Request schema for activating a rule."""

    actor: str = Field(..., description="Who is activating this rule")
    approver: str | None = Field(
        None, description="Approver (required for approval-required transitions)"
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Reason for activation (min 10 characters)",
    )


class ActivateRuleResponse(BaseModel):
    """Response schema for activating a rule."""

    rule: DraftRuleResponse = Field(..., description="Activated rule")
    activated_at: str = Field(..., description="Activation timestamp (ISO format)")


class DisableRuleRequest(BaseModel):
    """Request schema for disabling a rule."""

    actor: str = Field(..., description="Who is disabling this rule")
    reason: str = Field(
        default="",
        description="Optional reason for disabling",
    )


class DisableRuleResponse(BaseModel):
    """Response schema for disabling a rule."""

    rule: DraftRuleResponse = Field(..., description="Disabled rule")
    disabled_at: str = Field(..., description="Disable timestamp (ISO format)")


class ShadowRuleRequest(BaseModel):
    """Request schema for moving a rule to shadow mode."""

    actor: str = Field(..., description="Who is moving this rule to shadow")
    reason: str = Field(
        default="",
        description="Optional reason for shadow mode",
    )


class ShadowRuleResponse(BaseModel):
    """Response schema for moving a rule to shadow mode."""

    rule: DraftRuleResponse = Field(..., description="Rule in shadow mode")
    shadowed_at: str = Field(..., description="Shadow timestamp (ISO format)")


# =============================================================================
# Version and Rollback Schemas
# =============================================================================


class RuleVersionResponse(BaseModel):
    """Response schema for a rule version."""

    rule_id: str = Field(..., description="Rule identifier")
    version_id: str = Field(..., description="Version identifier")
    rule: DraftRuleResponse = Field(..., description="Rule at this version")
    timestamp: str = Field(..., description="Version timestamp (ISO format)")
    created_by: str = Field(..., description="Who created this version")
    reason: str = Field(default="", description="Reason for this version")


class RuleVersionListResponse(BaseModel):
    """Response schema for listing rule versions."""

    versions: list[RuleVersionResponse] = Field(
        ..., description="List of versions (oldest first)"
    )
    total: int = Field(..., description="Total number of versions")


class RollbackRuleRequest(BaseModel):
    """Request schema for rolling back a rule."""

    actor: str = Field(..., description="Who is performing the rollback")
    reason: str = Field(
        default="",
        description="Optional reason for rollback",
    )


class RollbackRuleResponse(BaseModel):
    """Response schema for rolling back a rule."""

    rule: DraftRuleResponse = Field(..., description="Rolled back rule")
    version_id: str = Field(..., description="New version ID created by rollback")
    rolled_back_to: str = Field(..., description="Version ID rolled back to")
    rolled_back_at: str = Field(..., description="Rollback timestamp (ISO format)")


# =============================================================================
# Audit Log Schemas
# =============================================================================


class AuditRecordResponse(BaseModel):
    """Response schema for an audit record."""

    rule_id: str = Field(..., description="Rule identifier")
    action: str = Field(..., description="Action type")
    actor: str = Field(..., description="Who performed the action")
    timestamp: str = Field(..., description="Timestamp (ISO format)")
    before_state: dict[str, Any] | None = Field(None, description="State before change")
    after_state: dict[str, Any] | None = Field(None, description="State after change")
    reason: str = Field(default="", description="Reason for the action")


class AuditLogQueryResponse(BaseModel):
    """Response schema for audit log query."""

    records: list[AuditRecordResponse] = Field(
        ..., description="Matching audit records"
    )
    total: int = Field(..., description="Total number of matching records")
