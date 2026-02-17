/**
 * API types shared with BFF
 * Re-exported from generated protos where available.
 */

// Analytics Service
export type {
  GetKpisRequest,
  GetKpisResponse,
  KpiBucket,
  GetVolumeSeriesRequest,
  GetVolumeSeriesResponse,
  VolumePoint,
  GetConfusionMatrixRequest,
  GetConfusionMatrixResponse,
  GetDatasetProfileRequest,
  GetDatasetProfileResponse,
  Bucket,
  FeatureProfile,
  ValueCount,
  GetDailyStatsRequest,
  GetDailyStatsResponse,
  DailyStat,
  GetTransactionDetailsRequest,
  GetTransactionDetailsResponse,
  TransactionDetail,
  SearchTransactionsRequest,
  SearchTransactionsResponse,
  GetRecentAlertsRequest,
  GetRecentAlertsResponse,
  Alert as RecentAlert,
  GetOverviewMetricsRequest,
  GetOverviewMetricsResponse,
  GetFeatureSampleRequest,
  GetFeatureSampleResponse,
  FeatureSample,
  GetTrainingDataRequest,
  GetTrainingDataResponse,
  GetBacktestFeaturesRequest,
  GetBacktestFeaturesResponse,
  BacktestFeatureVector,
  BacktestMetrics,
  BacktestResult,
  SaveBacktestResultRequest,
  SaveBacktestResultResponse,
  ListDatasetProfilesRequest,
  ListBacktestResultsRequest,
  ListBacktestResultsResponse,
  GetBacktestResultRequest,
  GetBacktestResultResponse,
  GetDriftWindowRequest,
  GetDriftWindowResponse,
  GetInferenceScoresRequest,
  GetInferenceScoresResponse,
  StoreGeneratedDataRequest,
  StoreGeneratedDataResponse,
  GeneratedRecord,
  EvaluationMetadata,
  ClearAllDataRequest,
  ClearAllDataResponse,
  MaterializeFeaturesRequest,
  MaterializeFeaturesResponse,
  GenerateDataRequest,
  GenerateDataResponse,
  Rule,
  SaveRuleRequest,
  SaveRuleResponse,
  GetRuleRequest,
  GetRuleResponse,
  ListRulesRequest,
  ListRulesResponse,
  DeleteRuleRequest,
  DeleteRuleResponse,
  RuleImpact,
  InferenceEvent,
  DecisionThresholds,
  LogInferenceEventRequest,
  LogInferenceEventResponse,
  ListRuleVersionsRequest,
  ListRuleVersionsResponse,
  GetRuleVersionRequest,
  GetRuleVersionResponse,
  PublishRuleVersionRequest,
  PublishRuleVersionResponse,
  GetRuleReadinessRequest,
  GetRuleReadinessResponse,
  ReadinessCheck,
  DiffRuleVersionsRequest,
  DiffRuleVersionsResponse,
  RuleStats,
  GetRuleStatsRequest,
  GetRuleStatsResponse,
  GetAttributionRequest,
  GetAttributionResponse,
  DailyAttribution,
  ListDecisionsRequest,
  ListDecisionsResponse,
  DecisionSummary,
  GetDecisionRequest,
  GetDecisionResponse,
  GetDecisionTraceRequest,
  GetDecisionTraceResponse,
  GetRuleImpactRequest,
  GetRuleImpactResponse,
  RuleImpactBucket,
  UserFeatures,
  GetLatestUserFeaturesRequest,
  GetLatestUserFeaturesResponse,
  BatchGetLatestUserFeaturesRequest,
  BatchGetLatestUserFeaturesResponse,
  CompareBacktestsRequest,
  CompareBacktestsResponse,
  BacktestMetricsDelta,
  GetShadowComparisonRequest,
  GetShadowComparisonResponse,
  ShadowModeMetrics,
  Job,
  JobEvent,
  ListJobsRequest,
  ListJobsResponse,
  GetJobRequest,
  GetJobResponse,
  CancelJobRequest,
  CancelJobResponse,
  RetryJobRequest,
  RetryJobResponse,
  GetJobEventsRequest,
  GetJobEventsResponse,
  GetJobSummaryRequest,
  GetJobSummaryResponse,
  DatasetProfile as AnalyticsDatasetProfile,
  GetDatasetSummaryRequest,
  GetDatasetSummaryResponse,
  ListDatasetProfilesResponse as AnalyticsListDatasetProfilesResponse,
  GetLatestDatasetProfileRequest,
  GetLatestDatasetProfileResponse,
  CompareDatasetProfilesRequest,
  CompareDatasetProfilesResponse,
  FeatureDrift,
  TrainingRun,
  ListTrainingRunsRequest,
  ListTrainingRunsResponse,
  ListModelVersionsRequest,
  ListModelVersionsResponse,
  GetTrainingRunRequest,
  GetTrainingRunResponse,
  MetricPoint,
  GetMetricSeriesRequest,
  GetMetricSeriesResponse,
  ReportTrainingRunRequest,
  ReportTrainingRunResponse
} from './generated/analytics/v1/analytics';

// Common types
export type {
  CursorPageRequest,
  CursorPageResponse
} from './generated/common/v1/pagination';

// Inference Service
export type {
  ScoreRequest,
  ScoreResponse,
  ErrorDetail,
  InferenceService
} from './generated/inference/v1/inference';

// Gateway Service
export type {
  EvaluateRulesRequest,
  EvaluateRulesResponse,
  EvaluateRulesDiffRequest,
  EvaluateRulesDiffResponse,
  RulesDiffSummary,
  RuleSet,
  Explanation,
  Conflict,
  ReloadRulesRequest,
  ReloadRulesResponse,
  ExplainEvaluationRequest,
  ExplainEvaluationResponse,
  RuleTrace,
  ConditionTrace,
  GatewayService
} from './generated/inference/v1/gateway';

export {
  ReloadRulesRequest_Source
} from './generated/inference/v1/gateway';

// Training Service
export type {
  TrainRequest,
  TuningConfig,
  TrainResponse,
  GetTrainingRunInfoRequest,
  GetTrainingRunInfoResponse,
  GetModelInfoRequest,
  GetModelInfoResponse,
  ValidateTrainRequestResponse,
  ListFeatureSetsRequest,
  ListFeatureSetsResponse,
  StartTuningJobResponse,
  GetTuningStatusRequest,
  TuningJobStatusResponse,
  ListTuningJobsRequest,
  TuningJobSummary,
  ListTuningJobsResponse,
  GetQueueDepthRequest,
  GetQueueDepthResponse,
  PromoteTrialRequest,
  PromoteTrialResponse,
  GetTuningJobInfoRequest,
  TrialResult,
  GetTuningJobInfoResponse,
  ListTrialsRequest,
  TrialRecord,
  ListTrialsResponse,
  CancelTuningJobRequest,
  RequeueTuningJobRequest,
  RequeueTuningJobResponse,
  FinalizeTuningJobRequest,
  FinalizeTuningJobResponse,
  GetHealthRequest,
  GetHealthResponse as TrainingGetHealthResponse,
  TrainingService
} from './generated/training/v1/training';

// Standard error envelope for all API responses
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  upstream_status?: number;
  request_id?: string;
}

export interface ErrorResponse {
  error: ApiError;
}

// Health check response (not in protos yet)
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version?: string;
  model_loaded?: boolean;
  model_version?: string;
  uptime_seconds?: number;
  services?: Record<string, {
    status: string;
    latency_ms?: number;
  }>;
}

import type {
  Rule,
  TrainingRun,
  GetDailyStatsResponse,
  CompareBacktestsRequest,
  CompareBacktestsResponse,
  DecisionSummary
} from './generated/analytics/v1/analytics';

// Aliases for compatibility
export type DailyStatsResponse = GetDailyStatsResponse;
export type DraftRule = Rule;
export interface DashboardStats extends DailyStatsResponse { }

// Backtest aliases
export type BacktestCompareRequest = CompareBacktestsRequest;
export type BacktestCompareResponse = CompareBacktestsResponse;

export type Decision = DecisionSummary;

export type ModelVersion = TrainingRun;

export interface ApprovalSignalItem {
  signal_id: string;
  category: string;
  severity: 'info' | 'warning' | 'risk';
  value: unknown;
  label: string;
  description: string;
}
export interface ApprovalSignalsSummary {
  risk_count: number;
  warning_count: number;
  info_count: number;
  has_blockers: boolean;
}
export interface ApprovalSignalsResponse {
  rule_id: string;
  computed_at: string;
  signals: ApprovalSignalItem[];
  summary: ApprovalSignalsSummary;
  partial: boolean;
  unavailable_signals: string[];
}

export interface ApprovalSignal {
  type: string;
  label: string;
  severity: 'info' | 'warning' | 'risk';
  description: string;
  value?: unknown;
}
export interface RuleGovernance {
  actor?: string;
  reason?: string;
  readiness?: {
    status: 'pass' | 'warn' | 'fail';
    checks: Array<{
      name: string;
      status: 'pass' | 'warn' | 'fail';
      message: string;
    }>;
  };
  approval_signals?: ApprovalSignal[];
}
export interface PublishRuleRequest {
  actor: string;
  reason: string;
}
export interface PublishRuleResponse {
  status: string;
  message: string;
  rule_id: string;
  version?: number;
}
// Sandbox types
export interface SandboxEvaluateRequest {
  base_score: number;
  features: Record<string, unknown>;
  rule_ids?: string[];
  custom_ruleset?: unknown;
}
export interface SandboxEvaluateResponse {
  final_score: number;
  risk_label: 'LOW' | 'MEDIUM' | 'HIGH';
  matched_rules: any[];
  shadow_matched_rules: any[];
  evaluation_details: Record<string, unknown>;
}

export interface MatchedRule {
  rule_id: string;
  severity: string;
  reason: string;
  explanation?: string;
  name?: string;
  action?: string;
  score_adjustment?: number;
}

// Correlation types
export interface CorrelationPair {
  feature_a: string;
  feature_b: string;
  value: number;
}
export interface RelationshipMetric extends CorrelationPair {
  metric_type: string;
}
export interface DatasetCorrelationsResponse {
  pearson: CorrelationPair[];
  spearman: CorrelationPair[];
  cramers_v: CorrelationPair[];
  numeric_columns: string[];
  categorical_columns: string[];
}

// Suggestion types - manual
export interface RuleEvidence {
  mean?: number;
  sample_count?: number;
}

export interface RuleSuggestion {
  field: string;
  operator: string;
  threshold: number | string;
  confidence: number;
  reason: string;
  action: string;
  suggested_score: number;
  evidence?: RuleEvidence;
}

export interface AcceptSuggestionRequest {
  suggestion: RuleSuggestion;
  actor: string;
  custom_id: string;
}

// Restoring missing manual types
export interface SignalRequest {
  user_id: string;
  amount: number;
  currency: string;
  client_transaction_id: string;
}

export interface RiskComponent {
  key: string;
  label: string;
}

export interface SignalResponse {
  request_id: string;
  score: number;
  risk_label: 'LOW' | 'MEDIUM' | 'HIGH';
  latency_ms: number;
  risk_components: RiskComponent[];
  model_version: string;
  rules_version?: string;
  matched_rules: MatchedRule[];
  shadow_matched_rules?: MatchedRule[];
  debug?: Record<string, unknown>;
  model_score?: number;
}

export interface DeployRequest {
  model_version?: string;
  run_id?: string;
  actor: string;
  reason: string;
}

export interface DeployResponse {
  status: string;
  message: string;
  model_version?: string;
  success?: boolean;
  error?: string;
  deployed_at?: string;
  previous_version?: string;
  deployed_by?: string;
}
// Note: added deployed_by to DeployResponse based on likely usage

export interface DatasetProfile {
  profile_id: string;
  created_at: string;
  row_count: number;
  column_count: number;
  size_bytes?: number;
  columns_json?: string;
}

export interface DriftStatusResponse {
  status: 'ok' | 'warn' | 'fail';
  cached: boolean;
  hours_analyzed: number;
  live_size: number;
  top_features: Array<{
    feature: string;
    psi: number;
    status: string;
  }>;
}

export interface ListDatasetProfilesResponse {
  profiles: DatasetProfile[];
}

export interface DatasetSummary {
  columns: Record<string, {
    type: string;
    null_count: number;
    distinct_count: number;
  }>;
}

export interface CompareProfilesResponse {
  features: Array<{
    feature: string;
    psi: number;
    severity: 'low' | 'medium' | 'high';
  }>;
}

export interface MetricSeriesPoint {
  timestamp: string;
  value: number;
}

export type TransactionSearchRequest = SearchTransactionsRequest;
