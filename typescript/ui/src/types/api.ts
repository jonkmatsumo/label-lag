/**
 * API types shared with BFF
 */

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

// Health check response
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

// Signal evaluation types
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

export interface MatchedRule {
  rule_id: string;
  severity: string;
  reason: string;
  explanation?: string;
  name?: string;
  action?: string;
  score_adjustment?: number;
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

// Training types
export interface TuningConfig {
  enabled: boolean;
  n_trials?: number;
  timeout_minutes?: number;
  metric?: 'pr_auc' | 'roc_auc' | 'f1';
}

export interface TrainRequest {
  name?: string;
  test_size?: number;
  random_seed?: number;
  selected_feature_columns?: string[];
  training_window_days?: number;
  max_depth?: number;
  learning_rate?: number;
  n_estimators?: number;
  tuning_config?: TuningConfig;
}

export interface TrainResponse {
  run_id?: string;
  status: string;
  message?: string;
  metrics?: Record<string, number>;
  success?: boolean;
  error?: string;
}

// Model deployment types
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
}

// Rule types
export interface DraftRule {
  id: string;
  name: string;
  description: string;
  condition: string;
  action: string;
  score_adjustment?: number;
  status: 'draft' | 'pending_approval' | 'approved' | 'rejected' | 'published';
  created_at: string;
  updated_at: string;
  created_by?: string;
}

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

export interface DraftRulesResponse {
  rules: DraftRule[];
  total: number;
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

// Sandbox evaluation types
export interface SandboxEvaluateRequest {
  base_score: number;
  features: Record<string, unknown>;
  rule_ids?: string[];
  custom_ruleset?: unknown;
}

export interface SandboxEvaluateResponse {
  final_score: number;
  risk_label: 'LOW' | 'MEDIUM' | 'HIGH';
  matched_rules: MatchedRule[];
  shadow_matched_rules: MatchedRule[];
  evaluation_details: Record<string, unknown>;
}

// Backtest comparison types
export interface BacktestCompareRequest {
  base_version: string;
  candidate_version: string;
  start_date: string;
  end_date: string;
  rule_id?: string;
}

export interface BacktestMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  total_records: number;
  flagged_transactions: number;
  true_positives: number;
  false_positives: number;
  match_rate: number;
  rejected_count: number;
}

export interface BacktestCompareResponse {
  base: BacktestMetrics;
  candidate: BacktestMetrics;
  delta: {
    precision: number;
    recall: number;
    f1_score: number;
    flagged_rate_change: number;
    match_rate_delta: number;
    rejected_count_delta: number;
  };
  job_id?: string;
}

// Analytics types
export interface AnalyticsOverviewResponse {
  total_records: number;
  fraud_records: number;
  fraud_rate: number;
  unique_users: number;
  min_transaction_timestamp: string;
  max_transaction_timestamp: string;
  min_created_at: string;
  max_created_at: string;
  total_amount: number;
  fraud_amount: number;
}

export interface DailyStat {
  date: string;
  total_transactions: number;
  fraud_count: number;
  fraud_rate: number;
  total_amount: number;
  avg_z_score: number;
}

export interface DailyStatsResponse {
  stats: DailyStat[];
}

export interface TransactionDetail {
  record_id: string;
  user_id: string;
  created_at: string;
  is_train_eligible: boolean;
  is_pre_fraud: boolean;
  amount: number;
  is_fraudulent: boolean;
  fraud_type: string;
  is_off_hours_txn: boolean;
  merchant_risk_score: number;
  velocity_24h: number;
  amount_to_avg_ratio_30d: number;
  balance_volatility_z_score: number;
}

export interface TransactionDetailsResponse {
  transactions: TransactionDetail[];
}

export interface TransactionSearchRequest {
  user_id?: string;
  transaction_id?: string;
  min_amount?: number;
  max_amount?: number;
  start_date?: string;
  end_date?: string;
  is_fraudulent?: boolean;
  min_score?: number;
  max_score?: number;
  limit?: number;
  offset?: number;
}

export interface TransactionSearchResponse {
  transactions: TransactionDetail[];
  total: number;
}

export interface RecentAlert {
  record_id: string;
  user_id: string;
  created_at: string;
  amount: number;
  is_fraudulent: boolean;
  fraud_type: string;
  merchant_risk_score: number;
  velocity_24h: number;
  amount_to_avg_ratio_30d: number;
  balance_volatility_z_score: number;
  computed_risk_score: number;
}

export interface RecentAlertsResponse {
  alerts: RecentAlert[];
}

export interface TableFingerprint {
  count: number;
  max_created_at: string;
  max_timestamp: string;
  max_id: number;
}

export interface DatasetFingerprintResponse {
  generated_records: TableFingerprint;
  feature_snapshots: TableFingerprint;
}

export interface FeatureSample {
  record_id: string;
  is_fraudulent: boolean;
  velocity_24h: number;
  amount_to_avg_ratio_30d: number;
  balance_volatility_z_score: number;
}

export interface FeatureSampleResponse {
  samples: FeatureSample[];
}

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

export interface RuleHealthMetrics {
  period_start: string;
  period_end: string;
  production_matches: number;
  shadow_matches: number;
  production_only_count: number;
  shadow_only_count: number;
  mean_score_delta: number;
  mean_execution_time_ms: number;
}

export interface RuleHealthResponse {
  rule_id: string;
  status: string;
  reason: string;
  metrics: RuleHealthMetrics;
}

export interface RuleAnalyticsResponse {
  rule_id: string;
  health: RuleHealthResponse;
  statistics: {
    mean_score_delta: number;
    mean_latency_ms: number;
    total_matches: number;
  };
  history_summary: unknown[];
}

export interface RuleAttributionResponse {
  rule_id: string;
  total_matches: number;
  mean_model_score: number;
  mean_final_score: number;
  mean_impact: number;
  net_impact: number;
}

// Monitoring types
export interface FeatureDriftDetail {
  feature: string;
  psi: number;
  status: 'OK' | 'WARN' | 'FAIL';
  reference_mean?: number;
  live_mean?: number;
}

export interface DriftStatusResponse {
  status: 'ok' | 'warn' | 'fail' | 'error';
  message: string;
  drift_detected: boolean;
  cached: boolean;
  computed_at?: string;
  hours_analyzed?: number;
  live_size: number;
  reference_size: number;
  threshold?: number;
  top_features?: FeatureDriftDetail[];
}

export interface RuleMetricsItem {
  rule_id: string;
  production_matches: number;
  shadow_matches: number;
  overlap_count: number;
  production_only_count: number;
  shadow_only_count: number;
}

export interface ShadowComparisonResponse {
  period_start: string;
  period_end: string;
  rule_metrics: RuleMetricsItem[];
  total_requests: number;
}

export interface BacktestResult {
  job_id: string;
  rule_id: string | null;
  ruleset_version: string;
  created_at: string;
  completed_at?: string;
  status: string;
  metrics?: BacktestMetrics;
  error?: string;
}

export interface BacktestResultsListResponse {
  results: BacktestResult[];
  total: number;
}

// Rules detail types
export interface ReadinessCheck {
  name: string;
  status: 'pass' | 'warn' | 'fail';
  message: string;
}

export interface ReadinessReportResponse {
  rule_id: string;
  timestamp: string;
  overall_status: 'pass' | 'warn' | 'fail';
  checks: ReadinessCheck[];
}

export interface RuleVersionDetail {
  rule_id: string;
  field: string;
  op: string;
  value: unknown;
  action: string;
  score?: number;
  severity: string;
  reason: string;
  status: string;
  created_at?: string;
}

export interface RuleVersionResponse {
  rule_id: string;
  version_id: string;
  rule: RuleVersionDetail;
  timestamp: string;
  created_by: string;
  reason?: string;
}

export interface RuleVersionListResponse {
  versions: RuleVersionResponse[];
  total: number;
}

export interface FieldChange {
  field_name: string;
  change_type: 'modified' | 'unchanged';
  old_value: unknown;
  new_value: unknown;
}

export interface RuleDiffResponse {
  rule_id: string;
  version_a_id: string;
  version_b_id: string;
  changes: FieldChange[];
  is_breaking: boolean;
  version_a_timestamp?: string;
  version_a_created_by?: string;
  version_b_timestamp?: string;
  version_b_created_by?: string;
}

export interface ProductionRule {
  id: string;
  field: string;
  op: string;
  value: unknown;
  action: string;
  score?: number;
  severity: string;
  reason: string;
  status: string;
}

export interface ProductionRulesResponse {
  version: string;
  rules: ProductionRule[];
}

// Suggestion types
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

// Jobs types
export interface Job {
  job_id: string;
  job_type: string;
  status: string;
  created_at: string;
  started_at?: string;
  ended_at?: string;
  error_code?: string;
  error_message?: string;
  params_json?: string;
  metrics_json?: string;
}

export interface JobEvent {
  event_id: number;
  job_id: string;
  event_type: string;
  timestamp: string;
  details_json?: string;
}

export interface CursorPageResponse {
  next_cursor?: string;
  total?: number;
}

export interface ListJobsResponse {
  jobs: Job[];
  pagination: CursorPageResponse;
}

export interface ListJobEventsResponse {
  events: JobEvent[];
}

export interface CancelJobResponse {
  success: boolean;
}

export interface RetryJobResponse {
  new_job_id: string;
}
