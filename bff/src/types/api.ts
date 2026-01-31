/**
 * Shared API types for BFF
 */

// Standard error envelope for all API responses
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
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
  name: string;
  score: number;
  weight: number;
}

export interface MatchedRule {
  rule_id: string;
  name: string;
  action: string;
  score_adjustment?: number;
  explanation?: string;
}

export interface SignalResponse {
  request_id: string;
  score: number;
  risk_label: 'LOW' | 'MEDIUM' | 'HIGH';
  latency_ms: number;
  risk_components: RiskComponent[];
  model_version: string;
  ruleset_version?: string;
  matched_rules: MatchedRule[];
  shadow_matched_rules?: MatchedRule[];
  debug?: Record<string, unknown>;
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
  selected_columns?: string[];
  max_depth?: number;
  learning_rate?: number;
  n_estimators?: number;
  tuning_config?: TuningConfig;
}

export interface TrainResponse {
  run_id: string;
  status: string;
  message: string;
  metrics?: Record<string, number>;
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
  total_transactions: number;
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
  total_users: number;
  total_transactions: number;
  fraud_rate: number;
  fraud_count: number;
  unique_merchants: number;
  min_timestamp: string;
  max_timestamp: string;
}

export interface DailyStat {
  date: string;
  total_transactions: number;
  fraud_transactions: number;
  fraud_rate: number;
  total_amount: number;
}

export interface DailyStatsResponse {
  stats: DailyStat[];
  period_days: number;
}

export interface TransactionDetail {
  id: string;
  user_id: string;
  amount: number;
  timestamp: string;
  is_fraud: boolean;
  score?: number;
  merchant_category?: string;
}

export interface TransactionDetailsResponse {
  transactions: TransactionDetail[];
  total: number;
}

export interface RecentAlert {
  id: string;
  user_id: string;
  amount: number;
  score: number;
  created_at: string;
  matched_rules: string[];
}

export interface RecentAlertsResponse {
  alerts: RecentAlert[];
  total: number;
}

export interface DatasetFingerprintResponse {
  fingerprint: string;
  record_count: number;
  created_at: string;
  feature_stats: Record<string, {
    mean: number;
    std: number;
    min: number;
    max: number;
  }>;
}

export interface FeatureSample {
  record_id: string;
  is_fraudulent: boolean;
  [key: string]: unknown;
}

export interface FeatureSampleResponse {
  samples: FeatureSample[];
  total_sampled: number;
  stratified: boolean;
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