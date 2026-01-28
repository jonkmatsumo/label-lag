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
}

export interface SignalResponse {
  request_id: string;
  score: number;
  risk_components: RiskComponent[];
  model_version: string;
  matched_rules: MatchedRule[];
  shadow_matched_rules?: MatchedRule[];
}

// Training types
export interface TrainRequest {
  name?: string;
  test_size?: number;
  random_seed?: number;
  selected_columns?: string[];
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
}

export interface DeployResponse {
  status: string;
  message: string;
  model_version?: string;
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
}

export interface SandboxEvaluateResponse {
  final_score: number;
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
}

export interface BacktestCompareResponse {
  base: BacktestMetrics;
  candidate: BacktestMetrics;
  delta: {
    precision: number;
    recall: number;
    f1_score: number;
    flagged_rate_change: number;
  };
  job_id?: string;
}
