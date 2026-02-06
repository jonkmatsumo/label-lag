/**
 * BFF API endpoints
 */
import { apiClient } from './client';
import type {
  HealthResponse,
  SignalRequest,
  SignalResponse,
  TrainRequest,
  TrainResponse,
  DeployRequest,
  DeployResponse,
  DraftRulesResponse,
  PublishRuleResponse,
  SandboxEvaluateRequest,
  SandboxEvaluateResponse,
  BacktestCompareRequest,
  BacktestCompareResponse,
  AnalyticsOverviewResponse,
  DailyStatsResponse,
  RecentAlertsResponse,
  DriftStatusResponse,
  ShadowComparisonResponse,
  BacktestResultsListResponse,
  RuleAnalyticsResponse,
  ReadinessReportResponse,
  RuleVersionListResponse,
  RuleDiffResponse,
  RuleAttributionResponse,
  ApprovalSignalsResponse,
  TransactionSearchRequest,
  TransactionSearchResponse,
  FeatureSampleResponse,
} from '../types/api';

// Health endpoints
export const healthApi = {
  getHealth: () => apiClient.get<HealthResponse>('/bff/v1/health'),
};

// Signal evaluation
export const signalApi = {
  evaluate: (request: SignalRequest) =>
    apiClient.post<SignalResponse>('/bff/v1/evaluate/signal', request),
};

// Model training and deployment
export const modelApi = {
  train: (request: TrainRequest) =>
    apiClient.post<TrainResponse>('/bff/v1/train', request),
  deploy: (request: DeployRequest) =>
    apiClient.post<DeployResponse>('/bff/v1/models/deploy', request),
};

export const datasetApi = {
  getOverview: () => apiClient.get<AnalyticsOverviewResponse>('/bff/v1/dataset/overview'),
  getSchema: () => apiClient.get<{ columns: string[]; types: Record<string, string> }>('/bff/v1/dataset/schema'),
  getSample: (params?: { sample_size?: number; stratify?: boolean }) => {
    const searchParams = new URLSearchParams();
    if (params?.sample_size) searchParams.set('sample_size', String(params.sample_size));
    if (params?.stratify !== undefined) searchParams.set('stratify', String(params.stratify));
    const query = searchParams.toString();
    return apiClient.get<FeatureSampleResponse>(`/bff/v1/dataset/sample${query ? `?${query}` : ''}`);
  },
};

// Rules management
export const rulesApi = {
  getDraftRules: () =>
    apiClient.get<DraftRulesResponse>('/bff/v1/rules/draft'),
  publishRule: (ruleId: string, data: { actor: string; reason: string }) =>
    apiClient.post<PublishRuleResponse>(`/bff/v1/rules/${ruleId}/publish`, data),
  sandboxEvaluate: (request: SandboxEvaluateRequest) =>
    apiClient.post<SandboxEvaluateResponse>(
      '/bff/v1/rules/sandbox/evaluate',
      request
    ),
  getReadiness: (ruleId: string) =>
    apiClient.get<ReadinessReportResponse>(`/bff/v1/rules/${encodeURIComponent(ruleId)}/readiness`),
  getApprovalSignals: (ruleId: string) =>
    apiClient.get<ApprovalSignalsResponse>(`/bff/v1/rules/draft/${encodeURIComponent(ruleId)}/signals`),
  getVersions: (ruleId: string) =>
    apiClient.get<RuleVersionListResponse>(`/bff/v1/rules/${encodeURIComponent(ruleId)}/versions`),
  getDiff: (ruleId: string, versionA: string, versionB: string) =>
    apiClient.get<RuleDiffResponse>(`/bff/v1/rules/${encodeURIComponent(ruleId)}/diff?version_a=${versionA}&version_b=${versionB}`),
};

export const suggestionsApi = {
  getHeuristic: (params?: { field?: string; min_confidence?: number }) => {
    const searchParams = new URLSearchParams();
    if (params?.field) searchParams.set('field', params.field);
    if (params?.min_confidence) searchParams.set('min_confidence', String(params.min_confidence));
    return apiClient.get<{ suggestions: any[]; total: number }>(`/bff/v1/suggestions/heuristic?${searchParams.toString()}`);
  },
  accept: (data: any) => apiClient.post('/bff/v1/suggestions/accept', data),
};

// Backtest / What-if
export const backtestApi = {
  compare: (request: BacktestCompareRequest) =>
    apiClient.post<BacktestCompareResponse>('/bff/v1/backtest/compare', request),
  listResults: (params?: { rule_id?: string; start_date?: string; end_date?: string; limit?: number }) => {
    const searchParams = new URLSearchParams();
    if (params?.rule_id) searchParams.set('rule_id', params.rule_id);
    if (params?.start_date) searchParams.set('start_date', params.start_date);
    if (params?.end_date) searchParams.set('end_date', params.end_date);
    if (params?.limit) searchParams.set('limit', String(params.limit));
    const query = searchParams.toString();
    return apiClient.get<BacktestResultsListResponse>(`/bff/v1/backtest/results${query ? `?${query}` : ''}`);
  },
};

// Analytics endpoints
export const analyticsApi = {
  getOverview: () =>
    apiClient.get<AnalyticsOverviewResponse>('/bff/v1/analytics/overview'),
  getDailyStats: (days = 30) =>
    apiClient.get<DailyStatsResponse>(`/bff/v1/analytics/daily-stats?days=${days}`),
  getRecentAlerts: (limit = 50) =>
    apiClient.get<RecentAlertsResponse>(`/bff/v1/analytics/recent-alerts?limit=${limit}`),
  getRuleAnalytics: (ruleId: string, days = 7) =>
    apiClient.get<RuleAnalyticsResponse>(`/bff/v1/analytics/rules/${encodeURIComponent(ruleId)}?days=${days}`),
  getAttribution: (ruleId: string, days = 7) =>
    apiClient.get<RuleAttributionResponse>(`/bff/v1/analytics/attribution?rule_id=${encodeURIComponent(ruleId)}&days=${days}`),
  searchTransactions: (request: TransactionSearchRequest) =>
    apiClient.post<TransactionSearchResponse>('/bff/v1/analytics/transactions/search', request),
};

// Monitoring endpoints
export const monitoringApi = {
  getDrift: (params?: { hours?: number; threshold?: number; force_refresh?: boolean }) => {
    const searchParams = new URLSearchParams();
    if (params?.hours) searchParams.set('hours', String(params.hours));
    if (params?.threshold) searchParams.set('threshold', String(params.threshold));
    if (params?.force_refresh) searchParams.set('force_refresh', String(params.force_refresh));
    const query = searchParams.toString();
    return apiClient.get<DriftStatusResponse>(`/bff/v1/monitoring/drift${query ? `?${query}` : ''}`);
  },
  getShadowComparison: (startDate: string, endDate: string, ruleIds?: string) => {
    const searchParams = new URLSearchParams({ start_date: startDate, end_date: endDate });
    if (ruleIds) searchParams.set('rule_ids', ruleIds);
    return apiClient.get<ShadowComparisonResponse>(`/bff/v1/metrics/shadow/comparison?${searchParams.toString()}`);
  },
};

// Rules detail endpoints
export const rulesDetailApi = {
  getReadiness: (ruleId: string) =>
    apiClient.get<ReadinessReportResponse>(`/bff/v1/rules/${encodeURIComponent(ruleId)}/readiness`),
  getVersions: (ruleId: string) =>
    apiClient.get<RuleVersionListResponse>(`/bff/v1/rules/${encodeURIComponent(ruleId)}/versions`),
};
