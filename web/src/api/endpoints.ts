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
  OverviewMetrics,
  DailyStatsResponse,
  AlertsResponse,
  DriftStatusResponse,
  ShadowComparisonResponse,
  BacktestResultsListResponse,
  RuleAnalyticsResponse,
  ReadinessReportResponse,
  RuleVersionListResponse,
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

// Rules management
export const rulesApi = {
  getDraftRules: () =>
    apiClient.get<DraftRulesResponse>('/bff/v1/rules/draft'),
  publishRule: (ruleId: string) =>
    apiClient.post<PublishRuleResponse>(`/bff/v1/rules/${ruleId}/publish`),
  sandboxEvaluate: (request: SandboxEvaluateRequest) =>
    apiClient.post<SandboxEvaluateResponse>(
      '/bff/v1/rules/sandbox/evaluate',
      request
    ),
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
    apiClient.get<OverviewMetrics>('/bff/v1/analytics/overview'),
  getDailyStats: (days = 30) =>
    apiClient.get<DailyStatsResponse>(`/bff/v1/analytics/daily-stats?days=${days}`),
  getRecentAlerts: (limit = 50) =>
    apiClient.get<AlertsResponse>(`/bff/v1/analytics/recent-alerts?limit=${limit}`),
  getRuleAnalytics: (ruleId: string, days = 7) =>
    apiClient.get<RuleAnalyticsResponse>(`/bff/v1/analytics/rules/${encodeURIComponent(ruleId)}?days=${days}`),
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
