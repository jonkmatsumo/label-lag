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
  RuleSuggestion,
  AcceptSuggestionRequest,
  ListJobsResponse,
  Job,
  ListJobEventsResponse,
  CancelJobResponse,
  RetryJobResponse,
  ListDecisionsResponse,
  DecisionDetail,
  DecisionTrace,
  ListTrainingRunsResponse,
  TrainingRun,
  ModelVersion,
  ListModelVersionsResponse,
  ListDatasetProfilesResponse,
  DatasetProfile,
  DatasetSummary,
  CompareProfilesResponse,
  MetricSeriesPoint,
  KpisResponse,
  VolumeSeriesResponse,
  ConfusionMatrixResponse,
  GetRuleImpactResponse,
  GetJobSummaryResponse,
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
    return apiClient.get<{ suggestions: RuleSuggestion[]; total: number }>(`/bff/v1/suggestions/heuristic?${searchParams.toString()}`);
  },
  accept: (data: AcceptSuggestionRequest) => apiClient.post('/bff/v1/suggestions/accept', data),
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
  getOverview: (days?: number) =>
    apiClient.get<AnalyticsOverviewResponse>(`/bff/v1/analytics/overview${days ? `?days=${days}` : ''}`),
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
  getKpis: (params: { start_time: string; end_time: string; group_by?: 'hour' | 'day'; signal?: AbortSignal }) => {
    const { signal, ...rest } = params;
    const searchParams = new URLSearchParams();
    searchParams.set('start_time', rest.start_time);
    searchParams.set('end_time', rest.end_time);
    if (rest.group_by) searchParams.set('group_by', rest.group_by);
    return apiClient.get<KpisResponse>(`/bff/v1/kpis?${searchParams.toString()}`, signal);
  },
  getVolume: (params: { start_time: string; end_time: string; granularity?: 'hour' | 'day'; signal?: AbortSignal }) => {
    const { signal, ...rest } = params;
    const searchParams = new URLSearchParams();
    searchParams.set('start_time', rest.start_time);
    searchParams.set('end_time', rest.end_time);
    if (rest.granularity) searchParams.set('granularity', rest.granularity);
    return apiClient.get<VolumeSeriesResponse>(`/bff/v1/volume?${searchParams.toString()}`, signal);
  },
  getConfusionMatrix: (params: { start_time: string; end_time: string; threshold?: number; model_version?: string; signal?: AbortSignal }) => {
    const { signal, ...rest } = params;
    const searchParams = new URLSearchParams();
    searchParams.set('start_time', rest.start_time);
    searchParams.set('end_time', rest.end_time);
    if (rest.threshold !== undefined) searchParams.set('threshold', String(rest.threshold));
    if (rest.model_version) searchParams.set('model_version', rest.model_version);
    return apiClient.get<ConfusionMatrixResponse>(`/bff/v1/analytics/confusion-matrix?${searchParams.toString()}`, signal);
  },
  getRuleImpact: (ruleId: string, params?: { start_time?: string; end_time?: string; signal?: AbortSignal }) => {
    const { signal, ...queryParams } = params ?? {};
    const searchParams = new URLSearchParams();
    if (queryParams.start_time) searchParams.set('start_time', queryParams.start_time);
    if (queryParams.end_time) searchParams.set('end_time', queryParams.end_time);
    const query = searchParams.toString();
    return apiClient.get<GetRuleImpactResponse>(`/bff/v1/analytics/rules/${encodeURIComponent(ruleId)}/impact${query ? `?${query}` : ''}`, signal);
  },
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
  getSeries: (params: { metric: string; start_date: string; end_date: string; interval?: string; tags?: Record<string, string> }) => {
    const searchParams = new URLSearchParams({
      metric: params.metric,
      start_date: params.start_date,
      end_date: params.end_date,
    });
    if (params.interval) searchParams.set('interval', params.interval);
    if (params.tags) searchParams.set('tags', JSON.stringify(params.tags));
    return apiClient.get<{ series: MetricSeriesPoint[] }>(`/bff/v1/metrics/series?${searchParams.toString()}`);
  },
};

// Rules detail endpoints
export const rulesDetailApi = {
  getReadiness: (ruleId: string) =>
    apiClient.get<ReadinessReportResponse>(`/bff/v1/rules/${encodeURIComponent(ruleId)}/readiness`),
  getVersions: (ruleId: string) =>
    apiClient.get<RuleVersionListResponse>(`/bff/v1/rules/${encodeURIComponent(ruleId)}/versions`),
};

// Jobs endpoints
export const jobsApi = {
  list: (params?: { job_type?: string; status?: string; limit?: number; cursor?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.job_type) searchParams.set('job_type', params.job_type);
    if (params?.status) searchParams.set('status', params.status);
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.cursor) searchParams.set('cursor', params.cursor);
    const query = searchParams.toString();
    return apiClient.get<ListJobsResponse>(`/bff/v1/jobs${query ? `?${query}` : ''}`);
  },
  get: (jobId: string) =>
    apiClient.get<{ job: Job }>(`/bff/v1/jobs/${encodeURIComponent(jobId)}`),
  getEvents: (jobId: string) =>
    apiClient.get<ListJobEventsResponse>(`/bff/v1/jobs/${encodeURIComponent(jobId)}/events`),
  cancel: (jobId: string) =>
    apiClient.post<CancelJobResponse>(`/bff/v1/jobs/${encodeURIComponent(jobId)}/cancel`),
  retry: (jobId: string) =>
    apiClient.post<RetryJobResponse>(`/bff/v1/jobs/${encodeURIComponent(jobId)}/retry`),
  getSummary: (params?: { start_time?: string; end_time?: string; signal?: AbortSignal }) => {
    const { signal, ...queryParams } = params ?? {};
    const searchParams = new URLSearchParams();
    if (queryParams.start_time) searchParams.set('start_time', queryParams.start_time);
    if (queryParams.end_time) searchParams.set('end_time', queryParams.end_time);
    const query = searchParams.toString();
    return apiClient.get<GetJobSummaryResponse>(`/bff/v1/jobs/summary${query ? `?${query}` : ''}`, signal);
  },
};

// Decisions endpoints
export const decisionsApi = {
  list: (params?: { user_id?: string; decision?: string; start_time?: string; end_time?: string; limit?: number; cursor?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.user_id) searchParams.set('user_id', params.user_id);
    if (params?.decision) searchParams.set('decision', params.decision);
    if (params?.start_time) searchParams.set('start_time', params.start_time);
    if (params?.end_time) searchParams.set('end_time', params.end_time);
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.cursor) searchParams.set('cursor', params.cursor);
    const query = searchParams.toString();
    return apiClient.get<ListDecisionsResponse>(`/bff/v1/decisions${query ? `?${query}` : ''}`);
  },
  get: (id: string) => apiClient.get<DecisionDetail>(`/bff/v1/decisions/${encodeURIComponent(id)}`),
  getTrace: (id: string) => apiClient.get<DecisionTrace>(`/bff/v1/decisions/${encodeURIComponent(id)}/trace`),
};

// Training endpoints
export const trainingApi = {
  list: (params?: { model_name?: string; status?: string; limit?: number; cursor?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.model_name) searchParams.set('model_name', params.model_name);
    if (params?.status) searchParams.set('status', params.status);
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.cursor) searchParams.set('cursor', params.cursor);
    const query = searchParams.toString();
    return apiClient.get<ListTrainingRunsResponse>(`/bff/v1/training-runs${query ? `?${query}` : ''}`);
  },
  get: (id: string) => apiClient.get<TrainingRun>(`/bff/v1/training-runs/${encodeURIComponent(id)}`),
};

// Model versions endpoints
export const modelVersionsApi = {
  list: (params?: { model_name?: string; limit?: number; cursor?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.model_name) searchParams.set('model_name', params.model_name);
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.cursor) searchParams.set('cursor', params.cursor);
    const query = searchParams.toString();
    return apiClient.get<ListModelVersionsResponse>(`/bff/v1/models/versions${query ? `?${query}` : ''}`);
  },
  get: (version: string) => apiClient.get<ModelVersion>(`/bff/v1/models/versions/${encodeURIComponent(version)}`),
};

// Profiles endpoints
export const profilesApi = {
  list: (params?: { limit?: number; cursor?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.cursor) searchParams.set('cursor', params.cursor);
    const query = searchParams.toString();
    return apiClient.get<ListDatasetProfilesResponse>(`/bff/v1/dataset/profiles${query ? `?${query}` : ''}`);
  },
  get: (id: string) => apiClient.get<DatasetProfile>(`/bff/v1/dataset/profiles/${encodeURIComponent(id)}`),
  getSummary: (profileId?: string) => {
    const searchParams = new URLSearchParams();
    if (profileId) searchParams.set('profile_id', profileId);
    const query = searchParams.toString();
    return apiClient.get<DatasetSummary>(`/bff/v1/dataset/summary${query ? `?${query}` : ''}`);
  },
  compare: (baseId: string, targetId: string) => {
    const searchParams = new URLSearchParams({ base_id: baseId, target_id: targetId });
    return apiClient.get<CompareProfilesResponse>(`/bff/v1/dataset/profiles/compare?${searchParams.toString()}`);
  },
};
