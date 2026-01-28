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
};
