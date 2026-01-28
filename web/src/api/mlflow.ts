import { apiClient } from './client';

export interface MlflowExperiment {
  experiment_id: string;
  name: string;
  artifact_location: string;
  lifecycle_stage: string;
}

export interface MlflowRun {
  info: {
    run_id: string;
    experiment_id: string;
    status: string;
    start_time: number;
    end_time: number;
    lifecycle_stage: string;
  };
  data: {
    metrics: Record<string, number>;
    params: Record<string, string>;
    tags: Record<string, string>;
  };
}

export interface MlflowModelVersion {
  name: string;
  version: string;
  creation_timestamp: number;
  last_updated_timestamp: number;
  current_stage: string;
  description: string;
  source: string;
  run_id: string;
  status: string;
}

export interface CvMetricsArtifact {
  [metric: string]: number[];
}

export interface TuningTrial {
  trial: number;
  value: number;
  state: string;
  params: Record<string, number | string>;
}

export interface SplitManifest {
  train_size: number;
  test_size: number;
  train_fraud_rate: number;
  test_fraud_rate: number;
  strategy: string;
  seed: number;
  training_cutoff_date: string;
}

export const mlflowApi = {
  searchExperiments: (filter?: string) => 
    apiClient.get<{ experiments: MlflowExperiment[] }>(`/bff/v1/mlflow/experiments/search?filter=${encodeURIComponent(filter || '')}`),

  searchRuns: (experimentIds: string[], filter?: string) =>
    apiClient.post<{ runs: MlflowRun[] }>('/bff/v1/mlflow/runs/search', {
      experiment_ids: experimentIds,
      filter: filter,
      max_results: 50,
      order_by: ['start_time DESC']
    }),

  searchModelVersions: (filter: string = "name='ach-fraud-detection'") =>
    apiClient.get<{ model_versions: MlflowModelVersion[] }>(`/bff/v1/mlflow/model-versions/search?filter=${encodeURIComponent(filter)}`),

  getArtifact: <T>(runId: string, path: string) =>
    apiClient.get<T>(`/bff/v1/mlflow/runs/${runId}/artifacts?path=${encodeURIComponent(path)}`),
};
