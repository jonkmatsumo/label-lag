import { apiClient } from './client';

export interface DatasetOverview {
  total_records: number;
  fraud_records: number;
  fraud_rate: number;
  unique_users: number;
  min_transaction_timestamp: string | null;
  max_transaction_timestamp: string | null;
}

export interface SchemaColumn {
  table_name: string;
  column_name: string;
  data_type: string;
  is_nullable: string;
  ordinal_position: number;
}

export interface GenerateDataResponse {
  success: boolean;
  total_records: number;
  fraud_records: number;
  features_materialized: number;
  error?: string;
}

export interface FeatureSample {
  record_id: string;
  is_fraudulent: boolean;
  [key: string]: number | string | boolean;
}

export const datasetApi = {
  getOverview: () => apiClient.get<DatasetOverview>('/bff/v1/dataset/overview'),
  
  getSchema: () => apiClient.get<{ columns: SchemaColumn[] }>('/bff/v1/dataset/schema'),
  
  generateData: (params: { num_users: number; fraud_rate: number; drop_existing: boolean }) =>
    apiClient.post<GenerateDataResponse>('/bff/v1/dataset/generate', params),
    
  clearData: () => apiClient.delete<{ success: boolean; tables_cleared: string[] }>('/bff/v1/dataset/clear'),
  
  getFeatureSample: (sampleSize: number = 1000, stratify: boolean = true) =>
    apiClient.get<{ samples: FeatureSample[] }>(`/bff/v1/dataset/sample?sample_size=${sampleSize}&stratify=${stratify}`),
};
