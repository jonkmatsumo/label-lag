/**
 * BFF Configuration
 * All values from environment variables with sensible defaults
 */
export interface Config {
  port: number;
  host: string;
  pythonApiBaseUrl: string;
  mlflowTrackingUri: string;
  gatewayBaseUrl: string;
  requestTimeout: number;
  upstreamTimeout: number;
  cacheEnabled: boolean;
  cacheTtlMs: number;
  corsOrigin: string;
  logLevel: string;
  testMode: boolean;
  // Feature flags for Go migration
  enableGoDatasetClear: boolean;
  enableGoRulesSandbox: boolean;
  shadowModeEnabled: boolean;
}

function getEnvOrDefault(key: string, defaultValue: string): string {
  return process.env[key] ?? defaultValue;
}

function getEnvAsInt(key: string, defaultValue: number): number {
  const value = process.env[key];
  if (value === undefined) return defaultValue;
  const parsed = parseInt(value, 10);
  return isNaN(parsed) ? defaultValue : parsed;
}

function getEnvAsBool(key: string, defaultValue: boolean): boolean {
  const value = process.env[key];
  if (value === undefined) return defaultValue;
  return value.toLowerCase() === 'true' || value === '1';
}

export function loadConfig(): Config {
  return {
    port: getEnvAsInt('BFF_PORT', 3210),
    host: getEnvOrDefault('BFF_HOST', '0.0.0.0'),
    pythonApiBaseUrl: getEnvOrDefault('BFF_PYTHON_API_BASE_URL', 'http://api:8000'),
    mlflowTrackingUri: getEnvOrDefault('BFF_MLFLOW_TRACKING_URI', 'http://mlflow:5005'),
    gatewayBaseUrl: getEnvOrDefault('BFF_GATEWAY_BASE_URL', 'http://inference-gateway:8081'),
    requestTimeout: getEnvAsInt('BFF_REQUEST_TIMEOUT', 30000),
    upstreamTimeout: getEnvAsInt('BFF_UPSTREAM_TIMEOUT_MS', 5000),
    cacheEnabled: getEnvAsBool('BFF_CACHE_ENABLED', true),
    cacheTtlMs: getEnvAsInt('BFF_CACHE_TTL_MS', 30000),
    corsOrigin: getEnvOrDefault('BFF_CORS_ORIGIN', 'true'), // 'true' means reflect origin (dev), or comma-separated list
    logLevel: getEnvOrDefault('BFF_LOG_LEVEL', 'info'),
    testMode: getEnvAsBool('BFF_TEST_MODE', false),
    enableGoDatasetClear: getEnvAsBool('ENABLE_GO_DATASET_CLEAR', true),
    enableGoRulesSandbox: getEnvAsBool('ENABLE_GO_RULES_SANDBOX', true),
    shadowModeEnabled: getEnvAsBool('SHADOW_MODE_ENABLED', true),
  };
}
