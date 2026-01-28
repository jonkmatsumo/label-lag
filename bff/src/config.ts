/**
 * BFF Configuration
 * All values from environment variables with sensible defaults
 */
export interface Config {
  port: number;
  host: string;
  fastApiBaseUrl: string;
  mlflowTrackingUri: string;
  inferenceMode: 'fastapi' | 'gateway';
  gatewayBaseUrl: string;
  requestTimeout: number;
  upstreamTimeout: number;
  logLevel: string;
  testMode: boolean;
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
  const inferenceMode = getEnvOrDefault('BFF_INFERENCE_MODE', 'fastapi');
  if (inferenceMode !== 'fastapi' && inferenceMode !== 'gateway') {
    throw new Error(`Invalid BFF_INFERENCE_MODE: ${inferenceMode}. Must be 'fastapi' or 'gateway'`);
  }

  return {
    port: getEnvAsInt('BFF_PORT', 3000),
    host: getEnvOrDefault('BFF_HOST', '0.0.0.0'),
    fastApiBaseUrl: getEnvOrDefault('BFF_FASTAPI_BASE_URL', 'http://api:8000'),
    mlflowTrackingUri: getEnvOrDefault('BFF_MLFLOW_TRACKING_URI', 'http://mlflow:5005'),
    inferenceMode: inferenceMode as 'fastapi' | 'gateway',
    gatewayBaseUrl: getEnvOrDefault('BFF_GATEWAY_BASE_URL', 'http://inference-gateway:8081'),
    requestTimeout: getEnvAsInt('BFF_REQUEST_TIMEOUT', 30000),
    upstreamTimeout: getEnvAsInt('BFF_UPSTREAM_TIMEOUT_MS', 5000),
    logLevel: getEnvOrDefault('BFF_LOG_LEVEL', 'info'),
    testMode: getEnvAsBool('BFF_TEST_MODE', false),
  };
}
