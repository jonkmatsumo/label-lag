import { request } from 'undici';
import { Config } from '../config.js';
import { ApiError, ErrorResponse } from '../types/api.js';
import pino from 'pino';

export interface HttpClientOptions {
  config: Config;
  logger: pino.Logger;
}

export interface RequestOptions {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  body?: unknown;
  requestId: string;
  timeout?: number;
  target?: 'fastapi' | 'gateway' | 'mlflow';
  query?: Record<string, string | number | boolean | undefined>;
  authToken?: string;
}

export interface HttpResponse<T> {
  data: T;
  statusCode: number;
  headers: Record<string, string | string[] | undefined>;
}

/**
 * HTTP client for upstream service calls
 * Handles timeouts, error normalization, safe retries, and request ID forwarding
 */
export class HttpClient {
  public readonly config: Config;
  private logger: pino.Logger;

  constructor(options: HttpClientOptions) {
    this.config = options.config;
    this.logger = options.logger.child({ service: 'http-client' });
  }

  private getBaseUrl(target: 'fastapi' | 'gateway' | 'mlflow'): string {
    switch (target) {
      case 'gateway':
        return this.config.gatewayBaseUrl;
      case 'mlflow':
        return this.config.mlflowTrackingUri;
      case 'fastapi':
      default:
        return this.config.fastApiBaseUrl;
    }
  }

  async request<T>(options: RequestOptions): Promise<HttpResponse<T>> {
    const { method } = options;
    const maxRetries = method === 'GET' ? 1 : 0;
    
    let lastError: Error | unknown;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.executeRequest<T>(options, attempt);
      } catch (error) {
        lastError = error;
        const isRetryable = 
          (error instanceof UpstreamError && error.statusCode >= 500) ||
          (error instanceof Error && error.name === 'TimeoutError');
          
        if (attempt < maxRetries && isRetryable) {
          const delay = 100 * (attempt + 1); // simple backoff
          this.logger.warn({ 
            requestId: options.requestId, 
            attempt: attempt + 1, 
            error: error instanceof Error ? error.message : 'Unknown' 
          }, 'Retrying upstream request');
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        throw error;
      }
    }
    throw lastError;
  }

  private async executeRequest<T>(options: RequestOptions, attempt: number): Promise<HttpResponse<T>> {
    const { method, path, body, requestId, timeout, target = 'fastapi', query, authToken } = options;
    const baseUrl = this.getBaseUrl(target);
    
    // Construct URL with query params
    const urlObj = new URL(baseUrl + path);
    if (query) {
      Object.entries(query).forEach(([key, value]) => {
        if (value !== undefined) {
          urlObj.searchParams.append(key, String(value));
        }
      });
    }
    const url = urlObj.toString();

    // Use specific timeout if provided, else upstream timeout (short), else request timeout (long)
    // Actually per requirement: BFF_UPSTREAM_TIMEOUT_MS default (e.g. 5000), allow endpoint overrides.
    const requestTimeout = timeout ?? this.config.upstreamTimeout;

    const startTime = Date.now();
    this.logger.debug({ method, url, requestId, attempt }, 'Upstream request started');

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'X-Request-Id': requestId,
      };

      if (authToken) {
        headers['Authorization'] = authToken;
      }

      const response = await request(url, {
        method,
        headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(requestTimeout),
      });
      const duration = Date.now() - startTime;

      const responseBody = await response.body.text();
      let data: T;

      try {
        data = responseBody ? JSON.parse(responseBody) : ({} as T);
      } catch {
        // If response is not JSON, wrap it
        data = { message: responseBody } as T;
      }

      const responseHeaders: Record<string, string | string[] | undefined> = {};
      for (const [key, value] of Object.entries(response.headers)) {
        responseHeaders[key] = value;
      }

      this.logger.debug(
        { method, url, requestId, statusCode: response.statusCode, duration, attempt },
        'Upstream request completed'
      );

      // Handle non-2xx responses
      if (response.statusCode >= 400) {
        throw this.createUpstreamError(response.statusCode, data, requestId);
      }

      return {
        data,
        statusCode: response.statusCode,
        headers: responseHeaders,
      };
    } catch (error) {
      const duration = Date.now() - startTime;

      if (error instanceof UpstreamError) {
        // Log warn for client errors, error for server errors
        const level = error.statusCode >= 500 ? 'error' : 'warn';
        this.logger[level](
          { method, url, requestId, duration, error: error.message, attempt },
          'Upstream error response'
        );
        throw error;
      }

      // Handle timeout
      if (error instanceof Error && error.name === 'TimeoutError') {
        this.logger.error(
          { method, url, requestId, duration, timeout: requestTimeout, attempt },
          'Upstream request timeout'
        );
        throw new UpstreamError(
          504,
          {
            code: 'UPSTREAM_TIMEOUT',
            message: `Request to ${target} timed out after ${requestTimeout}ms`,
          },
          requestId
        );
      }

      // Handle connection errors
      this.logger.error(
        { method, url, requestId, duration, error: error instanceof Error ? error.message : 'Unknown error', attempt },
        'Upstream request failed'
      );
      throw new UpstreamError(
        502,
        {
          code: 'UPSTREAM_ERROR',
          message: `Failed to connect to ${target}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        },
        requestId
      );
    }
  }

  private createUpstreamError(statusCode: number, body: unknown, requestId: string): UpstreamError {
    // Try to extract error details from upstream response
    let apiError: ApiError;

    if (typeof body === 'object' && body !== null) {
      const errorBody = body as Record<string, unknown>;
      if ('error' in errorBody && typeof errorBody.error === 'object') {
        const innerError = errorBody.error as Record<string, unknown>;
        apiError = {
          code: String(innerError.code ?? 'UPSTREAM_ERROR'),
          message: String(innerError.message ?? 'Unknown upstream error'),
          details: innerError.details as Record<string, unknown> | undefined,
        };
      } else if ('detail' in errorBody) {
        // FastAPI validation error format
        apiError = {
          code: 'VALIDATION_ERROR',
          message: String(errorBody.detail),
          details: typeof errorBody.detail === 'object' ? errorBody.detail as Record<string, unknown> : undefined,
        };
      } else if ('message' in errorBody) {
        apiError = {
          code: 'UPSTREAM_ERROR',
          message: String(errorBody.message),
        };
      } else {
        apiError = {
          code: 'UPSTREAM_ERROR',
          message: JSON.stringify(body),
        };
      }
    } else {
      apiError = {
        code: 'UPSTREAM_ERROR',
        message: String(body ?? 'Unknown error'),
      };
    }

    return new UpstreamError(statusCode, apiError, requestId);
  }
}

export class UpstreamError extends Error {
  public readonly statusCode: number;
  public readonly apiError: ApiError;
  public readonly requestId: string;

  constructor(statusCode: number, apiError: ApiError, requestId: string) {
    super(apiError.message);
    this.name = 'UpstreamError';
    this.statusCode = statusCode;
    this.apiError = apiError;
    this.requestId = requestId;
  }

  toResponse(): ErrorResponse {
    return {
      error: this.apiError,
    };
  }
}