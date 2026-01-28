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
}

export interface HttpResponse<T> {
  data: T;
  statusCode: number;
  headers: Record<string, string | string[] | undefined>;
}

/**
 * HTTP client for upstream service calls
 * Handles timeouts, error normalization, and request ID forwarding
 */
export class HttpClient {
  private config: Config;
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
    const { method, path, body, requestId, timeout, target = 'fastapi' } = options;
    const baseUrl = this.getBaseUrl(target);
    const url = `${baseUrl}${path}`;
    const requestTimeout = timeout ?? this.config.requestTimeout;

    const startTime = Date.now();
    this.logger.debug({ method, url, requestId }, 'Upstream request started');

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'X-Request-Id': requestId,
      };

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
        { method, url, requestId, statusCode: response.statusCode, duration },
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
        this.logger.warn(
          { method, url, requestId, duration, error: error.message },
          'Upstream error response'
        );
        throw error;
      }

      // Handle timeout
      if (error instanceof Error && error.name === 'TimeoutError') {
        this.logger.error(
          { method, url, requestId, duration, timeout: requestTimeout },
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
        { method, url, requestId, duration, error: error instanceof Error ? error.message : 'Unknown error' },
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
