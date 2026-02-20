/**
 * API client for BFF communication
 */
import { v4 as uuidv4 } from 'uuid';
import type { ErrorResponse } from '../types/api';

const BFF_BASE_URL = import.meta.env.VITE_BFF_BASE_URL ?? '/api';
const DEBUG_REQUESTS = import.meta.env.VITE_DEBUG_REQUESTS === 'true';

export class ApiClient {
  private baseUrl: string;
  private tenantId = '';

  constructor(baseUrl: string = BFF_BASE_URL) {
    this.baseUrl = baseUrl;
    if (DEBUG_REQUESTS) {
      console.debug('[ApiClient] Initialized', { baseUrl });
    }
  }

  setTenantId(id: string): void {
    this.tenantId = id;
  }

  getTenantId(): string {
    return this.tenantId;
  }

  private async request<T>(
    path: string,
    options: RequestInit & { signal?: AbortSignal } = {}
  ): Promise<T> {
    const { signal, ...fetchOptions } = options;
    const url = `${this.baseUrl}${path}`;
    const requestId = uuidv4();

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'X-Request-Id': requestId,
      ...(this.tenantId ? { 'X-Tenant-Id': this.tenantId } : {}),
      ...fetchOptions.headers,
    };

    if (DEBUG_REQUESTS) {
      console.debug('[ApiClient] Request', { requestId, method: fetchOptions.method ?? 'GET', url, headers });
    }

    try {
      const response = await fetch(url, {
        ...fetchOptions,
        headers,
        signal,
      });

      const data = await response.json();

      if (!response.ok) {
        const errorData = data as ErrorResponse;
        throw new ApiError(
          response.status,
          errorData.error?.code ?? 'UNKNOWN_ERROR',
          errorData.error?.message ?? 'An unexpected error occurred',
          errorData.error?.request_id ?? requestId,
          errorData.error?.details,
          errorData.error?.upstream_status
        );
      }

      return data as T;
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      // Propagate abort errors as-is so TanStack Query can handle cancellation
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw error;
      }
      // Network errors or JSON parse errors
      throw new ApiError(
        0,
        'NETWORK_ERROR',
        error instanceof Error ? error.message : 'Network error',
        requestId
      );
    }
  }

  async get<T>(path: string, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, { method: 'GET', signal });
  }

  async post<T, B = unknown>(path: string, body?: B, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });
  }

  async put<T, B = unknown>(path: string, body?: B): Promise<T> {
    return this.request<T>(path, {
      method: 'PUT',
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  async delete<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: 'DELETE' });
  }
}

export class ApiError extends Error {
  public readonly statusCode: number;
  public readonly code: string;
  public readonly requestId: string;
  public readonly details?: Record<string, unknown>;
  public readonly upstreamStatus?: number;

  constructor(
    statusCode: number,
    code: string,
    message: string,
    requestId: string,
    details?: Record<string, unknown>,
    upstreamStatus?: number
  ) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
    this.upstreamStatus = upstreamStatus;
  }
}

// Singleton instance
export const apiClient = new ApiClient();

/**
 * Guard against sending both cursor and offset in pagination params.
 * Call this before building query strings for paginated endpoints.
 */
export function assertNoPaginationConflict(params: Record<string, unknown>): void {
  if (params.cursor && params.offset !== undefined) {
    throw new Error('Cannot provide both cursor and offset — pagination conflict');
  }
}
