/**
 * API client for BFF communication
 */
import { v4 as uuidv4 } from 'uuid';
import type { ErrorResponse } from '../types/api';

const BFF_BASE_URL = import.meta.env.VITE_BFF_BASE_URL ?? '/api';

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = BFF_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const requestId = uuidv4();

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'X-Request-Id': requestId,
      ...options.headers,
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      const data = await response.json();

      if (!response.ok) {
        const errorData = data as ErrorResponse;
        throw new ApiError(
          response.status,
          errorData.error?.code ?? 'UNKNOWN_ERROR',
          errorData.error?.message ?? 'An unexpected error occurred',
          requestId,
          errorData.error?.details
        );
      }

      return data as T;
    } catch (error) {
      if (error instanceof ApiError) {
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

  async get<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: 'GET' });
  }

  async post<T, B = unknown>(path: string, body?: B): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
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

  constructor(
    statusCode: number,
    code: string,
    message: string,
    requestId: string,
    details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}

// Singleton instance
export const apiClient = new ApiClient();
