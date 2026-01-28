/**
 * API client for BFF communication
 */
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

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

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
        errorData.error?.details
      );
    }

    return data as T;
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
  public readonly details?: Record<string, unknown>;

  constructor(
    statusCode: number,
    code: string,
    message: string,
    details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
  }
}

// Singleton instance
export const apiClient = new ApiClient();
