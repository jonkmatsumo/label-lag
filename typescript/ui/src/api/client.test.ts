import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ApiClient, ApiError } from './client';

globalThis.fetch = vi.fn();

describe('ApiClient', () => {
    let client: ApiClient;

    beforeEach(() => {
        client = new ApiClient('http://test-api');
        vi.clearAllMocks();
    });

    it('should throw ApiError with correct properties on error response', async () => {
        const errorResponse = {
            error: {
                code: 'TEST_ERROR',
                message: 'Something went wrong',
                request_id: 'req-123',
                details: { foo: 'bar' },
                upstream_status: 503,
            },
        };

        vi.mocked(globalThis.fetch).mockResolvedValue({
            ok: false,
            status: 400,
            json: async () => errorResponse,
        } as Response);

        try {
            await client.get('/test');
        } catch (error) {
            expect(error).toBeInstanceOf(ApiError);
            const apiError = error as ApiError;
            expect(apiError.statusCode).toBe(400);
            expect(apiError.code).toBe('TEST_ERROR');
            expect(apiError.message).toBe('Something went wrong');
            expect(apiError.requestId).toBe('req-123');
            expect(apiError.details).toEqual({ foo: 'bar' });
            expect(apiError.upstreamStatus).toBe(503);
        }
    });

    it('should include request_id from header if missing in error body', async () => {
        vi.mocked(globalThis.fetch).mockResolvedValue({
            ok: false,
            status: 500,
            json: async () => ({}),
        } as Response);

        try {
            await client.get('/test');
        } catch (error) {
            expect(error).toBeInstanceOf(ApiError);
            const apiError = error as ApiError;
            expect(apiError.requestId).toBeDefined();
            expect(apiError.requestId.length).toBeGreaterThan(0);
            expect(apiError.code).toBe('UNKNOWN_ERROR');
        }
    });
});
