import { describe, test, afterAll, beforeAll, expect } from 'vitest';
import { createTestApp, TestContext, restoreDispatcher } from '../setup.js';
import { request } from 'undici';

const IS_E2E = process.env.SMOKE_E2E === 'true';
const BASE_URL = process.env.BFF_BASE_URL || 'http://localhost:3210';

describe('Explorer Smoke Tests (BFF-level E2E)', () => {
    let context: TestContext | undefined;

    // Helper to abstract injection vs real HTTP request
    const invoke = async (options: { method: 'GET'; url: string; headers?: Record<string, string> }) => {
        if (IS_E2E) {
            const { statusCode, body } = await request(`${BASE_URL}${options.url}`, {
                method: options.method,
                headers: options.headers,
            });
            return {
                statusCode,
                body: await body.text(),
            };
        } else {
            if (!context) throw new Error('Test context not initialized');
            const response = await context.app.inject({
                method: options.method,
                url: options.url,
                headers: options.headers,
            });
            return {
                statusCode: response.statusCode,
                body: response.body,
            };
        }
    };

    beforeAll(async () => {
        if (!IS_E2E) {
            context = await createTestApp();
        } else {
            console.log(`Running in E2E mode against ${BASE_URL}`);
        }
    });

    afterAll(() => {
        if (context) {
            restoreDispatcher(context.originalDispatcher);
        }
    });

    const TENANT_ID = process.env.TENANT_ID || 'test-tenant-smoke';
    const BASE_HEADERS = {
        'X-Tenant-Id': TENANT_ID,
    };

    test('Decisions Explorer: should return 400 if tenant header is missing', async () => {
        if (!IS_E2E && context) {
            // Mock upstream 400 for missing tenant in mocked mode
            context.mockGatewayPool
                .intercept({
                    path: '/decisions',
                    method: 'GET',
                    query: { limit: 25 }
                })
                .reply(400, {
                    error: 'Bad Request'
                });
        }

        const response = await invoke({
            method: 'GET',
            url: '/bff/v1/decisions',
            // Missing header
        });
        expect(response.statusCode).toBe(400);
        const body = JSON.parse(response.body);
        expect(body).toHaveProperty('error');
    });

    test('Decisions Explorer: list reachable (tolerant of empty data)', async () => {
        if (!IS_E2E && context) {
            context.mockGatewayPool
                .intercept({
                    path: '/decisions',
                    method: 'GET',
                    query: { limit: '25' }
                })
                .reply(200, {
                    decisions: [],
                    next_cursor: null
                });
        }

        const response = await invoke({
            method: 'GET',
            url: '/bff/v1/decisions',
            headers: BASE_HEADERS,
        });

        expect(response.statusCode).toBe(200);
        const body = JSON.parse(response.body);
        // Allow empty array or actual data, just check shape
        if (body.decisions) {
            expect(Array.isArray(body.decisions)).toBe(true);
        }
    });

    test('Training Runs: should return 200 for list endpoint', async () => {
        if (!IS_E2E && context) {
            context.mockGatewayPool
                .intercept({
                    path: '/training-runs',
                    method: 'GET',
                    query: { limit: '25' }
                })
                .reply(200, {
                    runs: [],
                    next_cursor: null
                });
        }

        const response = await invoke({
            method: 'GET',
            url: '/bff/v1/training-runs',
            headers: BASE_HEADERS,
        });

        expect(response.statusCode).toBe(200);
        const body = JSON.parse(response.body);
        // Flexible assertion: either runs array exists or it's empty
        expect(body).toHaveProperty('runs');
    });

    test('Model Registry: versions list works with pagination query', async () => {
        if (!IS_E2E && context) {
            context.mockGatewayPool
                .intercept({
                    path: '/models/versions',
                    method: 'GET',
                    query: { limit: '10' }
                })
                .reply(200, {
                    versions: [],
                    next_cursor: 'abc'
                });
        }

        const response = await invoke({
            method: 'GET',
            url: '/bff/v1/models/versions?limit=10',
            headers: BASE_HEADERS,
        });

        expect(response.statusCode).toBe(200);
        const body = JSON.parse(response.body);
        // Just verify the request didn't 404 or 500
        expect(body).toBeDefined();
    });

    test('Dataset Profiles: compare endpoint validates parameters', async () => {
        const response = await invoke({
            method: 'GET',
            url: '/bff/v1/dataset/profiles/compare', // Missing params
            headers: BASE_HEADERS,
        });

        expect(response.statusCode).toBe(400);
    });

    test('Dataset Profiles: compare endpoint passes through valid request', async () => {
        if (!IS_E2E && context) {
            context.mockGatewayPool
                .intercept({
                    path: '/dataset/profiles/compare',
                    method: 'GET',
                    query: {
                        base_id: 'p1',
                        target_id: 'p2'
                    }
                })
                .reply(200, {
                    comparison: {}
                });
        }

        const response = await invoke({
            method: 'GET',
            url: '/bff/v1/dataset/profiles/compare?base_id=p1&target_id=p2',
            headers: BASE_HEADERS,
        });

        expect(response.statusCode).toBe(200);
    });
});
