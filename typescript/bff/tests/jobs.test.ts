import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Dispatcher, setGlobalDispatcher } from 'undici';
import { createTestApp, TestContext } from './setup';

describe('Jobs Routes', () => {
    let ctx: TestContext;
    let originalDispatcher: Dispatcher;

    beforeAll(async () => {
        ctx = await createTestApp();
        originalDispatcher = ctx.originalDispatcher;
    });

    afterAll(async () => {
        await ctx.app.close();
        setGlobalDispatcher(originalDispatcher);
        await ctx.mockAgent.close();
    });

    describe('GET /bff/v1/jobs/summary', () => {
        it('returns 400 for invalid time range', async () => {
            const response = await ctx.app.inject({
                method: 'GET',
                url: '/bff/v1/jobs/summary?start_time=2024-01-31&end_time=2024-01-01',
            });

            expect(response.statusCode).toBe(400);
            const data = response.json();
            expect(data.error.code).toBe('INVALID_RANGE');
        });

        it('returns 400 for time range exceeding 90 days', async () => {
            const response = await ctx.app.inject({
                method: 'GET',
                url: '/bff/v1/jobs/summary?start_time=2024-01-01&end_time=2024-05-01',
            });

            expect(response.statusCode).toBe(400);
            const data = response.json();
            expect(data.error.code).toBe('INVALID_RANGE');
            expect(data.error.message).toContain('90 days');
        });

        it('returns 504 on upstream timeout', async () => {
            ctx.mockGatewayPool.intercept({
                path: '/jobs/summary?start_time=2024-01-01&end_time=2024-01-10',
                method: 'GET',
            }).reply(504, {
                error: "gateway timeout",
                message: "deadline exceeded"
            }).times(2);

            const response = await ctx.app.inject({
                method: 'GET',
                url: '/bff/v1/jobs/summary?start_time=2024-01-01&end_time=2024-01-10',
            });

            if (response.statusCode !== 504) {
                console.error("RESPONSE IS:", response.body);
            }
            expect(response.statusCode).toBe(504);
            const data = response.json();
            expect(data.error.code).toBe('GATEWAY_TIMEOUT');
            expect(data.error.message).toBe('The upstream request timed out.');
            expect(data.error.request_id).toBeDefined();
        });
    });
});
