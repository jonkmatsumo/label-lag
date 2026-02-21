import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { Dispatcher, setGlobalDispatcher } from 'undici';
import { createTestApp, createTestConfig, TestContext } from './setup';

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

describe('Jobs Summary Cache', () => {
    let ctx: TestContext;
    let originalDispatcher: Dispatcher;

    beforeAll(async () => {
        const config = createTestConfig();
        config.cacheEnabled = true;
        config.cacheTtlMs = 20_000;
        ctx = await createTestApp(config);
        originalDispatcher = ctx.originalDispatcher;
    });

    afterAll(async () => {
        await ctx.app.close();
        setGlobalDispatcher(originalDispatcher);
        await ctx.mockAgent.close();
    });

    it('repeated requests hit cache and call upstream once', async () => {
        let upstreamCalls = 0;
        ctx.mockGatewayPool.intercept({
            path: '/jobs/summary?start_time=2024-01-01&end_time=2024-01-10',
            method: 'GET',
        }).reply(200, () => {
            upstreamCalls += 1;
            return {
                summaries: [
                    {
                        bucket_time: '2024-01-01T00:00:00Z',
                        total_jobs: '10',
                        completed_jobs: '9',
                        failed_jobs: '1',
                    },
                ],
            };
        });

        const req = {
            method: 'GET' as const,
            url: '/bff/v1/jobs/summary?start_time=2024-01-01&end_time=2024-01-10',
            headers: { 'x-tenant-id': 'tenant-cache-hit' },
        };
        const first = await ctx.app.inject(req);
        const second = await ctx.app.inject(req);

        expect(first.statusCode).toBe(200);
        expect(second.statusCode).toBe(200);
        expect(upstreamCalls).toBe(1);
    });

    it('dedupes concurrent identical requests', async () => {
        let upstreamCalls = 0;
        ctx.mockGatewayPool.intercept({
            path: '/jobs/summary?start_time=2024-01-11&end_time=2024-01-12',
            method: 'GET',
        }).reply(200, () => {
            upstreamCalls += 1;
            return {
                summaries: [
                    {
                        bucket_time: '2024-01-11T00:00:00Z',
                        total_jobs: '4',
                        completed_jobs: '4',
                        failed_jobs: '0',
                    },
                ],
            };
        }).delay(100);

        const req = {
            method: 'GET' as const,
            url: '/bff/v1/jobs/summary?start_time=2024-01-11&end_time=2024-01-12',
            headers: { 'x-tenant-id': 'tenant-dedupe' },
        };

        const [first, second] = await Promise.all([ctx.app.inject(req), ctx.app.inject(req)]);
        expect(first.statusCode).toBe(200);
        expect(second.statusCode).toBe(200);
        expect(upstreamCalls).toBe(1);
    });

    it('different tenants do not share cache entries', async () => {
        let upstreamCalls = 0;
        ctx.mockGatewayPool.intercept({
            path: '/jobs/summary?start_time=2024-01-13&end_time=2024-01-14',
            method: 'GET',
        }).reply(200, () => {
            upstreamCalls += 1;
            return {
                summaries: [
                    {
                        bucket_time: '2024-01-13T00:00:00Z',
                        total_jobs: '2',
                        completed_jobs: '1',
                        failed_jobs: '1',
                    },
                ],
            };
        }).times(2);

        const baseReq = {
            method: 'GET' as const,
            url: '/bff/v1/jobs/summary?start_time=2024-01-13&end_time=2024-01-14',
        };
        const tenantOne = await ctx.app.inject({
            ...baseReq,
            headers: { 'x-tenant-id': 'tenant-one' },
        });
        const tenantTwo = await ctx.app.inject({
            ...baseReq,
            headers: { 'x-tenant-id': 'tenant-two' },
        });

        expect(tenantOne.statusCode).toBe(200);
        expect(tenantTwo.statusCode).toBe(200);
        expect(upstreamCalls).toBe(2);
    });

    it('refetches after cache TTL expiry', async () => {
        const nowSpy = vi.spyOn(Date, 'now');
        let now = Date.parse('2024-01-15T00:00:00Z');
        nowSpy.mockImplementation(() => now);
        try {
            let upstreamCalls = 0;
            ctx.mockGatewayPool.intercept({
                path: '/jobs/summary?start_time=2024-01-15&end_time=2024-01-16',
                method: 'GET',
            }).reply(200, () => {
                upstreamCalls += 1;
                return {
                    summaries: [
                        {
                            bucket_time: '2024-01-15T00:00:00Z',
                            total_jobs: '7',
                            completed_jobs: '6',
                            failed_jobs: '1',
                        },
                    ],
                };
            }).times(2);

            const req = {
                method: 'GET' as const,
                url: '/bff/v1/jobs/summary?start_time=2024-01-15&end_time=2024-01-16',
                headers: { 'x-tenant-id': 'tenant-ttl' },
            };

            const first = await ctx.app.inject(req);
            expect(first.statusCode).toBe(200);

            now += 20_001;
            const second = await ctx.app.inject(req);
            expect(second.statusCode).toBe(200);
            expect(upstreamCalls).toBe(2);
        } finally {
            nowSpy.mockRestore();
        }
    });
});
