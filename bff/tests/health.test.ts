import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestApp, TestContext, restoreDispatcher } from './setup.js';

describe('Health Routes', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    ctx = await createTestApp();
  });

  afterAll(async () => {
    await ctx.app.close();
    await ctx.mockAgent.close();
    restoreDispatcher(ctx.originalDispatcher);
  });

  describe('GET /health', () => {
    it('returns healthy status for BFF own health', async () => {
      const response = await ctx.app.inject({
        method: 'GET',
        url: '/health',
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('healthy');
      expect(body.version).toBe('1.0.0');
      expect(body.uptime_seconds).toBeGreaterThanOrEqual(0);
    });
  });

  describe('GET /bff/v1/health', () => {
    it('proxies to FastAPI health endpoint', async () => {
      ctx.mockPool
        .intercept({ path: '/health', method: 'GET' })
        .reply(200, {
          status: 'healthy',
          model_loaded: true,
          model_version: 'v1.2.3',
          uptime_seconds: 3600,
        });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/health',
        headers: {
          'x-request-id': 'test-request-123',
        },
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('healthy');
      expect(body.model_loaded).toBe(true);
      expect(body.model_version).toBe('v1.2.3');
    });

    it('returns error when upstream is unavailable', async () => {
      ctx.mockPool
        .intercept({ path: '/health', method: 'GET' })
        .replyWithError(new Error('Connection refused'));

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/health',
      });

      expect(response.statusCode).toBe(502);
      const body = JSON.parse(response.payload);
      expect(body.error.code).toBe('UPSTREAM_ERROR');
    });

    it('returns 4xx error from upstream', async () => {
      ctx.mockPool
        .intercept({ path: '/health', method: 'GET' })
        .reply(404, { detail: 'Not found' });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/health',
      });

      expect(response.statusCode).toBe(404);
      const body = JSON.parse(response.payload);
      expect(body.error.code).toBe('VALIDATION_ERROR');
    });
  });
});
