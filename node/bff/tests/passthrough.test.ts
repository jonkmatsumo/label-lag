import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestApp, TestContext, restoreDispatcher } from './setup.js';

describe('501 Passthrough', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    ctx = await createTestApp();
  });

  afterAll(async () => {
    await ctx.app.close();
    await ctx.mockAgent.close();
    restoreDispatcher(ctx.originalDispatcher);
  });

  it('preserves flat 501 bodies from gateway', async () => {
    ctx.mockGatewayPool.intercept({
      path: '/analytics/overview',
      method: 'GET',
    }).reply(501, {
      error: 'not_implemented',
      path: '/analytics/overview',
      method: 'GET',
      request_id: 'req-501',
    });

    const response = await ctx.app.inject({
      method: 'GET',
      url: '/bff/v1/analytics/overview',
    });

    expect(response.statusCode).toBe(501);
    const body = response.json();
    expect(body.error).toBe('not_implemented');
    expect(body.request_id).toBe('req-501');
    expect(typeof body.error).toBe('string');
    expect(body.error?.code).toBeUndefined();
  });
});
