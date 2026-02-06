import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestApp, TestContext, restoreDispatcher } from './setup.js';

describe('Request ID Propagation', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    ctx = await createTestApp();
  });

  afterAll(async () => {
    await ctx.app.close();
    await ctx.mockAgent.close();
    restoreDispatcher(ctx.originalDispatcher);
  });

  it('forwards X-Request-Id to gateway', async () => {
    ctx.mockApiPool.intercept({
      path: '/monitoring/drift?hours=24&threshold=0.25&force_refresh=false',
      method: 'GET',
      headers: {
        'x-request-id': 'req-prop-1',
      },
    }).reply(200, { ok: true });

    const response = await ctx.app.inject({
      method: 'GET',
      url: '/bff/v1/monitoring/drift',
      headers: {
        'x-request-id': 'req-prop-1',
      },
    });

    expect(response.statusCode).toBe(200);
    expect(response.json().ok).toBe(true);
  });
});
