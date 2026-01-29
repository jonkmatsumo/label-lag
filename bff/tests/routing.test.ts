import { describe, it, expect } from 'vitest';
import { createTestApp, createTestConfig } from './setup';

describe('Inference Routing', () => {
  it('should route to FastAPI when mode is fastapi', async () => {
    const config = createTestConfig();
    config.inferenceMode = 'fastapi';
    config.fastApiBaseUrl = 'http://mock-api:8000';
    config.gatewayBaseUrl = 'http://mock-gateway';

    const ctx = await createTestApp(config);
    
    ctx.mockPool.intercept({
      path: '/evaluate/signal',
      method: 'POST'
    }).reply(200, { score: 10 });

    const response = await ctx.app.inject({
      method: 'POST',
      url: '/bff/v1/evaluate/signal',
      payload: { user_id: 'u1', amount: 100, currency: 'USD', client_transaction_id: 't1' }
    });

    expect(response.statusCode).toBe(200);
    expect(JSON.parse(response.payload).score).toBe(10);
    
    await ctx.app.close();
  });

  it('should route to Gateway when mode is gateway', async () => {
    const config = createTestConfig();
    config.inferenceMode = 'gateway';
    config.fastApiBaseUrl = 'http://mock-api';
    config.gatewayBaseUrl = 'http://mock-gateway';

    const ctx = await createTestApp(config);
    
    // Intercept on the GATEWAY url
    const gatewayPool = ctx.mockAgent.get('http://mock-gateway');
    gatewayPool.intercept({
      path: '/evaluate/signal',
      method: 'POST'
    }).reply(200, { score: 99 });

    const response = await ctx.app.inject({
      method: 'POST',
      url: '/bff/v1/evaluate/signal',
      payload: { user_id: 'u1', amount: 100, currency: 'USD', client_transaction_id: 't1' }
    });

    expect(response.statusCode).toBe(200);
    expect(JSON.parse(response.payload).score).toBe(99);

    await ctx.app.close();
  });
});
