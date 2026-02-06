import { describe, it, expect } from 'vitest';
import { createTestApp, createTestConfig } from './setup';

describe('Inference Routing', () => {
  it('routes evaluate to gateway by default', async () => {
    const config = createTestConfig();
    config.gatewayBaseUrl = 'http://mock-gateway:8081';

    const ctx = await createTestApp(config);

    const gatewayPool = ctx.mockAgent.get('http://mock-gateway:8081');
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

describe('Core UI Read Routing', () => {
  it('routes core read paths to expected upstreams', async () => {
    const ctx = await createTestApp();
    const mlflowPool = ctx.mockAgent.get('http://mock-mlflow:5000');

    const cases = [
      {
        name: 'analytics overview -> gateway',
        method: 'GET',
        url: '/bff/v1/analytics/overview',
        target: 'gateway',
        path: '/analytics/overview',
      },
      {
        name: 'monitoring drift -> python',
        method: 'GET',
        url: '/bff/v1/monitoring/drift',
        target: 'python',
        path: '/monitoring/drift?hours=24&threshold=0.25&force_refresh=false',
      },
      {
        name: 'backtest results -> gateway',
        method: 'GET',
        url: '/bff/v1/backtest/results',
        target: 'gateway',
        path: '/backtest/results?limit=50',
      },
      {
        name: 'dataset overview -> gateway',
        method: 'GET',
        url: '/bff/v1/dataset/overview',
        target: 'gateway',
        path: '/analytics/overview',
      },
      {
        name: 'analytics attribution -> python',
        method: 'GET',
        url: '/bff/v1/analytics/attribution?rule_id=rule-1',
        target: 'python',
        path: '/analytics/attribution?rule_id=rule-1&days=7',
      },
      {
        name: 'backtest compare -> python',
        method: 'POST',
        url: '/bff/v1/backtest/compare',
        target: 'python',
        path: '/backtest/compare',
        body: {
          base_version: 'v1',
          candidate_version: 'v2',
          start_date: '2025-01-01',
          end_date: '2025-01-02',
        },
      },
      {
        name: 'mlflow experiments -> mlflow',
        method: 'GET',
        url: '/bff/v1/mlflow/experiments/search',
        target: 'mlflow',
        path: '/api/2.0/mlflow/experiments/search',
      },
    ];

    for (const tc of cases) {
      let pool = ctx.mockPool;
      if (tc.target === 'gateway') {
        pool = ctx.mockGatewayPool;
      } else if (tc.target === 'mlflow') {
        pool = mlflowPool;
      }

      pool.intercept({
        path: tc.path,
        method: tc.method as 'GET' | 'POST',
      }).reply(200, { ok: true });

      const response = await ctx.app.inject({
        method: tc.method,
        url: tc.url,
        payload: tc.body,
        headers: tc.method === 'POST' ? { 'content-type': 'application/json' } : undefined,
      });

      expect(response.statusCode).toBe(200);
      expect(JSON.parse(response.payload).ok).toBe(true);
    }

    await ctx.app.close();
  });
});
