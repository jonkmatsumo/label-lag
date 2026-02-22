import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { MockAgent, setGlobalDispatcher, getGlobalDispatcher, Dispatcher } from 'undici';
import { createTestApp, TestContext } from './setup';

describe('Analytics Routes', () => {
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

  describe('GET /bff/v1/analytics/overview', () => {
    it('returns overview metrics', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/analytics/overview',
        method: 'GET',
      }).reply(200, {
        total_users: 1000,
        total_transactions: 50000,
        fraud_rate: 0.02,
        unique_merchants: 500,
        date_range: { min: '2024-01-01', max: '2024-01-31' },
      });

      ctx.mockApiPool.intercept({
        path: '/analytics/overview',
        method: 'GET',
      }).reply(200, {
        total_users: 1000,
        total_transactions: 50000,
        fraud_rate: 0.02,
        unique_merchants: 500,
        date_range: { min: '2024-01-01', max: '2024-01-31' },
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/overview',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.total_users).toBe(1000);
      expect(data.fraud_rate).toBe(0.02);
    });

    it('handles upstream authentication required error', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/analytics/overview',
        method: 'GET',
      }).reply(500, {
        message: 'Authentication required',
      }).times(2);

      ctx.mockApiPool.intercept({
        path: '/analytics/overview',
        method: 'GET',
      }).reply(200, {});

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/overview',
      });

      expect(response.statusCode).toBe(500);
      const body = response.json();
      expect(body.error.code).toBe('UPSTREAM_AUTH_ERROR');
      expect(body.error.message).toContain('Upstream service rejected the request');
      expect(body.error.upstream_status).toBe(500);
      expect(body.error.request_id).toBeDefined();
    });
  });

  describe('GET /bff/v1/analytics/daily-stats', () => {
    it('returns daily stats with default days', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/analytics/daily-stats?days=30',
        method: 'GET',
      }).reply(200, {
        stats: [
          { date: '2024-01-01', transaction_count: 100, fraud_count: 2, total_amount: 10000, avg_amount: 100 },
        ],
        period_days: 30,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/daily-stats',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.stats).toHaveLength(1);
      expect(data.period_days).toBe(30);
    });

    it('accepts custom days parameter', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/analytics/daily-stats?days=7',
        method: 'GET',
      }).reply(200, {
        stats: [],
        period_days: 7,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/daily-stats?days=7',
      });

      expect(response.statusCode).toBe(200);
    });
  });

  describe('GET /bff/v1/analytics/recent-alerts', () => {
    it('returns recent alerts', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/analytics/recent-alerts?limit=50',
        method: 'GET',
      }).reply(200, {
        alerts: [
          {
            transaction_id: 'tx-001',
            user_id: 'user-001',
            amount: 5000,
            score: 85,
            timestamp: '2024-01-15T10:30:00Z',
            matched_rules: ['high-amount', 'velocity'],
          },
        ],
        total: 1,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/recent-alerts',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.alerts).toHaveLength(1);
      expect(data.alerts[0].score).toBe(85);
    });
  });

  describe('GET /bff/v1/analytics/rules/:rule_id', () => {
    it('returns rule analytics', async () => {
      ctx.mockPool.intercept({
        path: '/analytics/rules/rule-001?days=7',
        method: 'GET',
      }).reply(200, {
        rule_id: 'rule-001',
        health: {
          rule_id: 'rule-001',
          status: 'healthy',
          reason: 'Operating normally',
          metrics: {
            period_start: '2024-01-08',
            period_end: '2024-01-15',
            production_matches: 150,
            shadow_matches: 145,
          },
        },
        statistics: {
          mean_score_delta: 5.2,
          mean_latency_ms: 2.1,
          total_matches: 295,
        },
        history_summary: [],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/rules/rule-001',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.rule_id).toBe('rule-001');
      expect(data.health.status).toBe('healthy');
    });

    it('returns 404 for unknown rule', async () => {
      ctx.mockPool.intercept({
        path: '/analytics/rules/unknown-rule?days=7',
        method: 'GET',
      }).reply(404, {
        detail: 'Rule unknown-rule not found in active ruleset',
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/rules/unknown-rule',
      });

      expect(response.statusCode).toBe(404);
    });
  });

  describe('GET /bff/v1/analytics/rules/:rule_id/impact', () => {
    it('returns 400 for invalid time range', async () => {
      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/rules/rule-1/impact?start_time=2024-01-31&end_time=2024-01-01',
      });

      expect(response.statusCode).toBe(400);
      const data = response.json();
      expect(data.error.code).toBe('INVALID_RANGE');
    });

    it('returns 400 for time range exceeding 90 days', async () => {
      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/rules/rule-1/impact?start_time=2024-01-01&end_time=2024-05-01',
      });

      expect(response.statusCode).toBe(400);
      const data = response.json();
      expect(data.error.code).toBe('INVALID_RANGE');
      expect(data.error.message).toContain('90 days');
    });

    it('returns 504 on upstream timeout', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/analytics/rules/rule-1/impact?start_date=2024-01-01&end_date=2024-01-10',
        method: 'GET',
      }).reply(504, {
        error: "gateway timeout",
        message: "deadline exceeded"
      }).times(2);

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/analytics/rules/rule-1/impact?start_time=2024-01-01&end_time=2024-01-10',
      });

      expect(response.statusCode).toBe(504);
      const data = response.json();
      expect(data.error.code).toBe('GATEWAY_TIMEOUT');
      expect(data.error.message).toBe('The upstream request timed out.');
      expect(data.error.request_id).toBeDefined();
    });

  });

  describe('POST /bff/v1/analytics/transactions/search', () => {
    it('normalizes response into items, next_cursor, and truncated', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/analytics/transactions/search',
        method: 'POST',
        body: (body) => {
          const payload = JSON.parse(body);
          return payload.limit === 10 && payload.include_features === false;
        }
      }).reply(200, {
        transactions: [{ record_id: 'rec-1', amount: 100 }],
        next_cursor: 'encoded-cursor-value',
        truncated: false
      });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/analytics/transactions/search',
        payload: {
          limit: 10
        }
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.items).toHaveLength(1);
      expect(data.items[0].record_id).toBe('rec-1');
      expect(data.next_cursor).toBe('encoded-cursor-value');
      expect(data.truncated).toBe(false);
      expect(data.transactions).toBeUndefined(); // Normalized away
    });

    it('passes include_features flag to upstream', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/analytics/transactions/search',
        method: 'POST',
        body: (body) => {
          const payload = JSON.parse(body);
          return payload.include_features === true;
        }
      }).reply(200, {
        transactions: [{ record_id: 'rec-1' }],
        next_cursor: '',
        truncated: true
      });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/analytics/transactions/search',
        payload: {
          limit: 50,
          include_features: true
        }
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.items).toHaveLength(1);
      expect(data.truncated).toBe(true);
    });
  });
});
