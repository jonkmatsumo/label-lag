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
      ctx.mockPool.intercept({
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
  });

  describe('GET /bff/v1/analytics/daily-stats', () => {
    it('returns daily stats with default days', async () => {
      ctx.mockPool.intercept({
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
      ctx.mockPool.intercept({
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
      ctx.mockPool.intercept({
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
});
