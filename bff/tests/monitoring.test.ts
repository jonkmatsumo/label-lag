import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { MockAgent, setGlobalDispatcher, getGlobalDispatcher, Dispatcher } from 'undici';
import { createTestApp, TestContext } from './setup';

describe('Monitoring Routes', () => {
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

  describe('GET /bff/v1/monitoring/drift', () => {
    it('returns drift status', async () => {
      ctx.mockPool.intercept({
        path: '/monitoring/drift?hours=24&threshold=0.25&force_refresh=false',
        method: 'GET',
      }).reply(200, {
        status: 'ok',
        message: 'No significant drift detected',
        drift_detected: false,
        cached: false,
        computed_at: '2024-01-15T10:30:00Z',
        hours_analyzed: 24,
        threshold: 0.25,
        feature_details: [
          {
            feature_name: 'amount',
            psi_value: 0.05,
            status: 'ok',
          },
        ],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/monitoring/drift',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.status).toBe('ok');
      expect(data.drift_detected).toBe(false);
      expect(data.feature_details).toHaveLength(1);
    });

    it('returns drift warning status', async () => {
      ctx.mockPool.intercept({
        path: '/monitoring/drift?hours=24&threshold=0.25&force_refresh=false',
        method: 'GET',
      }).reply(200, {
        status: 'warning',
        message: 'Drift detected in 2 features',
        drift_detected: true,
        cached: true,
        computed_at: '2024-01-15T09:00:00Z',
        hours_analyzed: 24,
        threshold: 0.25,
        feature_details: [
          { feature_name: 'velocity_24h', psi_value: 0.35, status: 'warning' },
          { feature_name: 'amount', psi_value: 0.42, status: 'critical' },
        ],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/monitoring/drift',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.status).toBe('warning');
      expect(data.drift_detected).toBe(true);
    });

    it('accepts custom parameters', async () => {
      ctx.mockPool.intercept({
        path: '/monitoring/drift?hours=48&threshold=0.3&force_refresh=true',
        method: 'GET',
      }).reply(200, {
        status: 'ok',
        message: 'No significant drift detected',
        drift_detected: false,
        cached: false,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/monitoring/drift?hours=48&threshold=0.3&force_refresh=true',
      });

      expect(response.statusCode).toBe(200);
    });
  });

  describe('GET /bff/v1/metrics/shadow/comparison', () => {
    it('returns shadow comparison metrics', async () => {
      ctx.mockPool.intercept({
        path: '/metrics/shadow/comparison?start_date=2024-01-01&end_date=2024-01-15',
        method: 'GET',
      }).reply(200, {
        period_start: '2024-01-01',
        period_end: '2024-01-15',
        rule_metrics: [
          {
            rule_id: 'rule-001',
            period_start: '2024-01-01',
            period_end: '2024-01-15',
            production_matches: 100,
            shadow_matches: 110,
            overlap_count: 95,
            production_only_count: 5,
            shadow_only_count: 15,
          },
        ],
        total_requests: 5000,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/metrics/shadow/comparison?start_date=2024-01-01&end_date=2024-01-15',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.period_start).toBe('2024-01-01');
      expect(data.rule_metrics).toHaveLength(1);
      expect(data.total_requests).toBe(5000);
    });

    it('validates required date parameters', async () => {
      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/metrics/shadow/comparison',
      });

      expect(response.statusCode).toBe(400);
    });
  });

  describe('GET /bff/v1/backtest/results', () => {
    it('returns backtest results list', async () => {
      ctx.mockPool.intercept({
        path: '/backtest/results?limit=50',
        method: 'GET',
      }).reply(200, {
        results: [
          {
            id: 'bt-001',
            rule_id: 'rule-001',
            created_at: '2024-01-15T10:00:00Z',
            status: 'completed',
            metrics: {
              precision: 0.85,
              recall: 0.72,
              f1_score: 0.78,
              total_transactions: 1000,
              flagged_transactions: 100,
              true_positives: 72,
              false_positives: 28,
            },
          },
        ],
        total: 1,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/backtest/results',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.results).toHaveLength(1);
      expect(data.results[0].status).toBe('completed');
    });

    it('accepts rule_id filter', async () => {
      ctx.mockPool.intercept({
        path: '/backtest/results?limit=50&rule_id=rule-001',
        method: 'GET',
      }).reply(200, {
        results: [],
        total: 0,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/backtest/results?rule_id=rule-001',
      });

      expect(response.statusCode).toBe(200);
    });
  });
});
