import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestApp, TestContext, restoreDispatcher } from './setup.js';

describe('Backtest Routes', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    ctx = await createTestApp();
  });

  afterAll(async () => {
    await ctx.app.close();
    await ctx.mockAgent.close();
    restoreDispatcher(ctx.originalDispatcher);
  });

  describe('POST /bff/v1/backtest/compare', () => {
    const validRequest = {
      base_version: 'v1.0.0',
      candidate_version: 'v1.1.0',
      start_date: '2024-01-01',
      end_date: '2024-01-07',
    };

    it('compares backtests successfully', async () => {
      ctx.mockPool
        .intercept({ path: '/backtest/compare', method: 'POST' })
        .reply(200, {
          base: {
            precision: 0.85,
            recall: 0.75,
            f1_score: 0.80,
            total_transactions: 10000,
            flagged_transactions: 500,
            true_positives: 425,
            false_positives: 75,
          },
          candidate: {
            precision: 0.88,
            recall: 0.78,
            f1_score: 0.83,
            total_transactions: 10000,
            flagged_transactions: 480,
            true_positives: 422,
            false_positives: 58,
          },
          delta: {
            precision: 0.03,
            recall: 0.03,
            f1_score: 0.03,
            flagged_rate_change: -0.002,
          },
          job_id: 'job_123',
        });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/backtest/compare',
        headers: { 'content-type': 'application/json' },
        payload: validRequest,
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.base.precision).toBe(0.85);
      expect(body.candidate.precision).toBe(0.88);
      expect(body.delta.precision).toBe(0.03);
      expect(body.job_id).toBe('job_123');
    });

    it('validates required fields', async () => {
      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/backtest/compare',
        headers: { 'content-type': 'application/json' },
        payload: { base_version: 'v1.0.0' }, // missing required fields
      });

      expect(response.statusCode).toBe(400);
    });

    it('handles no data in date range', async () => {
      ctx.mockPool
        .intercept({ path: '/backtest/compare', method: 'POST' })
        .reply(400, { detail: 'No transactions found in the specified date range' });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/backtest/compare',
        headers: { 'content-type': 'application/json' },
        payload: validRequest,
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.payload);
      expect(body.error.message).toContain('No transactions');
    });
  });
});
