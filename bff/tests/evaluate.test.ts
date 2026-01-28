import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestApp, TestContext, restoreDispatcher } from './setup.js';

describe('Evaluate Routes', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    ctx = await createTestApp();
  });

  afterAll(async () => {
    await ctx.app.close();
    await ctx.mockAgent.close();
    restoreDispatcher(ctx.originalDispatcher);
  });

  describe('POST /bff/v1/evaluate/signal', () => {
    const validRequest = {
      user_id: 'user_123',
      amount: 100.50,
      currency: 'USD',
      client_transaction_id: 'txn_abc123',
    };

    it('evaluates signal successfully', async () => {
      ctx.mockPool
        .intercept({ path: '/evaluate/signal', method: 'POST' })
        .reply(200, {
          request_id: 'req_123',
          score: 45,
          risk_components: [
            { name: 'velocity', score: 30, weight: 0.5 },
            { name: 'amount', score: 60, weight: 0.5 },
          ],
          model_version: 'v1.0.0',
          matched_rules: [],
        });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/evaluate/signal',
        headers: {
          'content-type': 'application/json',
          'x-request-id': 'test-req-456',
        },
        payload: validRequest,
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.score).toBe(45);
      expect(body.request_id).toBe('req_123');
      expect(body.risk_components).toHaveLength(2);
    });

    it('validates required fields', async () => {
      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/evaluate/signal',
        headers: { 'content-type': 'application/json' },
        payload: { user_id: 'user_123' }, // missing required fields
      });

      expect(response.statusCode).toBe(400);
    });

    it('handles upstream error response', async () => {
      ctx.mockPool
        .intercept({ path: '/evaluate/signal', method: 'POST' })
        .reply(422, { detail: 'Invalid amount' });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/evaluate/signal',
        headers: { 'content-type': 'application/json' },
        payload: validRequest,
      });

      expect(response.statusCode).toBe(422);
      const body = JSON.parse(response.payload);
      expect(body.error.code).toBe('VALIDATION_ERROR');
    });
  });
});
