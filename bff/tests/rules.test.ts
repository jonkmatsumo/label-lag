import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestApp, TestContext, restoreDispatcher } from './setup.js';

describe('Rules Routes', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    ctx = await createTestApp();
  });

  afterAll(async () => {
    await ctx.app.close();
    await ctx.mockAgent.close();
    restoreDispatcher(ctx.originalDispatcher);
  });

  describe('GET /bff/v1/rules/draft', () => {
    it('fetches draft rules successfully', async () => {
      ctx.mockPool
        .intercept({ path: '/rules/draft', method: 'GET' })
        .reply(200, {
          rules: [
            {
              id: 'rule_1',
              name: 'High Amount Rule',
              description: 'Flag high amount transactions',
              condition: 'amount > 10000',
              action: 'clamp_min',
              score_adjustment: 20,
              status: 'approved',
              created_at: '2024-01-01T00:00:00Z',
              updated_at: '2024-01-02T00:00:00Z',
            },
          ],
          total: 1,
        });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules/draft',
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.rules).toHaveLength(1);
      expect(body.rules[0].name).toBe('High Amount Rule');
      expect(body.rules[0].status).toBe('approved');
    });

    it('handles empty rules list', async () => {
      ctx.mockPool
        .intercept({ path: '/rules/draft', method: 'GET' })
        .reply(200, { rules: [], total: 0 });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules/draft',
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.rules).toHaveLength(0);
    });
  });

  describe('POST /bff/v1/rules/:id/publish', () => {
    it('publishes rule successfully', async () => {
      ctx.mockPool
        .intercept({ path: '/rules/rule_1/publish', method: 'POST' })
        .reply(200, {
          status: 'success',
          message: 'Rule published to production',
          rule_id: 'rule_1',
          version: 1,
        });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/rules/rule_1/publish',
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('success');
      expect(body.rule_id).toBe('rule_1');
    });

    it('handles rule not found', async () => {
      ctx.mockPool
        .intercept({ path: '/rules/nonexistent/publish', method: 'POST' })
        .reply(404, { detail: 'Rule not found' });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/rules/nonexistent/publish',
      });

      expect(response.statusCode).toBe(404);
    });

    it('handles rule not approved error', async () => {
      ctx.mockPool
        .intercept({ path: '/rules/rule_2/publish', method: 'POST' })
        .reply(400, { detail: 'Rule must be approved before publishing' });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/rules/rule_2/publish',
      });

      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.payload);
      expect(body.error.message).toContain('approved');
    });
  });

  describe('POST /bff/v1/rules/sandbox/evaluate', () => {
    const validRequest = {
      base_score: 50,
      features: {
        amount: 5000,
        velocity_24h: 10,
      },
    };

    it('evaluates sandbox successfully', async () => {
      ctx.mockPool
        .intercept({ path: '/rules/sandbox/evaluate', method: 'POST' })
        .reply(200, {
          final_score: 70,
          matched_rules: [
            { rule_id: 'rule_1', name: 'High Velocity', action: 'clamp_min', score_adjustment: 20 },
          ],
          shadow_matched_rules: [],
          evaluation_details: {},
        });

      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/rules/sandbox/evaluate',
        headers: { 'content-type': 'application/json' },
        payload: validRequest,
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.final_score).toBe(70);
      expect(body.matched_rules).toHaveLength(1);
    });

    it('validates required fields', async () => {
      const response = await ctx.app.inject({
        method: 'POST',
        url: '/bff/v1/rules/sandbox/evaluate',
        headers: { 'content-type': 'application/json' },
        payload: { features: {} }, // missing base_score
      });

      expect(response.statusCode).toBe(400);
    });
  });
});
