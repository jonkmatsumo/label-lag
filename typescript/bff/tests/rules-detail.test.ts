import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { MockAgent, setGlobalDispatcher, getGlobalDispatcher, Dispatcher } from 'undici';
import { createTestApp, TestContext } from './setup';

describe('Rules Detail Routes', () => {
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

  describe('GET /bff/v1/rules', () => {
    it('returns production rules', async () => {
      ctx.mockPool.intercept({
        path: '/rules',
        method: 'GET',
      }).reply(200, {
        version: 'v1.2.0',
        rules: [
          {
            id: 'rule-001',
            field: 'amount',
            op: 'gt',
            value: 10000,
            action: 'flag',
            score: null,
            severity: 'high',
            reason: 'High amount transaction',
            status: 'active',
          },
        ],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.version).toBe('v1.2.0');
      expect(data.rules).toHaveLength(1);
    });
  });

  describe('GET /bff/v1/rules/:rule_id/readiness', () => {
    it('returns readiness report', async () => {
      ctx.mockPool.intercept({
        path: '/rules/rule-001/readiness',
        method: 'GET',
      }).reply(200, {
        rule_id: 'rule-001',
        timestamp: '2024-01-15T10:30:00Z',
        overall_status: 'ready',
        checks: [
          {
            policy_type: 'performance',
            name: 'Match Rate Check',
            status: 'passed',
            message: 'Match rate within acceptable range',
            details: { match_rate: 0.05 },
          },
          {
            policy_type: 'stability',
            name: 'Score Stability',
            status: 'passed',
            message: 'Score delta stable',
            details: { mean_delta: 2.5 },
          },
        ],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules/rule-001/readiness',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.rule_id).toBe('rule-001');
      expect(data.overall_status).toBe('ready');
      expect(data.checks).toHaveLength(2);
    });

    it('returns 404 for unknown rule', async () => {
      ctx.mockPool.intercept({
        path: '/rules/unknown-rule/readiness',
        method: 'GET',
      }).reply(404, {
        detail: 'Rule unknown-rule not found',
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules/unknown-rule/readiness',
      });

      expect(response.statusCode).toBe(404);
    });
  });

  describe('GET /bff/v1/rules/:rule_id/versions', () => {
    it('returns rule version history', async () => {
      ctx.mockPool.intercept({
        path: '/rules/rule-001/versions',
        method: 'GET',
      }).reply(200, {
        versions: [
          {
            rule_id: 'rule-001',
            version_id: 'v1',
            rule: {
              rule_id: 'rule-001',
              field: 'amount',
              op: 'gt',
              value: 5000,
              action: 'flag',
              severity: 'medium',
              reason: 'High amount',
              status: 'active',
            },
            timestamp: '2024-01-01T00:00:00Z',
            created_by: 'admin',
            reason: 'Initial version',
          },
          {
            rule_id: 'rule-001',
            version_id: 'v2',
            rule: {
              rule_id: 'rule-001',
              field: 'amount',
              op: 'gt',
              value: 10000,
              action: 'flag',
              severity: 'high',
              reason: 'High amount transaction',
              status: 'active',
            },
            timestamp: '2024-01-10T00:00:00Z',
            created_by: 'admin',
            reason: 'Increased threshold',
          },
        ],
        total: 2,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules/rule-001/versions',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.versions).toHaveLength(2);
      expect(data.total).toBe(2);
    });
  });

  describe('GET /bff/v1/rules/:rule_id/versions/:version_id', () => {
    it('returns specific version details', async () => {
      ctx.mockPool.intercept({
        path: '/rules/rule-001/versions/v2',
        method: 'GET',
      }).reply(200, {
        rule_id: 'rule-001',
        version_id: 'v2',
        rule: {
          rule_id: 'rule-001',
          field: 'amount',
          op: 'gt',
          value: 10000,
          action: 'flag',
          severity: 'high',
          reason: 'High amount transaction',
          status: 'active',
        },
        timestamp: '2024-01-10T00:00:00Z',
        created_by: 'admin',
        reason: 'Increased threshold',
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules/rule-001/versions/v2',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.version_id).toBe('v2');
      expect(data.rule.value).toBe(10000);
    });
  });

  describe('GET /bff/v1/rules/:rule_id/diff', () => {
    it('returns diff between versions', async () => {
      ctx.mockPool.intercept({
        path: '/rules/rule-001/diff?version_a=v2&version_b=v1',
        method: 'GET',
      }).reply(200, {
        rule_id: 'rule-001',
        version_a: 'v2',
        version_b: 'v1',
        changes: [
          { field: 'value', old_value: 5000, new_value: 10000 },
          { field: 'severity', old_value: 'medium', new_value: 'high' },
        ],
        is_breaking: false,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules/rule-001/diff?version_a=v2&version_b=v1',
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.changes).toHaveLength(2);
      expect(data.is_breaking).toBe(false);
    });

    it('uses default versions when not specified', async () => {
      ctx.mockPool.intercept({
        path: '/rules/rule-001/diff',
        method: 'GET',
      }).reply(200, {
        rule_id: 'rule-001',
        version_a: 'v2',
        version_b: 'v1',
        changes: [],
        is_breaking: false,
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: '/bff/v1/rules/rule-001/diff',
      });

      expect(response.statusCode).toBe(200);
    });
  });
});
