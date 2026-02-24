import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { setGlobalDispatcher } from 'undici';
import { createTestApp, TestContext } from './setup';
import { QUERY_LEGACY_MISMATCH_MESSAGE } from '../src/utils/analytics-query-envelope.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixtureDir = join(__dirname, '..', 'testdata', 'contracts', 'query-envelope-migration');

type RequestFixture = Record<string, unknown>;
type FixtureSet = Record<string, RequestFixture>;

function loadFixtures(filename: string): FixtureSet {
  return JSON.parse(readFileSync(join(fixtureDir, filename), 'utf-8')) as FixtureSet;
}

function buildUrl(path: string, fixture: RequestFixture): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(fixture)) {
    if (value === undefined || value === null) continue;
    if (key === 'query' && typeof value === 'object') {
      params.set('query', JSON.stringify(value));
      continue;
    }
    params.set(key, String(value));
  }
  return `${path}?${params.toString()}`;
}

describe('Analytics Query Migration Contract', () => {
  let ctx: TestContext;
  const kpiFixtures = loadFixtures('kpis.requests.json');
  const volumeFixtures = loadFixtures('volume.requests.json');

  beforeAll(async () => {
    ctx = await createTestApp();
  });

  afterAll(async () => {
    await ctx.app.close();
    setGlobalDispatcher(ctx.originalDispatcher);
    await ctx.mockAgent.close();
  });

  describe('GET /bff/v1/kpis', () => {
    const expectedPath = '/kpis?start_time=2024-01-01&end_time=2024-01-10&group_by=day';

    it('accepts legacy-only request', async () => {
      ctx.mockGatewayPool.intercept({ path: expectedPath, method: 'GET' }).reply(200, {
        total_decisions: '10',
        total_alerts: '1',
        alert_rate: 0.1,
        avg_score: 55,
        rules_fired_total: '2',
        buckets: [],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/kpis', kpiFixtures.legacy_only),
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.meta).toEqual({ truncated: false, partial: false });
    });

    it('accepts envelope-only request', async () => {
      ctx.mockGatewayPool.intercept({ path: expectedPath, method: 'GET' }).reply(200, {
        total_decisions: '10',
        total_alerts: '1',
        alert_rate: 0.1,
        avg_score: 55,
        rules_fired_total: '2',
        buckets: [],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/kpis', kpiFixtures.envelope_only),
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.meta).toEqual({ truncated: false, partial: false });
    });

    it('accepts mixed request when envelope and legacy values match', async () => {
      ctx.mockGatewayPool.intercept({ path: expectedPath, method: 'GET' }).reply(200, {
        total_decisions: '10',
        total_alerts: '1',
        alert_rate: 0.1,
        avg_score: 55,
        rules_fired_total: '2',
        buckets: [],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/kpis', kpiFixtures.both_equal),
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.meta).toEqual({ truncated: false, partial: false });
    });

    it('returns deterministic 400 when envelope and legacy values mismatch', async () => {
      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/kpis', kpiFixtures.both_mismatch),
      });

      expect(response.statusCode).toBe(400);
      const data = response.json();
      expect(data.error.code).toBe('INVALID_RANGE');
      expect(data.error.message).toBe(QUERY_LEGACY_MISMATCH_MESSAGE);
    });

    it('accepts hourly windows larger than legacy cap', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/kpis?start_time=2024-01-01&end_time=2024-02-01&group_by=hour',
        method: 'GET',
      }).reply(200, {
        total_decisions: '10',
        total_alerts: '1',
        alert_rate: 0.1,
        avg_score: 55,
        rules_fired_total: '2',
        buckets: [],
        meta: { partial: true },
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/kpis', kpiFixtures.window_too_large_hour),
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.meta).toEqual({ truncated: false, partial: true });
    });

    it('returns 400 for start_time >= end_time', async () => {
      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/kpis', kpiFixtures.start_gte_end),
      });

      expect(response.statusCode).toBe(400);
      const data = response.json();
      expect(data.error.code).toBe('INVALID_RANGE');
      expect(data.error.message).toContain('must be before');
    });
  });

  describe('GET /bff/v1/volume', () => {
    const expectedPath = '/volume?start_time=2024-01-01&end_time=2024-01-10&granularity=day';

    it('accepts legacy-only request', async () => {
      ctx.mockGatewayPool.intercept({ path: expectedPath, method: 'GET' }).reply(200, {
        points: [
          { timestamp: '2024-01-01T00:00:00Z', count: '10', alerts: '1' },
        ],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/volume', volumeFixtures.legacy_only),
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.meta).toEqual({ truncated: false, partial: false });
    });

    it('accepts envelope-only request', async () => {
      ctx.mockGatewayPool.intercept({ path: expectedPath, method: 'GET' }).reply(200, {
        points: [
          { timestamp: '2024-01-01T00:00:00Z', count: '10', alerts: '1' },
        ],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/volume', volumeFixtures.envelope_only),
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.meta).toEqual({ truncated: false, partial: false });
    });

    it('accepts mixed request when envelope and legacy values match', async () => {
      ctx.mockGatewayPool.intercept({ path: expectedPath, method: 'GET' }).reply(200, {
        points: [
          { timestamp: '2024-01-01T00:00:00Z', count: '10', alerts: '1' },
        ],
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/volume', volumeFixtures.both_equal),
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.meta).toEqual({ truncated: false, partial: false });
    });

    it('returns deterministic 400 when envelope and legacy values mismatch', async () => {
      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/volume', volumeFixtures.both_mismatch),
      });

      expect(response.statusCode).toBe(400);
      const data = response.json();
      expect(data.error.code).toBe('INVALID_RANGE');
      expect(data.error.message).toBe(QUERY_LEGACY_MISMATCH_MESSAGE);
    });

    it('accepts hourly windows larger than legacy cap', async () => {
      ctx.mockGatewayPool.intercept({
        path: '/volume?start_time=2024-01-01&end_time=2024-02-01&granularity=hour',
        method: 'GET',
      }).reply(200, {
        points: [
          { timestamp: '2024-01-01T00:00:00Z', count: '10', alerts: '1' },
        ],
        meta: { partial: true },
      });

      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/volume', volumeFixtures.window_too_large_hour),
      });

      expect(response.statusCode).toBe(200);
      const data = response.json();
      expect(data.meta).toEqual({ truncated: false, partial: true });
    });

    it('returns 400 for start_time >= end_time', async () => {
      const response = await ctx.app.inject({
        method: 'GET',
        url: buildUrl('/bff/v1/volume', volumeFixtures.start_gte_end),
      });

      expect(response.statusCode).toBe(400);
      const data = response.json();
      expect(data.error.code).toBe('INVALID_RANGE');
      expect(data.error.message).toContain('must be before');
    });
  });
});
