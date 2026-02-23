import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { describe, expect, it } from 'vitest';

interface RouteBlock {
  path: string;
  block: string;
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROUTES_DIR = join(__dirname, '..', 'src', 'routes');
const TIME_RANGE_TOKEN_PATTERN = /\b(start_time|end_time|start_date|end_date|granularity|group_by)\b/;
const EXPECTED_TIME_RANGE_ROUTE_PATHS = [
  '/bff/v1/analytics/confusion-matrix',
  '/bff/v1/analytics/rules/:rule_id/impact',
  '/bff/v1/analytics/transactions/search',
  '/bff/v1/jobs/summary',
  '/bff/v1/kpis',
  '/bff/v1/volume',
];

function extractRouteBlocks(fileName: string): RouteBlock[] {
  const source = readFileSync(join(ROUTES_DIR, fileName), 'utf8');
  const routeStarts = [...source.matchAll(/^\s*fastify\.(?:get|post)\b/gm)]
    .map((match) => match.index)
    .filter((index): index is number => index !== undefined);

  const blocks: RouteBlock[] = [];
  for (let i = 0; i < routeStarts.length; i += 1) {
    const start = routeStarts[i];
    const end = i + 1 < routeStarts.length ? routeStarts[i + 1] : source.length;
    const block = source.slice(start, end);
    const pathMatch = block.match(/['"](?<path>\/bff\/v1\/[^'"]+)['"]/);
    if (!pathMatch?.groups?.path) {
      continue;
    }
    blocks.push({
      path: pathMatch.groups.path,
      block,
    });
  }
  return blocks;
}

describe('Analytics query validation guardrail', () => {
  it('requires shared query resolver for all analytics time-range routes', () => {
    const routes = [
      ...extractRouteBlocks('analytics.ts'),
      ...extractRouteBlocks('jobs.ts'),
    ];
    const timeRangeRoutes = routes.filter((route) =>
      TIME_RANGE_TOKEN_PATTERN.test(route.block)
    );

    const discoveredPaths = timeRangeRoutes
      .map((route) => route.path)
      .sort();
    const expectedPaths = [...EXPECTED_TIME_RANGE_ROUTE_PATHS].sort();
    expect(discoveredPaths).toEqual(expectedPaths);

    const routesMissingResolver = timeRangeRoutes
      .filter((route) => !route.block.includes('resolveAnalyticsQueryInput('))
      .map((route) => route.path)
      .sort();
    expect(routesMissingResolver).toEqual([]);
  });
});
