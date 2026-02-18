import { describe, expect, it } from 'vitest';
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import {
  transformReadinessResponse,
  transformRuleDiffResponse,
} from '../src/contracts/rules-detail.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixtureDir = join(__dirname, '..', 'testdata', 'contracts', 'rules-detail');

function loadFixture<T>(name: string): T {
  return JSON.parse(readFileSync(join(fixtureDir, name), 'utf-8')) as T;
}

describe('rules-detail contract transforms', () => {
  it('maps readiness payload to observed BFF contract shape', () => {
    const upstream = loadFixture<unknown>('readiness.upstream.json');
    const expected = loadFixture<unknown>('readiness.bff.json');

    expect(transformReadinessResponse(upstream)).toEqual(expected);
  });

  it('normalizes unknown readiness statuses to a safe enum value', () => {
    const transformed = transformReadinessResponse({
      rule_id: 'rule-001',
      ready: false,
      overall_status: 'UNSUPPORTED_STATUS',
      checks: [
        {
          name: 'integrity',
          passed: false,
          status: 'UNSUPPORTED_STATUS',
          message: 'failing integrity check',
        },
      ],
    });

    expect(transformed.overall_status).toBe('READINESS_STATUS_UNSPECIFIED');
    expect(transformed.checks[0]?.status).toBe('READINESS_STATUS_UNSPECIFIED');
  });

  it('maps diff payload to observed BFF contract shape', () => {
    const upstream = loadFixture<unknown>('diff.upstream.json');
    const expected = loadFixture<unknown>('diff.bff.json');

    expect(transformRuleDiffResponse(upstream)).toEqual(expected);
  });

  it('preserves explicit breaking flags from upstream diff payloads', () => {
    const breakingUpstream = loadFixture<unknown>('diff.breaking.upstream.json');
    const breakingExpected = loadFixture<unknown>('diff.breaking.bff.json');

    expect(transformRuleDiffResponse(breakingUpstream)).toEqual(breakingExpected);

    const nonBreakingUpstream = loadFixture<unknown>('diff.non-breaking.upstream.json');
    const nonBreakingExpected = loadFixture<unknown>('diff.non-breaking.bff.json');

    expect(transformRuleDiffResponse(nonBreakingUpstream)).toEqual(nonBreakingExpected);
  });
});
