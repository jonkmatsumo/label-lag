import { describe, expect, it } from 'vitest';
import type { GetRuleReadinessResponse, ReadinessCheck } from '../types/api';
import { ReadinessStatus } from '../types/generated/analytics/v1/analytics';
import {
  canPublishFromReadiness,
  toCheckReadinessStatus,
  toOverallReadinessStatus,
} from './ruleReadiness';

function buildReadinessResponse(
  overrides: Partial<GetRuleReadinessResponse>
): GetRuleReadinessResponse {
  return {
    rule_id: 'rule-001',
    ready: true,
    overall_status: ReadinessStatus.READINESS_STATUS_PASS,
    checks: [],
    ...overrides,
  };
}

describe('rule readiness status mapping', () => {
  it('maps pass to publishable status', () => {
    const response = buildReadinessResponse({
      ready: true,
      overall_status: ReadinessStatus.READINESS_STATUS_PASS,
    });

    expect(toOverallReadinessStatus(response)).toBe('pass');
    expect(canPublishFromReadiness(response)).toBe(true);
  });

  it('maps fail to non-publishable status', () => {
    const response = buildReadinessResponse({
      ready: false,
      overall_status: ReadinessStatus.READINESS_STATUS_FAIL,
    });

    expect(toOverallReadinessStatus(response)).toBe('fail');
    expect(canPublishFromReadiness(response)).toBe(false);
  });

  it('treats unrecognized statuses as unknown and non-publishable', () => {
    const response = buildReadinessResponse({
      ready: true,
      overall_status: ReadinessStatus.UNRECOGNIZED,
    });

    expect(toOverallReadinessStatus(response)).toBe('unknown');
    expect(canPublishFromReadiness(response)).toBe(false);
  });

  it('maps unknown check status to unknown safely', () => {
    const check: ReadinessCheck = {
      name: 'integrity',
      passed: true,
      status: 'SOMETHING_NEW' as unknown as ReadinessStatus,
      message: 'unknown status value',
    };

    expect(toCheckReadinessStatus(check)).toBe('unknown');
  });
});
