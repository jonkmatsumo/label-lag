import type { GetRuleReadinessResponse, ReadinessCheck } from '../types/api';
import { ReadinessStatus } from '../types/generated/analytics/v1/analytics';

export type ReadinessDisplayStatus = 'pass' | 'warn' | 'fail' | 'unknown';

function statusFromReadyFlag(ready: boolean | undefined): ReadinessDisplayStatus {
  if (ready === true) {
    return 'pass';
  }
  if (ready === false) {
    return 'fail';
  }
  return 'unknown';
}

export function toOverallReadinessStatus(
  response: GetRuleReadinessResponse | undefined
): ReadinessDisplayStatus {
  if (!response) {
    return 'unknown';
  }

  switch (response.overall_status) {
    case ReadinessStatus.READINESS_STATUS_PASS:
      return 'pass';
    case ReadinessStatus.READINESS_STATUS_WARN:
      return 'warn';
    case ReadinessStatus.READINESS_STATUS_FAIL:
      return 'fail';
    case ReadinessStatus.READINESS_STATUS_UNSPECIFIED:
      return statusFromReadyFlag(response.ready);
    case ReadinessStatus.UNRECOGNIZED:
      return 'unknown';
    default:
      return 'unknown';
  }
}

export function toCheckReadinessStatus(check: ReadinessCheck): ReadinessDisplayStatus {
  switch (check.status) {
    case ReadinessStatus.READINESS_STATUS_PASS:
      return 'pass';
    case ReadinessStatus.READINESS_STATUS_WARN:
      return 'warn';
    case ReadinessStatus.READINESS_STATUS_FAIL:
      return 'fail';
    case ReadinessStatus.READINESS_STATUS_UNSPECIFIED:
      return check.passed ? 'pass' : 'fail';
    case ReadinessStatus.UNRECOGNIZED:
      return 'unknown';
    default:
      return 'unknown';
  }
}

export function canPublishFromReadiness(
  response: GetRuleReadinessResponse | undefined
): boolean {
  const status = toOverallReadinessStatus(response);
  return status === 'pass' || status === 'warn';
}
