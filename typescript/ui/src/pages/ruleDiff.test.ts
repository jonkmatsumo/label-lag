import { describe, expect, it } from 'vitest';
import type { DiffRuleVersionsResponse } from '../types/api';
import { hasBreakingChanges } from './ruleDiff';

function buildDiffResponse(
  overrides: Partial<DiffRuleVersionsResponse>
): DiffRuleVersionsResponse {
  return {
    rule_id: 'rule-001',
    version_a: 'v2',
    version_b: 'v1',
    is_breaking: false,
    changes: [],
    ...overrides,
  };
}

describe('rule diff breaking signal', () => {
  it('returns true when is_breaking is true', () => {
    expect(hasBreakingChanges(buildDiffResponse({ is_breaking: true }))).toBe(true);
  });

  it('returns false when is_breaking is false even if descriptions mention breaking', () => {
    const diff = buildDiffResponse({
      is_breaking: false,
      changes: [
        {
          field_name: 'value',
          change_type: 'modified',
          before_value: '5',
          after_value: '10',
          description: 'This is not a breaking change',
        },
      ],
    });

    expect(hasBreakingChanges(diff)).toBe(false);
  });

  it('returns false when the diff payload is missing', () => {
    expect(hasBreakingChanges(undefined)).toBe(false);
  });
});
