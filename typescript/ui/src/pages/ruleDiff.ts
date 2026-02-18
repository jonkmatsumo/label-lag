import type { DiffRuleVersionsResponse } from '../types/api';

export function hasBreakingChanges(diff: DiffRuleVersionsResponse | undefined): boolean {
  return diff?.is_breaking === true;
}
