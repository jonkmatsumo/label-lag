interface ReadinessCheckContract {
  name: string;
  passed: boolean;
  status: ReadinessStatus;
  message: string;
}

export type ReadinessStatus =
  | 'READINESS_STATUS_UNSPECIFIED'
  | 'READINESS_STATUS_PASS'
  | 'READINESS_STATUS_WARN'
  | 'READINESS_STATUS_FAIL'
  | 'UNRECOGNIZED';

export interface ReadinessResponseContract {
  rule_id: string;
  ready: boolean;
  overall_status: ReadinessStatus;
  checks: ReadinessCheckContract[];
}

interface RuleDiffChangeContract {
  field: string;
  old_value: string;
  new_value: string;
  description: string;
}

export interface RuleDiffResponseContract {
  rule_id: string;
  version_a: string;
  version_b: string;
  changes: RuleDiffChangeContract[];
  is_breaking: boolean;
}

function asRecord(value: unknown, context: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${context} must be an object`);
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown, context: string): string {
  if (typeof value !== 'string') {
    throw new Error(`${context} must be a string`);
  }
  return value;
}

function asBoolean(value: unknown, context: string): boolean {
  if (typeof value !== 'boolean') {
    throw new Error(`${context} must be a boolean`);
  }
  return value;
}

function asArray(value: unknown, context: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context} must be an array`);
  }
  return value;
}

const VALID_READINESS_STATUS: ReadinessStatus[] = [
  'READINESS_STATUS_UNSPECIFIED',
  'READINESS_STATUS_PASS',
  'READINESS_STATUS_WARN',
  'READINESS_STATUS_FAIL',
  'UNRECOGNIZED',
];

function parseReadinessStatus(
  value: unknown,
  fallback: ReadinessStatus
): ReadinessStatus {
  if (typeof value !== 'string') {
    return fallback;
  }
  if (VALID_READINESS_STATUS.includes(value as ReadinessStatus)) {
    return value as ReadinessStatus;
  }
  return 'READINESS_STATUS_UNSPECIFIED';
}

function readinessStatusFromPassed(passed: boolean): ReadinessStatus {
  return passed ? 'READINESS_STATUS_PASS' : 'READINESS_STATUS_FAIL';
}

export function transformReadinessResponse(payload: unknown): ReadinessResponseContract {
  const response = asRecord(payload, 'readiness response');
  const ready = asBoolean(response.ready, 'readiness response.ready');
  const overall_status = parseReadinessStatus(
    response.overall_status,
    readinessStatusFromPassed(ready)
  );
  const checks = asArray(response.checks, 'readiness response.checks').map((item, index) => {
    const check = asRecord(item, `readiness response.checks[${index}]`);
    const passed = asBoolean(check.passed, `readiness response.checks[${index}].passed`);
    return {
      name: asString(check.name, `readiness response.checks[${index}].name`),
      passed,
      status: parseReadinessStatus(
        check.status,
        readinessStatusFromPassed(passed)
      ),
      message: asString(check.message, `readiness response.checks[${index}].message`),
    };
  });

  return {
    rule_id: asString(response.rule_id, 'readiness response.rule_id'),
    ready,
    overall_status,
    checks,
  };
}

export function transformRuleDiffResponse(payload: unknown): RuleDiffResponseContract {
  const response = asRecord(payload, 'rule diff response');
  const changes = asArray(response.changes, 'rule diff response.changes').map((item, index) => {
    const change = asRecord(item, `rule diff response.changes[${index}]`);
    return {
      field: asString(change.field, `rule diff response.changes[${index}].field`),
      old_value: asString(change.old_value, `rule diff response.changes[${index}].old_value`),
      new_value: asString(change.new_value, `rule diff response.changes[${index}].new_value`),
      description: asString(change.description, `rule diff response.changes[${index}].description`),
    };
  });
  const is_breaking = typeof response.is_breaking === 'boolean'
    ? response.is_breaking
    : inferBreakingFromChanges(changes);

  return {
    rule_id: asString(response.rule_id, 'rule diff response.rule_id'),
    version_a: asString(response.version_a, 'rule diff response.version_a'),
    version_b: asString(response.version_b, 'rule diff response.version_b'),
    changes,
    is_breaking,
  };
}

function inferBreakingFromChanges(changes: RuleDiffChangeContract[]): boolean {
  return changes.some((change) => {
    if (change.old_value.trim() !== '' && change.new_value.trim() === '') {
      return true;
    }

    if (change.old_value.trim() === '' && change.new_value.trim() !== '' && isRequiredRuleField(change.field)) {
      return true;
    }

    if (change.field === 'value' && inferValueType(change.old_value) !== inferValueType(change.new_value)) {
      return true;
    }

    return change.field === 'field' || change.field === 'op' || change.field === 'action';
  });
}

function isRequiredRuleField(field: string): boolean {
  return field === 'field' || field === 'op' || field === 'value' || field === 'action';
}

function inferValueType(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed === '') {
    return 'empty';
  }

  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (parsed === null) {
      return 'null';
    }
    if (Array.isArray(parsed)) {
      return 'array';
    }
    switch (typeof parsed) {
      case 'boolean':
        return 'bool';
      case 'number':
        return 'number';
      case 'string':
        return 'string';
      case 'object':
        return 'object';
      default:
        return 'unknown';
    }
  } catch {
    return 'string';
  }
}
