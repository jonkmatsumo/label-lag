interface ReadinessCheckContract {
  name: string;
  passed: boolean;
  message: string;
}

export interface ReadinessResponseContract {
  rule_id: string;
  ready: boolean;
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

export function transformReadinessResponse(payload: unknown): ReadinessResponseContract {
  const response = asRecord(payload, 'readiness response');
  const checks = asArray(response.checks, 'readiness response.checks').map((item, index) => {
    const check = asRecord(item, `readiness response.checks[${index}]`);
    return {
      name: asString(check.name, `readiness response.checks[${index}].name`),
      passed: asBoolean(check.passed, `readiness response.checks[${index}].passed`),
      message: asString(check.message, `readiness response.checks[${index}].message`),
    };
  });

  return {
    rule_id: asString(response.rule_id, 'readiness response.rule_id'),
    ready: asBoolean(response.ready, 'readiness response.ready'),
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

  return {
    rule_id: asString(response.rule_id, 'rule diff response.rule_id'),
    version_a: asString(response.version_a, 'rule diff response.version_a'),
    version_b: asString(response.version_b, 'rule diff response.version_b'),
    changes,
  };
}
