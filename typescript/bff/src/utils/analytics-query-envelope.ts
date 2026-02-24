export type AnalyticsGranularity = 'hour' | 'day';

export interface AnalyticsQueryEnvelope {
  start_time?: string;
  end_time?: string;
  granularity: AnalyticsGranularity;
  window_days?: number;
}

export interface AnalyticsQueryEnvelopeInput {
  start_time?: string;
  end_time?: string;
  granularity?: string;
}

export interface ValidateAnalyticsQueryOptions {
  required?: boolean;
  defaultGranularity?: AnalyticsGranularity;
  startField?: string;
  endField?: string;
  maxWindowDaysByGranularity?: Partial<Record<AnalyticsGranularity, number>>;
}

export interface ResolveAnalyticsQueryInput {
  query?: unknown;
  legacy: AnalyticsQueryEnvelopeInput;
  options?: ValidateAnalyticsQueryOptions;
}

interface ValidationErrorEnvelope {
  error: {
    code: 'INVALID_RANGE';
    message: string;
  };
}

export type AnalyticsQueryEnvelopeValidationResult =
  | {
      ok: true;
      value: AnalyticsQueryEnvelope;
    }
  | {
      ok: false;
      statusCode: 400;
      body: ValidationErrorEnvelope;
    };

const MAX_DAYS_BY_GRANULARITY: Record<AnalyticsGranularity, number> = {
  day: 90,
  hour: 14,
};

const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const ISO_TIMESTAMP_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$/;
export const QUERY_LEGACY_MISMATCH_MESSAGE =
  'query and legacy time fields must match when both are provided';

function invalidRange(message: string): AnalyticsQueryEnvelopeValidationResult {
  return {
    ok: false,
    statusCode: 400,
    body: {
      error: {
        code: 'INVALID_RANGE',
        message,
      },
    },
  };
}

function parseTimestamp(value: string): Date | null {
  if (ISO_DATE_RE.test(value)) {
    const parsed = new Date(`${value}T00:00:00.000Z`);
    if (Number.isNaN(parsed.getTime())) {
      return null;
    }
    const normalizedDate = parsed.toISOString().slice(0, 10);
    return normalizedDate === value ? parsed : null;
  }

  if (!ISO_TIMESTAMP_RE.test(value)) {
    return null;
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }

  return parsed;
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function toOptionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  return typeof value === 'string' ? value : undefined;
}

function normalizeGranularity(
  value: string | undefined,
  fallback: AnalyticsGranularity
): AnalyticsGranularity | null {
  if (!value || value === '') {
    return fallback;
  }
  if (value === 'day' || value === 'hour') {
    return value;
  }
  return null;
}

function hasAnyQueryFields(input: AnalyticsQueryEnvelopeInput): boolean {
  return Boolean(
    (input.start_time && input.start_time !== '') ||
      (input.end_time && input.end_time !== '') ||
      (input.granularity && input.granularity !== '')
  );
}

function parseQueryEnvelopeInput(
  value: unknown
): AnalyticsQueryEnvelopeValidationResult | { ok: true; value?: AnalyticsQueryEnvelopeInput } {
  if (value === undefined || value === null || value === '') {
    return {
      ok: true,
      value: undefined,
    };
  }

  let queryObject: Record<string, unknown>;
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      if (!isObjectRecord(parsed)) {
        return invalidRange('query must be a JSON object');
      }
      queryObject = parsed;
    } catch {
      return invalidRange('query must be a valid JSON object');
    }
  } else if (isObjectRecord(value)) {
    queryObject = value;
  } else {
    return invalidRange('query must be a JSON object');
  }

  const start_time = toOptionalString(queryObject.start_time);
  const end_time = toOptionalString(queryObject.end_time);
  const granularity = toOptionalString(queryObject.granularity);

  if (queryObject.start_time !== undefined && start_time === undefined) {
    return invalidRange('query.start_time must be a string');
  }
  if (queryObject.end_time !== undefined && end_time === undefined) {
    return invalidRange('query.end_time must be a string');
  }
  if (queryObject.granularity !== undefined && granularity === undefined) {
    return invalidRange('query.granularity must be a string');
  }

  return {
    ok: true,
    value: {
      start_time,
      end_time,
      granularity,
    },
  };
}

function timestampsEqual(left: string | undefined, right: string | undefined): boolean {
  if (left === undefined || right === undefined) {
    return left === right;
  }
  const leftDate = parseTimestamp(left);
  const rightDate = parseTimestamp(right);
  if (!leftDate || !rightDate) {
    return left === right;
  }
  return leftDate.getTime() === rightDate.getTime();
}

function ensureLegacyMatchesQuery(
  queryValue: AnalyticsQueryEnvelope,
  legacyRaw: AnalyticsQueryEnvelopeInput,
  legacyValue: AnalyticsQueryEnvelope
): AnalyticsQueryEnvelopeValidationResult | { ok: true } {
  if (legacyRaw.start_time !== undefined) {
    if (!timestampsEqual(queryValue.start_time, legacyValue.start_time)) {
      return invalidRange(QUERY_LEGACY_MISMATCH_MESSAGE);
    }
  }
  if (legacyRaw.end_time !== undefined) {
    if (!timestampsEqual(queryValue.end_time, legacyValue.end_time)) {
      return invalidRange(QUERY_LEGACY_MISMATCH_MESSAGE);
    }
  }
  if (legacyRaw.granularity !== undefined) {
    if (queryValue.granularity !== legacyValue.granularity) {
      return invalidRange(QUERY_LEGACY_MISMATCH_MESSAGE);
    }
  }
  return { ok: true };
}

export function validateAnalyticsQuery(
  input: AnalyticsQueryEnvelopeInput,
  options: ValidateAnalyticsQueryOptions = {}
): AnalyticsQueryEnvelopeValidationResult {
  const required = options.required ?? true;
  const defaultGranularity = options.defaultGranularity ?? 'day';
  const startField = options.startField ?? 'start_time';
  const endField = options.endField ?? 'end_time';
  const maxWindowDaysByGranularity = {
    ...MAX_DAYS_BY_GRANULARITY,
    ...(options.maxWindowDaysByGranularity ?? {}),
  };
  const { start_time, end_time } = input;

  if (!start_time && !end_time) {
    if (required) {
      return invalidRange(`${startField} and ${endField} are required`);
    }

    const granularity = normalizeGranularity(input.granularity, defaultGranularity);
    if (!granularity) {
      return invalidRange("granularity must be 'day' or 'hour'");
    }

    return {
      ok: true,
      value: {
        granularity,
      },
    };
  }

  if (!start_time || !end_time) {
    return invalidRange(`${startField} and ${endField} must be provided together`);
  }

  const granularity = normalizeGranularity(input.granularity, defaultGranularity);
  if (!granularity) {
    return invalidRange("granularity must be 'day' or 'hour'");
  }

  const start = parseTimestamp(start_time);
  const end = parseTimestamp(end_time);
  if (!start || !end) {
    return invalidRange(`${startField} and ${endField} must be valid ISO timestamps`);
  }

  if (start.getTime() >= end.getTime()) {
    return invalidRange(`${startField} must be before ${endField}`);
  }

  const windowDays = (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24);
  const maxWindowDays = maxWindowDaysByGranularity[granularity];
  if (maxWindowDays > 0 && windowDays > maxWindowDays) {
    return invalidRange(
      `Time range cannot exceed ${maxWindowDays} days for ${granularity} granularity`
    );
  }

  return {
    ok: true,
    value: {
      start_time,
      end_time,
      granularity,
      window_days: windowDays,
    },
  };
}

export function resolveAnalyticsQueryInput({
  query,
  legacy,
  options = {},
}: ResolveAnalyticsQueryInput): AnalyticsQueryEnvelopeValidationResult {
  const parsedQuery = parseQueryEnvelopeInput(query);
  if (!parsedQuery.ok) {
    return parsedQuery;
  }

  const queryInput = parsedQuery.value;
  if (queryInput && hasAnyQueryFields(queryInput)) {
    const validatedQuery = validateAnalyticsQuery(queryInput, options);
    if (!validatedQuery.ok) {
      return validatedQuery;
    }

    if (!hasAnyQueryFields(legacy)) {
      return validatedQuery;
    }

    const validatedLegacy = validateAnalyticsQuery(legacy, {
      ...options,
      required: false,
      defaultGranularity: validatedQuery.value.granularity,
    });
    if (!validatedLegacy.ok) {
      return validatedLegacy;
    }

    const matchResult = ensureLegacyMatchesQuery(
      validatedQuery.value,
      legacy,
      validatedLegacy.value
    );
    if (!matchResult.ok) {
      return matchResult;
    }

    return validatedQuery;
  }

  return validateAnalyticsQuery(legacy, options);
}

export const validateAnalyticsQueryEnvelope = validateAnalyticsQuery;
