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

export function validateAnalyticsQuery(
  input: AnalyticsQueryEnvelopeInput,
  options: ValidateAnalyticsQueryOptions = {}
): AnalyticsQueryEnvelopeValidationResult {
  const required = options.required ?? true;
  const defaultGranularity = options.defaultGranularity ?? 'day';
  const startField = options.startField ?? 'start_time';
  const endField = options.endField ?? 'end_time';
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
  const maxWindowDays = MAX_DAYS_BY_GRANULARITY[granularity];
  if (windowDays > maxWindowDays) {
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

export const validateAnalyticsQueryEnvelope = validateAnalyticsQuery;
