export type AnalyticsGranularity = 'hour' | 'day';

export interface AnalyticsQueryEnvelope {
  start_time: string;
  end_time: string;
  granularity: AnalyticsGranularity;
  window_days: number;
}

export interface AnalyticsQueryEnvelopeInput {
  start_time?: string;
  end_time?: string;
  granularity?: string;
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
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return parsed;
}

function normalizeGranularity(value?: string): AnalyticsGranularity | null {
  if (!value || value === '') {
    return 'day';
  }
  if (value === 'day' || value === 'hour') {
    return value;
  }
  return null;
}

export function validateAnalyticsQueryEnvelope(
  input: AnalyticsQueryEnvelopeInput
): AnalyticsQueryEnvelopeValidationResult {
  const { start_time, end_time } = input;
  if (!start_time || !end_time) {
    return invalidRange('start_time and end_time are required');
  }

  const granularity = normalizeGranularity(input.granularity);
  if (!granularity) {
    return invalidRange("granularity must be 'day' or 'hour'");
  }

  const start = parseTimestamp(start_time);
  const end = parseTimestamp(end_time);
  if (!start || !end) {
    return invalidRange('start_time and end_time must be valid ISO dates');
  }

  if (start.getTime() >= end.getTime()) {
    return invalidRange('start_time must be before end_time');
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
