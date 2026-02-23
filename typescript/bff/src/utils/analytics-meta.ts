import type { AnalyticsResponseMeta, PartialReason } from '../types/api.js';

const VALID_PARTIAL_REASONS: ReadonlySet<PartialReason> = new Set([
  'TIMEOUT',
  'ROW_LIMIT',
  'UPSTREAM_ERROR',
  'EMPTY',
  'UNKNOWN',
]);

function toIsoOrFallback(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toISOString();
}

function normalizePartialReason(value: unknown): PartialReason {
  if (typeof value !== 'string') return 'UNKNOWN';
  const normalized = value.toUpperCase() as PartialReason;
  return VALID_PARTIAL_REASONS.has(normalized) ? normalized : 'UNKNOWN';
}

function getRawMeta(raw: Record<string, unknown>): Record<string, unknown> | undefined {
  if (typeof raw.meta === 'object' && raw.meta !== null) {
    return raw.meta as Record<string, unknown>;
  }
  return undefined;
}

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

interface NormalizeAnalyticsMetaOptions {
  raw: Record<string, unknown>;
  startTime: string;
  endTime: string;
  hasData: boolean;
}

export function normalizeAnalyticsMeta(options: NormalizeAnalyticsMetaOptions): AnalyticsResponseMeta {
  const { raw, startTime, endTime, hasData } = options;
  const rawMeta = getRawMeta(raw);

  const rawReason = rawMeta?.partial_reason ?? raw.partial_reason;
  let partialReason = normalizePartialReason(rawReason);

  const rawIsPartial = rawMeta?.is_partial ?? raw.is_partial;
  const explicitIsPartial = typeof rawIsPartial === 'boolean' ? rawIsPartial : undefined;
  const rawPartial = rawMeta?.partial ?? raw.partial;
  const explicitPartial = typeof rawPartial === 'boolean' ? rawPartial : undefined;
  const truncated = rawMeta?.truncated === true || raw.truncated === true;

  if (partialReason === 'UNKNOWN') {
    if (truncated) {
      partialReason = 'ROW_LIMIT';
    } else if (!hasData) {
      partialReason = 'EMPTY';
    }
  }

  const sampleRateCandidate = rawMeta?.sample_rate ?? raw.sample_rate;
  const sampleRate =
    typeof sampleRateCandidate === 'number' && Number.isFinite(sampleRateCandidate)
      ? sampleRateCandidate
      : undefined;

  const isPartial = explicitIsPartial ?? partialReason !== 'UNKNOWN';
  const partial = explicitPartial ?? isPartial;
  const effectiveLimit = toFiniteNumber(rawMeta?.effective_limit ?? raw.effective_limit);

  return {
    time_range: {
      start: toIsoOrFallback(startTime),
      end: toIsoOrFallback(endTime),
    },
    is_partial: isPartial,
    partial_reason: partialReason,
    truncated,
    partial,
    ...(sampleRate !== undefined ? { sample_rate: sampleRate } : {}),
    ...(effectiveLimit !== undefined ? { effective_limit: effectiveLimit } : {}),
  };
}
