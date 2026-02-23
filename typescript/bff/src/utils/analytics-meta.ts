import type { AnalyticsResponseMeta } from '../types/api.js';

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
  hasData: boolean;
}

export function normalizeAnalyticsMeta(options: NormalizeAnalyticsMetaOptions): AnalyticsResponseMeta {
  const { raw, hasData } = options;
  const rawMeta = getRawMeta(raw);
  const truncated = rawMeta?.truncated === true || raw.truncated === true;

  const rawPartial =
    rawMeta?.partial ??
    raw.partial ??
    rawMeta?.is_partial ??
    raw.is_partial;
  const partial =
    typeof rawPartial === 'boolean'
      ? rawPartial
      : truncated || !hasData;

  const effectiveLimit = toFiniteNumber(rawMeta?.effective_limit ?? raw.effective_limit);

  return {
    truncated,
    partial,
    ...(effectiveLimit !== undefined ? { effective_limit: effectiveLimit } : {}),
  };
}
