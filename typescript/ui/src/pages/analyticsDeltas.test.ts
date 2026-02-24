import { describe, expect, it } from 'vitest';
import { formatKpiDelta, getDeltaTone } from './analyticsDeltas';

describe('formatKpiDelta', () => {
  it('returns undefined when previous is missing', () => {
    expect(formatKpiDelta(10, undefined, { metricFormatter: String })).toBeUndefined();
  });

  it('formats positive absolute and percentage deltas', () => {
    const result = formatKpiDelta(120, 100, { metricFormatter: (value) => value.toFixed(0) });
    expect(result).toBe('vs previous: +20 (+20.0%)');
  });

  it('formats safe divide-by-zero deltas', () => {
    const nonZero = formatKpiDelta(5, 0, { metricFormatter: (value) => value.toFixed(0) });
    const zero = formatKpiDelta(0, 0, { metricFormatter: (value) => value.toFixed(0) });

    expect(nonZero).toBe('vs previous: +5 (+n/a)');
    expect(zero).toBe('vs previous: 0 (0.0%)');
  });

  it('supports custom delta formatter (percentage points)', () => {
    const result = formatKpiDelta(0.12, 0.1, {
      metricFormatter: (value) => `${(value * 100).toFixed(1)}%`,
      deltaFormatter: (value) => `${(value * 100).toFixed(1)}pp`,
    });
    expect(result).toBe('vs previous: +2.0pp (+20.0%)');
  });
});

describe('getDeltaTone', () => {
  it('returns neutral for missing or equal values', () => {
    expect(getDeltaTone(undefined, 10)).toBe('neutral');
    expect(getDeltaTone(10, undefined)).toBe('neutral');
    expect(getDeltaTone(10, 10)).toBe('neutral');
  });

  it('returns positive/negative based on direction', () => {
    expect(getDeltaTone(11, 10)).toBe('positive');
    expect(getDeltaTone(9, 10)).toBe('negative');
  });
});
