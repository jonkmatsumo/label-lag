import { beforeEach, describe, expect, it, vi } from 'vitest';
import { analyticsApi, buildAnalyticsQueryEnvelope } from './endpoints';

globalThis.fetch = vi.fn();

describe('analyticsApi compare_to_previous propagation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => ({}),
    } as Response);
  });

  it('includes compare_to_previous=true for KPI requests', async () => {
    await analyticsApi.getKpis({
      start_time: '2024-01-01',
      end_time: '2024-01-07',
      granularity: 'day',
      compare_to_previous: true,
    });

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    const parsed = new URL(String(url), 'http://localhost');
    expect(parsed.pathname).toBe('/api/bff/v1/kpis');
    expect(parsed.searchParams.get('start_time')).toBe('2024-01-01');
    expect(parsed.searchParams.get('end_time')).toBe('2024-01-07');
    expect(parsed.searchParams.get('granularity')).toBe('day');
    expect(parsed.searchParams.get('compare_to_previous')).toBe('true');
  });

  it('includes compare_to_previous=true for volume requests', async () => {
    await analyticsApi.getVolume({
      start_time: '2024-01-01',
      end_time: '2024-01-07',
      granularity: 'hour',
      compare_to_previous: true,
    });

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    const parsed = new URL(String(url), 'http://localhost');
    expect(parsed.pathname).toBe('/api/bff/v1/volume');
    expect(parsed.searchParams.get('granularity')).toBe('hour');
    expect(parsed.searchParams.get('compare_to_previous')).toBe('true');
  });

  it('does not forward compare_to_previous to confusion-matrix endpoint', async () => {
    await analyticsApi.getConfusionMatrix({
      start_time: '2024-01-01',
      end_time: '2024-01-07',
      granularity: 'day',
      compare_to_previous: true,
    });

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    const parsed = new URL(String(url), 'http://localhost');
    expect(parsed.pathname).toBe('/api/bff/v1/analytics/confusion-matrix');
    expect(parsed.searchParams.get('compare_to_previous')).toBeNull();
  });
});

describe('buildAnalyticsQueryEnvelope', () => {
  it('maps compareToPrevious to compare_to_previous', () => {
    const envelope = buildAnalyticsQueryEnvelope({
      start: '2024-01-01',
      end: '2024-01-07',
      granularity: 'day',
      compareToPrevious: true,
    });

    expect(envelope).toEqual({
      start_time: '2024-01-01',
      end_time: '2024-01-07',
      granularity: 'day',
      compare_to_previous: true,
    });
  });
});
