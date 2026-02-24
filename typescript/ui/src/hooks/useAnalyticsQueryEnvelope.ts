import { useCallback, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { DateRange } from '../components/DateRangePicker';
import { buildAnalyticsQueryEnvelope } from '../api/endpoints';
import type { AnalyticsQueryEnvelopeParams, AnalyticsQueryGranularity } from '../api/endpoints';

const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}/;

function isValidDate(value: string | null): value is string {
  return !!value && ISO_DATE_RE.test(value);
}

function isValidGranularity(value: string | null): value is AnalyticsQueryGranularity {
  return value === 'hour' || value === 'day';
}

function isValidCompareToPrevious(value: string | null): boolean {
  return value === null || value === 'true' || value === 'false';
}

function getDefaultEnvelope(): AnalyticsQueryEnvelopeParams {
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - 7);
  return buildAnalyticsQueryEnvelope({
    start: start.toISOString().split('T')[0],
    end: end.toISOString().split('T')[0],
    granularity: 'day',
  });
}

export function useAnalyticsQueryEnvelope() {
  const [searchParams, setSearchParams] = useSearchParams();
  const defaults = getDefaultEnvelope();

  const rawStart = searchParams.get('start_time');
  const rawEnd = searchParams.get('end_time');
  const rawGranularity = searchParams.get('granularity');
  const rawCompareToPrevious = searchParams.get('compare_to_previous');

  const startTime = isValidDate(rawStart) ? rawStart : defaults.start_time;
  const endTime = isValidDate(rawEnd) ? rawEnd : defaults.end_time;
  const granularity = isValidGranularity(rawGranularity)
    ? rawGranularity
    : defaults.granularity ?? 'day';
  const compareToPrevious = rawCompareToPrevious === 'true';

  useEffect(() => {
    const needsFix =
      !isValidDate(rawStart) ||
      !isValidDate(rawEnd) ||
      !isValidGranularity(rawGranularity) ||
      !isValidCompareToPrevious(rawCompareToPrevious);
    if (needsFix) {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set('start_time', startTime);
          next.set('end_time', endTime);
          next.set('granularity', granularity);
          next.delete('compare_to_previous');
          return next;
        },
        { replace: true }
      );
    }
    // Run once so we normalize invalid/missing params on first render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setDateRange = useCallback(
    (range: DateRange) => {
      const next = buildAnalyticsQueryEnvelope({
        start: range.start,
        end: range.end,
        granularity,
        compareToPrevious,
      });
      setSearchParams((prev) => {
        const params = new URLSearchParams(prev);
        params.set('start_time', next.start_time);
        params.set('end_time', next.end_time);
        params.set('granularity', next.granularity ?? 'day');
        if (next.compare_to_previous) {
          params.set('compare_to_previous', 'true');
        } else {
          params.delete('compare_to_previous');
        }
        return params;
      });
    },
    [compareToPrevious, granularity, setSearchParams]
  );

  const setGranularity = useCallback(
    (nextGranularity: AnalyticsQueryGranularity) => {
      const next = buildAnalyticsQueryEnvelope({
        start: startTime,
        end: endTime,
        granularity: nextGranularity,
        compareToPrevious,
      });
      setSearchParams((prev) => {
        const params = new URLSearchParams(prev);
        params.set('start_time', next.start_time);
        params.set('end_time', next.end_time);
        params.set('granularity', next.granularity ?? 'day');
        if (next.compare_to_previous) {
          params.set('compare_to_previous', 'true');
        } else {
          params.delete('compare_to_previous');
        }
        return params;
      });
    },
    [compareToPrevious, endTime, setSearchParams, startTime]
  );

  const setCompareToPrevious = useCallback(
    (enabled: boolean) => {
      setSearchParams((prev) => {
        const params = new URLSearchParams(prev);
        if (enabled) {
          params.set('compare_to_previous', 'true');
        } else {
          params.delete('compare_to_previous');
        }
        return params;
      });
    },
    [setSearchParams]
  );

  return {
    query: buildAnalyticsQueryEnvelope({
      start: startTime,
      end: endTime,
      granularity,
      compareToPrevious,
    }) as AnalyticsQueryEnvelopeParams,
    dateRange: {
      start: startTime,
      end: endTime,
    } as DateRange,
    granularity,
    compareToPrevious,
    setDateRange,
    setGranularity,
    setCompareToPrevious,
    searchParams,
    setSearchParams,
  };
}
