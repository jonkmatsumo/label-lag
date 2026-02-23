import { useCallback, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { DateRange } from '../components/DateRangePicker';
import type { AnalyticsQueryEnvelopeParams, AnalyticsQueryGranularity } from '../api/endpoints';

const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}/;

function isValidDate(value: string | null): value is string {
  return !!value && ISO_DATE_RE.test(value);
}

function isValidGranularity(value: string | null): value is AnalyticsQueryGranularity {
  return value === 'hour' || value === 'day';
}

function getDefaultEnvelope(): AnalyticsQueryEnvelopeParams {
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - 7);
  return {
    start_time: start.toISOString().split('T')[0],
    end_time: end.toISOString().split('T')[0],
    granularity: 'day',
  };
}

export function useAnalyticsQueryEnvelope() {
  const [searchParams, setSearchParams] = useSearchParams();
  const defaults = getDefaultEnvelope();

  const rawStart = searchParams.get('start_time');
  const rawEnd = searchParams.get('end_time');
  const rawGranularity = searchParams.get('granularity');

  const startTime = isValidDate(rawStart) ? rawStart : defaults.start_time;
  const endTime = isValidDate(rawEnd) ? rawEnd : defaults.end_time;
  const granularity = isValidGranularity(rawGranularity)
    ? rawGranularity
    : defaults.granularity ?? 'day';

  useEffect(() => {
    const needsFix =
      !isValidDate(rawStart) ||
      !isValidDate(rawEnd) ||
      !isValidGranularity(rawGranularity);
    if (needsFix) {
      setSearchParams(
        {
          start_time: startTime,
          end_time: endTime,
          granularity,
        },
        { replace: true }
      );
    }
    // Run once so we normalize invalid/missing params on first render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setDateRange = useCallback(
    (range: DateRange) => {
      setSearchParams({
        start_time: range.start,
        end_time: range.end,
        granularity,
      });
    },
    [granularity, setSearchParams]
  );

  const setGranularity = useCallback(
    (nextGranularity: AnalyticsQueryGranularity) => {
      setSearchParams({
        start_time: startTime,
        end_time: endTime,
        granularity: nextGranularity,
      });
    },
    [endTime, setSearchParams, startTime]
  );

  return {
    query: {
      start_time: startTime,
      end_time: endTime,
      granularity,
    } as AnalyticsQueryEnvelopeParams,
    dateRange: {
      start: startTime,
      end: endTime,
    } as DateRange,
    granularity,
    setDateRange,
    setGranularity,
    searchParams,
    setSearchParams,
  };
}
