import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { AnalyticsQueryGranularity } from '../api/endpoints';

interface ExplorerTimeWindow {
  start_time?: string;
  end_time?: string;
  granularity?: AnalyticsQueryGranularity;
}

export interface AnalyticsExplorerFilters {
  user_id: string;
  transaction_id: string;
  start_date: string;
  end_date: string;
  min_amount?: number;
  max_amount?: number;
  min_score?: number;
  max_score?: number;
  is_fraudulent?: boolean;
}

function parseNumber(value: string | null): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseBoolean(value: string | null): boolean | undefined {
  if (value === 'true') return true;
  if (value === 'false') return false;
  return undefined;
}

function setOrDelete(params: URLSearchParams, key: string, value: unknown): void {
  if (value === undefined || value === null || value === '') {
    params.delete(key);
    return;
  }
  params.set(key, String(value));
}

export function useAnalyticsExplorerSearchParams() {
  const [searchParams, setSearchParams] = useSearchParams();

  const timeWindow = useMemo<ExplorerTimeWindow>(() => {
    const granularity = searchParams.get('granularity');
    const normalizedGranularity =
      granularity === 'hour' || granularity === 'day' ? granularity : undefined;

    return {
      start_time: searchParams.get('start_time') ?? searchParams.get('start_date') ?? undefined,
      end_time: searchParams.get('end_time') ?? searchParams.get('end_date') ?? undefined,
      granularity: normalizedGranularity,
    };
  }, [searchParams]);

  const filters = useMemo<AnalyticsExplorerFilters>(
    () => ({
      user_id: searchParams.get('user_id') ?? '',
      transaction_id: searchParams.get('transaction_id') ?? '',
      start_date: searchParams.get('start_time') ?? searchParams.get('start_date') ?? '',
      end_date: searchParams.get('end_time') ?? searchParams.get('end_date') ?? '',
      min_amount: parseNumber(searchParams.get('min_amount')),
      max_amount: parseNumber(searchParams.get('max_amount')),
      min_score: parseNumber(searchParams.get('min_score')),
      max_score: parseNumber(searchParams.get('max_score')),
      is_fraudulent: parseBoolean(searchParams.get('is_fraudulent')),
    }),
    [searchParams]
  );

  const updateFilters = useCallback(
    (updates: Partial<AnalyticsExplorerFilters>) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          setOrDelete(next, 'user_id', updates.user_id);
          setOrDelete(next, 'transaction_id', updates.transaction_id);
          setOrDelete(next, 'min_amount', updates.min_amount);
          setOrDelete(next, 'max_amount', updates.max_amount);
          setOrDelete(next, 'min_score', updates.min_score);
          setOrDelete(next, 'max_score', updates.max_score);
          setOrDelete(next, 'is_fraudulent', updates.is_fraudulent);

          if ('start_date' in updates) {
            setOrDelete(next, 'start_time', updates.start_date);
          }
          if ('end_date' in updates) {
            setOrDelete(next, 'end_time', updates.end_date);
          }

          // Keep URL canonical; explorer uses start_time/end_time.
          next.delete('start_date');
          next.delete('end_date');
          return next;
        },
        { replace: true }
      );
    },
    [setSearchParams]
  );

  const setTimeWindow = useCallback(
    (updates: ExplorerTimeWindow) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if ('start_time' in updates) {
            setOrDelete(next, 'start_time', updates.start_time);
          }
          if ('end_time' in updates) {
            setOrDelete(next, 'end_time', updates.end_time);
          }
          if ('granularity' in updates) {
            setOrDelete(next, 'granularity', updates.granularity);
          }
          next.delete('start_date');
          next.delete('end_date');
          return next;
        },
        { replace: true }
      );
    },
    [setSearchParams]
  );

  return {
    filters,
    timeWindow,
    updateFilters,
    setTimeWindow,
  };
}
