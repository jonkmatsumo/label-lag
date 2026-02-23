import { useState, useCallback } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';

export interface CursorPage<TItem> {
  items: TItem[];
  nextCursor?: string;
  total?: number;
  truncated?: boolean;
}

export interface UseCursorPaginationOptions<TItem> {
  queryKeyBase: unknown[];
  fetchPage: (params: { cursor?: string; limit: number; signal?: AbortSignal }) => Promise<CursorPage<TItem>>;
  limit?: number;
  filters?: Record<string, unknown>;
  enabled?: boolean;
}

export interface UseCursorPaginationResult<TItem> {
  data: TItem[];
  isLoading: boolean;
  isFetching: boolean;
  isError: boolean;
  error: Error | null;
  hasNextPage: boolean;
  loadNext: () => void;
  reset: () => void;
  total?: number;
  cursor?: string;
  truncated?: boolean;
}

export function useCursorPagination<TItem>(
  options: UseCursorPaginationOptions<TItem>
): UseCursorPaginationResult<TItem> {
  const { queryKeyBase, fetchPage, limit = 25, filters, enabled = true } = options;
  const [cursor, setCursor] = useState<string | undefined>(undefined);

  // Track previous filters to reset cursor on change (state-from-props pattern)
  const filtersKey = JSON.stringify(filters);
  const [prevFiltersKey, setPrevFiltersKey] = useState(filtersKey);

  if (filtersKey !== prevFiltersKey) {
    setPrevFiltersKey(filtersKey);
    setCursor(undefined);
  }

  const queryKey = [...queryKeyBase, { cursor, limit, ...filters }];

  const query = useQuery({
    queryKey,
    queryFn: ({ signal }) => {
      // Guard: never send both cursor and offset
      if (cursor && filters?.offset !== undefined) {
        throw new Error('Cannot provide both cursor and offset');
      }
      return fetchPage({ cursor, limit, signal });
    },
    placeholderData: keepPreviousData,
    enabled,
  });

  const nextCursor = query.data?.nextCursor;

  const loadNext = useCallback(() => {
    if (nextCursor) {
      setCursor(nextCursor);
    }
  }, [nextCursor]);

  const reset = useCallback(() => {
    setCursor(undefined);
  }, []);

  return {
    data: query.data?.items ?? [],
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    isError: query.isError,
    error: query.error,
    hasNextPage: !!query.data?.nextCursor,
    loadNext,
    reset,
    total: query.data?.total,
    cursor,
    truncated: query.data?.truncated,
  };
}
