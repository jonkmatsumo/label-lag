import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';

type FetchResponse = { label: string };
type Fetcher = (filter: string, signal?: AbortSignal) => Promise<FetchResponse>;

function FilterDrivenQuery({ filter, fetcher }: { filter: string; fetcher: Fetcher }) {
  const query = useQuery({
    queryKey: ['analytics-filter', filter],
    queryFn: ({ signal }) => fetcher(filter, signal),
    retry: false,
  });

  return <div data-testid="query-value">{query.data?.label ?? 'loading'}</div>;
}

describe('TanStack Query cancellation', () => {
  it('cancels stale requests when filters change rapidly', async () => {
    const events: string[] = [];
    const fetcher = vi.fn((filter: string, signal?: AbortSignal) => {
      return new Promise<FetchResponse>((resolve, reject) => {
        const delayMs = filter === 'slow-range' ? 80 : 10;
        const timer = setTimeout(() => {
          events.push(`resolved:${filter}`);
          resolve({ label: filter });
        }, delayMs);

        signal?.addEventListener(
          'abort',
          () => {
            clearTimeout(timer);
            events.push(`aborted:${filter}`);
            reject(new DOMException('Aborted', 'AbortError'));
          },
          { once: true }
        );
      });
    });

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    const { rerender } = render(
      <QueryClientProvider client={queryClient}>
        <FilterDrivenQuery filter="slow-range" fetcher={fetcher} />
      </QueryClientProvider>
    );

    rerender(
      <QueryClientProvider client={queryClient}>
        <FilterDrivenQuery filter="fast-range" fetcher={fetcher} />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('query-value')).toHaveTextContent('fast-range');
    });

    expect(events).toContain('aborted:slow-range');
    expect(events).toContain('resolved:fast-range');
    expect(events).not.toContain('resolved:slow-range');
  });
});
