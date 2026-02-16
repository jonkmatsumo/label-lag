import { renderHook, waitFor } from '@testing-library/react';
import { useCursorPagination } from './useCursorPagination';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

// Mock dependencies
const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
        },
    },
});

const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient} > {children} </QueryClientProvider>
);

describe('useCursorPagination', () => {
    beforeEach(() => {
        queryClient.clear();
        vi.restoreAllMocks();
    });

    it('should initialize with empty data', async () => {
        const fetchPage = vi.fn().mockResolvedValue({ items: [], nextCursor: null });
        const { result } = renderHook(() => useCursorPagination({
            queryKeyBase: ['test'],
            fetchPage,
            limit: 10,
        }), { wrapper });

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(result.current.data).toEqual([]);
        expect(result.current.hasNextPage).toBe(false);
    });

    it('should fetch first page and set data', async () => {
        const mockData = [{ id: 1 }, { id: 2 }];
        const fetchPage = vi.fn().mockResolvedValue({
            items: mockData,
            nextCursor: 'cursor-1',
            total: 100
        });

        const { result } = renderHook(() => useCursorPagination({
            queryKeyBase: ['test-2'],
            fetchPage,
            limit: 10,
        }), { wrapper });

        await waitFor(() => expect(result.current.isLoading).toBe(false));

        expect(result.current.data).toEqual(mockData);
        expect(result.current.hasNextPage).toBe(true);
        expect(result.current.total).toBe(100);
        expect(fetchPage).toHaveBeenCalledWith({ cursor: undefined, limit: 10 });
    });

    it('should fetch next page when loadNext is called', async () => {
        const page1 = [{ id: 1 }];
        const page2 = [{ id: 2 }];

        const fetchPage = vi.fn()
            .mockResolvedValueOnce({ items: page1, nextCursor: 'cursor-1' })
            .mockResolvedValueOnce({ items: page2, nextCursor: null });

        const { result } = renderHook(() => useCursorPagination({
            queryKeyBase: ['test-3'],
            fetchPage,
            limit: 10,
        }), { wrapper });

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(result.current.data).toEqual(page1);

        await result.current.loadNext();

        await waitFor(() => expect(result.current.data).toEqual(page2));
        expect(fetchPage).toHaveBeenCalledTimes(2);
        expect(fetchPage).toHaveBeenLastCalledWith({ cursor: 'cursor-1', limit: 10 });
    });
});
