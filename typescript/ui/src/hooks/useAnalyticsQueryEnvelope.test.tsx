import type { ReactNode } from 'react';
import { act, renderHook } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { useAnalyticsQueryEnvelope } from './useAnalyticsQueryEnvelope';

function createWrapper(initialEntry: string) {
  return ({ children }: { children: ReactNode }) => (
    <MemoryRouter initialEntries={[initialEntry]}>{children}</MemoryRouter>
  );
}

describe('useAnalyticsQueryEnvelope', () => {
  it('hydrates compare_to_previous from URL params', () => {
    const wrapper = createWrapper('/analytics?start_time=2024-01-01&end_time=2024-01-07&granularity=day&compare_to_previous=true');
    const { result } = renderHook(() => useAnalyticsQueryEnvelope(), { wrapper });

    expect(result.current.compareToPrevious).toBe(true);
    expect(result.current.query.compare_to_previous).toBe(true);
  });

  it('updates compare_to_previous when toggled', () => {
    const wrapper = createWrapper('/analytics?start_time=2024-01-01&end_time=2024-01-07&granularity=day');
    const { result } = renderHook(() => useAnalyticsQueryEnvelope(), { wrapper });

    expect(result.current.compareToPrevious).toBe(false);
    expect(result.current.searchParams.get('compare_to_previous')).toBeNull();

    act(() => {
      result.current.setCompareToPrevious(true);
    });

    expect(result.current.compareToPrevious).toBe(true);
    expect(result.current.searchParams.get('compare_to_previous')).toBe('true');

    act(() => {
      result.current.setCompareToPrevious(false);
    });

    expect(result.current.compareToPrevious).toBe(false);
    expect(result.current.searchParams.get('compare_to_previous')).toBeNull();
  });
});
