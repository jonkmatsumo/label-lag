import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DataQualityBadge } from './DataQualityBadge';

describe('DataQualityBadge', () => {
  it('renders partial indicator when is_partial is true', () => {
    render(
      <DataQualityBadge
        meta={{
          time_range: {
            start: '2024-01-01T00:00:00.000Z',
            end: '2024-01-07T00:00:00.000Z',
          },
          is_partial: true,
          partial_reason: 'ROW_LIMIT',
        }}
      />
    );

    expect(screen.getByTestId('data-quality-badge')).toHaveTextContent('Partial (row limit)');
  });

  it('does not render when is_partial is false', () => {
    render(
      <DataQualityBadge
        meta={{
          time_range: {
            start: '2024-01-01T00:00:00.000Z',
            end: '2024-01-07T00:00:00.000Z',
          },
          is_partial: false,
          partial_reason: 'UNKNOWN',
        }}
      />
    );

    expect(screen.queryByTestId('data-quality-badge')).toBeNull();
  });
});
