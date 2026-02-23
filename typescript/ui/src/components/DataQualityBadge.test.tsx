import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DataQualityBadge } from './DataQualityBadge';

describe('DataQualityBadge', () => {
  it('renders partial indicator from meta.partial', () => {
    render(
      <DataQualityBadge
        meta={{
          truncated: true,
          partial: true,
          effective_limit: 500,
        }}
      />
    );

    expect(screen.getByTestId('data-quality-badge')).toHaveTextContent('Partial (truncated)');
    expect(screen.getByTestId('data-quality-badge')).toHaveTextContent('limit 500');
  });

  it('does not render when meta.partial is false', () => {
    render(
      <DataQualityBadge
        meta={{
          truncated: false,
          partial: false,
        }}
      />
    );

    expect(screen.queryByTestId('data-quality-badge')).toBeNull();
  });
});
