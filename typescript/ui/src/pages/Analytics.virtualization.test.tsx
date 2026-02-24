import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TransactionTable } from './Analytics';
import type { TransactionDetail } from '../types/api';

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (!globalThis.ResizeObserver) {
  (globalThis as unknown as { ResizeObserver: typeof ResizeObserverMock }).ResizeObserver = ResizeObserverMock;
}

function buildTransaction(index: number): TransactionDetail {
  return {
    record_id: `record-${index}`,
    user_id: `user-${index}`,
    created_at: new Date(Date.UTC(2024, 0, 1, 0, index)),
    is_train_eligible: true,
    is_pre_fraud: false,
    amount: 100 + index,
    is_fraudulent: index % 5 === 0,
    fraud_type: index % 5 === 0 ? 'card_testing' : '',
    is_off_hours_txn: false,
    merchant_risk_score: index % 100,
    velocity_24h: 0,
    amount_to_avg_ratio_30d: 1,
    balance_volatility_z_score: 0,
    numerical_features: {},
    categorical_features: {},
  };
}

describe('TransactionTable virtualization', () => {
  it('renders initial rows and updates rendered window on scroll', async () => {
    const rows = Array.from({ length: 600 }, (_, index) => buildTransaction(index));
    render(<TransactionTable data={rows} />);

    await waitFor(() => {
      expect(screen.getByText('user-0')).toBeInTheDocument();
    });
    expect(screen.queryByText('user-599')).toBeNull();

    const scrollContainer = screen.getByTestId('transaction-table-scroll');
    Object.defineProperty(scrollContainer, 'clientHeight', {
      value: 520,
      configurable: true,
    });

    const initialRenderedRows = scrollContainer.querySelectorAll('tbody tr[data-index]');
    expect(initialRenderedRows.length).toBeGreaterThan(0);
    expect(initialRenderedRows.length).toBeLessThan(100);
    expect(scrollContainer.scrollTop).toBe(0);

    scrollContainer.scrollTop = 58 * 350;
    fireEvent.scroll(scrollContainer);

    await waitFor(() => {
      expect(scrollContainer.scrollTop).toBe(58 * 350);
    });
    const renderedAfterScroll = scrollContainer.querySelectorAll('tbody tr[data-index]');
    expect(renderedAfterScroll.length).toBeGreaterThan(0);
    expect(renderedAfterScroll.length).toBeLessThan(100);
  });
});
