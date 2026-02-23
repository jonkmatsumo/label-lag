import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { analyticsApi, buildAnalyticsQueryEnvelope } from '../api';
import type { RecentAlert, TransactionSearchRequest, TransactionDetail } from '../types/api';
import {
  Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart
} from 'recharts';
import { Search, ChevronDown, ChevronRight, ShieldCheck } from 'lucide-react';
import { ErrorBanner } from '../components/ErrorBanner';
import { DateRangePicker, KpiCard } from '../components';
import { DataQualityBadge } from '../components/DataQualityBadge';
import { useTenant } from '../hooks/useTenant';
import { useCursorPagination } from '../hooks/useCursorPagination';
import { useAnalyticsSearchParams } from '../hooks/useAnalyticsSearchParams';
import { useAnalyticsQueryEnvelope } from '../hooks/useAnalyticsQueryEnvelope';
import type { DateRange } from '../components/DateRangePicker';

export function Analytics() {
  const [daysFilter] = useState(30);
  const { tenantId } = useTenant();
  const {
    dateRange,
    granularity,
    setDateRange,
    setGranularity,
    searchParams,
    setSearchParams,
  } = useAnalyticsQueryEnvelope();
  const analyticsQuery = useMemo(
    () =>
      buildAnalyticsQueryEnvelope({
        start: dateRange.start,
        end: dateRange.end,
        granularity,
      }),
    [dateRange.end, dateRange.start, granularity]
  );

  // ── Handlers ─────────────────────────────────────────────────────────────────
  const handleDateRangeChange = (range: DateRange) => {
    setDateRange(range);
  };

  const handleGranularityChange = (g: 'hour' | 'day') => {
    setGranularity(g);
  };

  // Fetch performance KPIs — signal enables TanStack Query to cancel stale requests
  const kpisQuery = useQuery({
    queryKey: ['analytics', tenantId, 'kpis', dateRange.start, dateRange.end, granularity],
    queryFn: ({ signal }) => analyticsApi.getKpis({
      ...analyticsQuery,
      signal,
    }),
    staleTime: 30_000,
  });

  // Fetch volume timeseries — signal enables TanStack Query to cancel stale requests
  const volumeQuery = useQuery({
    queryKey: ['analytics', tenantId, 'volume', dateRange.start, dateRange.end, granularity],
    queryFn: ({ signal }) => analyticsApi.getVolume({
      ...analyticsQuery,
      signal,
    }),
    staleTime: 30_000,
  });

  // Fetch confusion matrix — signal enables TanStack Query to cancel stale requests
  const confusionMatrixQuery = useQuery({
    queryKey: ['analytics', tenantId, 'confusion-matrix', dateRange.start, dateRange.end],
    queryFn: ({ signal }) => analyticsApi.getConfusionMatrix({
      ...analyticsQuery,
      signal,
    }),
    staleTime: 30_000,
  });

  // Fetch overview metrics (legacy/static)
  const overviewQuery = useQuery({
    queryKey: ['analytics', tenantId, 'overview'],
    queryFn: ({ signal }) => analyticsApi.getOverview(daysFilter, signal),
  });

  // Fetch recent alerts for FPR calculation
  const alertsQuery = useQuery({
    queryKey: ['analytics', tenantId, 'alerts'],
    queryFn: ({ signal }) => analyticsApi.getRecentAlerts(20, signal),
  });

  const hourlyWindowDays = useMemo(() => {
    if (granularity !== 'hour') {
      return 0;
    }
    const start = new Date(dateRange.start);
    const end = new Date(dateRange.end);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
      return 0;
    }
    return (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24);
  }, [dateRange.end, dateRange.start, granularity]);

  const showHourlyWindowHint = granularity === 'hour' && hourlyWindowDays > 60;

  const toNumber = (value: number | string | undefined | null) => {
    const parsed = typeof value === 'string' ? Number(value) : value ?? 0;
    return Number.isFinite(parsed) ? parsed : 0;
  };

  const formatNumber = (value: number | string | undefined | null) => {
    const n = toNumber(value);
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toLocaleString();
  };

  const formatPercent = (n: number) => `${(n * 100).toFixed(2)}%`;

  const kpis = kpisQuery.data;
  const volume = volumeQuery.data;
  const confusionMatrix = confusionMatrixQuery.data;

  return (
    <div className="page">
      <h2>Historical Analytics</h2>
      <p>Dataset overview and fraud trends</p>

      {/* KPI Dashboard Controls */}
      <div className="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-3">
        <DateRangePicker onChange={handleDateRangeChange} />
        <div className="btn-group btn-group-sm">
          <button
            className={`btn ${granularity === 'hour' ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => handleGranularityChange('hour')}
          >
            Hourly
          </button>
          <button
            className={`btn ${granularity === 'day' ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => handleGranularityChange('day')}
          >
            Daily
          </button>
        </div>
      </div>
      {showHourlyWindowHint && (
        <div className="alert alert-warning py-2 px-3 small mb-4">
          Hourly windows over 60 days may be rejected by the server. Use daily granularity or shorten the range.
        </div>
      )}
      {/* KPI Cards */}
      <div className="row g-3 mb-4">
        <div className="col-md">
          <div style={{ minHeight: '96px' }}>
            <KpiCard
              label="Total Decisions"
              value={kpis?.total_decisions ?? 0}
              loading={kpisQuery.isLoading}
              error={kpisQuery.error}
              formatter={(val) => formatNumber(val as string | number | null | undefined)}
              badge={<DataQualityBadge meta={kpis?.meta} />}
            />
          </div>
        </div>
        <div className="col-md">
          <div style={{ minHeight: '96px' }}>
            <KpiCard
              label="Total Alerts"
              value={kpis?.total_alerts ?? 0}
              loading={kpisQuery.isLoading}
              error={kpisQuery.error}
              formatter={(val) => formatNumber(val as string | number | null | undefined)}
              badge={<DataQualityBadge meta={kpis?.meta} />}
            />
          </div>
        </div>
        <div className="col-md">
          <div style={{ minHeight: '96px' }}>
            <KpiCard
              label="Alert Rate"
              value={kpis?.alert_rate ?? 0}
              loading={kpisQuery.isLoading}
              error={kpisQuery.error}
              formatter={(val) => `${(Number(val) * 100).toFixed(1)}%`}
              badge={<DataQualityBadge meta={kpis?.meta} />}
            />
          </div>
        </div>
        <div className="col-md">
          <div style={{ minHeight: '96px' }}>
            <KpiCard
              label="Avg Risk Score"
              value={kpis?.avg_score ?? 0}
              loading={kpisQuery.isLoading}
              error={kpisQuery.error}
              formatter={(val) => Number(val).toFixed(2)}
              badge={<DataQualityBadge meta={kpis?.meta} />}
            />
          </div>
        </div>
        <div className="col-md">
          <div style={{ minHeight: '96px' }}>
            <KpiCard
              label="Rules Fired"
              value={kpis?.rules_fired_total ?? 0}
              loading={kpisQuery.isLoading}
              error={kpisQuery.error}
              formatter={(val) => formatNumber(val as string | number | null | undefined)}
              badge={<DataQualityBadge meta={kpis?.meta} />}
            />
          </div>
        </div>
      </div>

      {/* Model Performance Card (Confusion Matrix) */}
      <div className="card h-100 shadow-sm border-0 mb-4">
        <div className="card-body p-4">
          <div className="d-flex justify-content-between align-items-center mb-4">
            <h5 className="card-title mb-0 fw-bold">Model Precision (Confusion Matrix)</h5>
            <DataQualityBadge meta={confusionMatrix?.meta} />
          </div>
          {confusionMatrixQuery.isLoading ? (
            <div className="d-flex align-items-center justify-content-center h-100" style={{ minHeight: '100px' }}>
              <div className="spinner-border spinner-border-sm text-success me-2" />
              <span className="text-muted small">Loading model performance…</span>
            </div>
          ) : confusionMatrixQuery.isError ? (
            <div className="text-danger small p-2">Failed to load model performance metrics</div>
          ) : confusionMatrix?.insufficient_labels ? (
            <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '100px' }}>
              <div className="text-center text-muted">
                <ShieldCheck size={28} className="mb-2 opacity-25" />
                <div className="small">Insufficient labeled data for the selected period</div>
              </div>
            </div>
          ) : confusionMatrix ? (
            <div className="row g-4 align-items-start">
              {/* Precision / Recall / F1 */}
              <div className="col-md-6">
                <div className="row g-2">
                  {([
                    { label: 'Precision', value: confusionMatrixQuery.data.precision },
                    { label: 'Recall', value: confusionMatrixQuery.data.recall },
                    { label: 'F1 Score', value: confusionMatrixQuery.data.f1_score },
                  ] as { label: string; value: number }[]).map(({ label, value }) => (
                    <div key={label} className="col-4">
                      <div className="text-center p-2 bg-light rounded">
                        <div className="fw-bold fs-6">{(value * 100).toFixed(1)}%</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>{label}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {/* TP / FP / TN / FN compact 2×2 table */}
              <div className="col-md-6">
                <div className="row g-1" style={{ maxWidth: '280px' }}>
                  {([
                    { label: 'TP', value: confusionMatrixQuery.data.true_positives, bg: 'bg-success-subtle text-success' },
                    { label: 'FP', value: confusionMatrixQuery.data.false_positives, bg: 'bg-danger-subtle text-danger' },
                    { label: 'FN', value: confusionMatrixQuery.data.false_negatives, bg: 'bg-warning-subtle text-warning' },
                    { label: 'TN', value: confusionMatrixQuery.data.true_negatives, bg: 'bg-success-subtle text-success' },
                  ] as { label: string; value: number; bg: string }[]).map(({ label, value, bg }) => (
                    <div key={label} className="col-6">
                      <div className={`text-center p-2 rounded border ${bg}`}>
                        <div className="fw-bold">{formatNumber(value)}</div>
                        <div style={{ fontSize: '0.7rem' }}>{label}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>

      {/* Volume Chart */}
      <div className="card h-100 shadow-sm border-0 mb-4">
        <div className="card-body p-4">
          <div className="d-flex justify-content-between align-items-center mb-4">
            <h5 className="card-title mb-0 fw-bold">Anomaly Volume</h5>
            <div className="d-flex gap-2 align-items-center">
              <DataQualityBadge meta={volume?.meta} />
              <select
                className="form-select form-select-sm"
                value={granularity}
                onChange={(e) => handleGranularityChange(e.target.value as 'hour' | 'day')}
                disabled={volumeQuery.isLoading}
              >
                <option value="hour">Hourly</option>
                <option value="day">Daily</option>
              </select>
            </div>
          </div>
          {volumeQuery.isLoading ? (
            <div className="d-flex align-items-center justify-content-center h-100">
              <div className="spinner-border spinner-border-sm text-primary me-2" /> Loading volume data...
            </div>
          ) : volumeQuery.isError ? (
            <div className="text-danger p-4">Failed to load volume chart</div>
          ) : volume?.points && volume.points.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart
                data={[...volume.points].sort((a, b) =>
                  new Date(a.timestamp ?? 0).getTime() - new Date(b.timestamp ?? 0).getTime()
                )}
                onClick={(eventState) => {
                  const activePayload = (
                    eventState as
                      | {
                          activePayload?: Array<{
                            payload?: { timestamp?: string };
                          }>;
                        }
                      | undefined
                  )?.activePayload;
                  const timestamp = activePayload?.[0]?.payload?.timestamp;
                  if (timestamp) {
                    const date = new Date(timestamp);
                    const dateStr = date.toISOString().split('T')[0];
                    // If granularity is hour, we filter for that specific day or range
                    // For now, let's drill down by setting both start/end to this day
                    setSearchParams({
                      ...Object.fromEntries(searchParams),
                      start_date: dateStr,
                      end_date: dateStr
                    });
                  }
                }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <XAxis
                  dataKey="timestamp"
                  fontSize={10}
                  tickFormatter={(val) => {
                    const d = new Date(val);
                    return granularity === 'hour'
                      ? `${d.getUTCHours().toString().padStart(2, '0')}:00`
                      : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                  }}
                />
                <YAxis fontSize={10} />
                <Tooltip
                  labelFormatter={(label) => new Date(label).toLocaleString()}
                />
                <Legend />
                <Bar dataKey="count" name="Decisions" fill="#cfe2ff" radius={[4, 4, 0, 0]} />
                <Line type="monotone" dataKey="alerts" name="Alerts" stroke="#dc3545" strokeWidth={2} dot={{ r: 3 }} />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div className="d-flex align-items-center justify-content-center h-100 text-muted">
              No data available for the selected range
            </div>
          )}
        </div>
      </div>

      {/* Overview Metrics */}
      <div className="card shadow-sm border-0 mb-4">
        <div className="card-header bg-white border-bottom py-3">
          <h3 className="card-title h6 fw-bold mb-0">Dataset Overview</h3>
        </div>
        {overviewQuery.isLoading ? (
          <div className="loading p-4">Loading overview metrics...</div>
        ) : overviewQuery.isError ? (
          <ErrorBanner error={overviewQuery.error} title="Failed to load overview" className="m-4 alert alert-danger" />
        ) : overviewQuery.data ? (
          <div className="metrics-grid p-4">
            <div className="metric-card shadow-sm border-0">
              <div className="metric-label">Total Records</div>
              <div className="metric-value">{formatNumber(overviewQuery.data.total_records)}</div>
            </div>
            <div className="metric-card shadow-sm border-0">
              <div className="metric-label">Fraud Count</div>
              <div className="metric-value text-danger">{formatNumber(overviewQuery.data.fraud_records)}</div>
            </div>
            <div className="metric-card shadow-sm border-0">
              <div className="metric-label">Fraud Rate</div>
              <div className="metric-value">{formatPercent(overviewQuery.data.fraud_rate)}</div>
            </div>
            <div className="metric-card shadow-sm border-0">
              <div className="metric-label">Est. FPR</div>
              <div className="metric-value text-warning">
                {alertsQuery.data?.alerts && overviewQuery.data ?
                  `${((alertsQuery.data.alerts.filter((a: RecentAlert) => a.computed_risk_score >= 80).length / Math.max(toNumber(overviewQuery.data.total_records) * 0.05, 1)) * 100).toFixed(1)}%`
                  : '--'}
              </div>
              <div className="small text-muted mt-1" style={{ fontSize: '0.7em' }}>False Positive Rate (Est)</div>
            </div>
          </div>
        ) : null}
        {overviewQuery.data && (
          <div className="date-range-info px-4 pb-3 small text-muted">
            Data range: {overviewQuery.data.min_transaction_timestamp
              ? new Date(overviewQuery.data.min_transaction_timestamp).toLocaleDateString()
              : '--'} - {overviewQuery.data.max_transaction_timestamp
                ? new Date(overviewQuery.data.max_transaction_timestamp).toLocaleDateString()
                : '--'}
          </div>
        )}
      </div>

      <TransactionExplorer />
    </div>
  );
}


function TransactionExplorer() {
  const { tenantId } = useTenant();
  const { filters, updateParams } = useAnalyticsSearchParams();

  const pagination = useCursorPagination<TransactionDetail>({
    queryKeyBase: ['analytics', tenantId, 'search'],
    fetchPage: ({ cursor, limit, signal }) =>
      analyticsApi.searchTransactions({
        ...filters,
        tenant_id: tenantId,
        cursor,
        limit,
        include_features: false, // Prevent overfetching on list view
        signal,
      }).then(res => ({
        items: res.items,
        nextCursor: res.next_cursor,
        truncated: res.meta.truncated,
      })),
    limit: 20,
    filters,
  });

  // No longer use pagination.total since backend doesn't return it for search
  // Instead, we show 'Truncated' if the server flagged it or we hit the next page state

  return (
    <div className="card shadow-sm border-0 mt-4 mb-5">
      <div className="card-header bg-white border-bottom py-3">
        <h3 className="card-title h6 fw-bold mb-0">Transaction Explorer</h3>
      </div>
      <div className="card-body">
        <TransactionFilters filters={filters} onChange={updateParams} />

        {/* Truncation warning */}
        {pagination.truncated && (
          <div className="alert alert-warning py-2 small d-flex align-items-center mb-3">
            <ShieldCheck size={16} className="me-2 flex-shrink-0" />
            <div><strong>Results truncated by server.</strong> Request limit was automatically capped at 500 for performance. Use filters to narrow results.</div>
          </div>
        )}

        {pagination.isLoading ? (
          <div className="text-center p-5"><div className="spinner-border text-primary" /></div>
        ) : pagination.isError ? (
          <ErrorBanner error={pagination.error} title="Error loading transactions" className="alert alert-danger" />
        ) : (
          <>
            <TransactionTable data={pagination.data} />

            {/* Pagination Controls */}
            {pagination.data.length > 0 && (
              <div className="d-flex justify-content-between align-items-center mt-3">
                <div className="small text-muted">
                  {pagination.data.length > 0 ? `Showing page results` : `No transactions`}
                  {pagination.truncated ? ' (Capped)' : ''}
                </div>
                <div className="btn-group btn-group-sm">
                  {/* Since cursor pagination doesn't naturally support 'Previous' easily without a stack, we'll offer Reset and Next */}
                  <button
                    className="btn btn-outline-secondary"
                    disabled={!pagination.cursor}
                    onClick={() => pagination.reset()}
                  >
                    Reset to First Page
                  </button>
                  <button
                    className="btn btn-outline-secondary"
                    disabled={!pagination.hasNextPage}
                    onClick={() => pagination.loadNext()}
                  >
                    Next Page
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function TransactionFilters({ filters, onChange }: { filters: Partial<TransactionSearchRequest>, onChange: (f: Partial<TransactionSearchRequest>) => void }) {
  const [localFilters, setLocalFilters] = useState(filters);
  const [expanded, setExpanded] = useState(false);

  const applyFilters = (e: React.FormEvent) => {
    e.preventDefault();
    onChange({ ...localFilters });
  };

  return (
    <form onSubmit={applyFilters} className="mb-4 p-3 bg-light rounded border">
      <div className="row g-3">
        <div className="col-md-3">
          <label className="form-label small fw-bold">User ID</label>
          <input
            type="text" className="form-control form-control-sm"
            value={localFilters.user_id || ''}
            onChange={e => setLocalFilters({ ...localFilters, user_id: e.target.value })}
            placeholder="Search User..."
          />
        </div>
        <div className="col-md-3">
          <label className="form-label small fw-bold">Transaction ID</label>
          <input
            type="text" className="form-control form-control-sm"
            value={localFilters.transaction_id || ''}
            onChange={e => setLocalFilters({ ...localFilters, transaction_id: e.target.value })}
            placeholder="Search Txn ID..."
          />
        </div>
        <div className="col-md-2">
          <label className="form-label small fw-bold">Min Amount</label>
          <input
            type="number" className="form-control form-control-sm"
            value={localFilters.min_amount || ''}
            onChange={e => setLocalFilters({ ...localFilters, min_amount: e.target.value ? Number(e.target.value) : undefined })}
          />
        </div>
        <div className="col-md-2">
          <label className="form-label small fw-bold">Status</label>
          <select
            className="form-select form-select-sm"
            value={localFilters.is_fraudulent === undefined ? '' : String(localFilters.is_fraudulent)}
            onChange={e => setLocalFilters({ ...localFilters, is_fraudulent: e.target.value === '' ? undefined : e.target.value === 'true' })}
          >
            <option value="">All Transactions</option>
            <option value="false">Legitimate Only</option>
            <option value="true">Fraudulent Only</option>
          </select>
        </div>
        <div className="col-md-2 d-flex align-items-end">
          <button type="submit" className="btn btn-sm btn-primary w-100 d-flex align-items-center justify-content-center gap-2">
            <Search size={14} /> Search
          </button>
        </div>
      </div>

      {expanded && (
        <div className="row g-3 mt-1 pt-3 border-top">
          <div className="col-md-3">
            <label className="form-label small fw-bold">Start Date</label>
            <input
              type="date" className="form-control form-control-sm"
              value={localFilters.start_date || ''}
              onChange={e => setLocalFilters({ ...localFilters, start_date: e.target.value })}
            />
          </div>
          <div className="col-md-3">
            <label className="form-label small fw-bold">End Date</label>
            <input
              type="date" className="form-control form-control-sm"
              value={localFilters.end_date || ''}
              onChange={e => setLocalFilters({ ...localFilters, end_date: e.target.value })}
            />
          </div>
          <div className="col-md-2">
            <label className="form-label small fw-bold">Min Score</label>
            <input
              type="number" className="form-control form-control-sm"
              value={localFilters.min_score || ''}
              onChange={e => setLocalFilters({ ...localFilters, min_score: e.target.value ? Number(e.target.value) : undefined })}
            />
          </div>
          <div className="col-md-2">
            <label className="form-label small fw-bold">Max Score</label>
            <input
              type="number" className="form-control form-control-sm"
              value={localFilters.max_score || ''}
              onChange={e => setLocalFilters({ ...localFilters, max_score: e.target.value ? Number(e.target.value) : undefined })}
            />
          </div>
        </div>
      )}

      <div className="text-center mt-2">
        <button
          type="button"
          className="btn btn-link btn-sm text-decoration-none text-muted"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? <><ChevronDown size={14} /> Simple Filters</> : <><ChevronRight size={14} /> Advanced Filters</>}
        </button>
      </div>
    </form>
  );
}

function TransactionTable({ data }: { data: TransactionDetail[] }) {
  return (
    <div className="table-responsive">
      <table className="table table-hover align-middle mb-0">
        <thead className="table-light">
          <tr>
            <th>Timestamp</th>
            <th>User ID</th>
            <th>Amount</th>
            <th>Status</th>
            <th>Score</th>
            <th>Fraud Type</th>
            <th className="text-end">Actions</th>
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr><td colSpan={7} className="text-center p-4 text-muted">No transactions found matching filters</td></tr>
          ) : (
            data.map((tx) => (
              <tr key={tx.record_id}>
                <td className="small text-muted">{tx.created_at ? new Date(tx.created_at).toLocaleString() : '-'}</td>
                <td className="font-monospace small">{tx.user_id}</td>
                <td className="fw-bold">${tx.amount.toFixed(2)}</td>
                <td>
                  {tx.is_fraudulent ? (
                    <span className="badge bg-danger-subtle text-danger border border-danger-subtle">FRAUD</span>
                  ) : (
                    <span className="badge bg-success-subtle text-success border border-success-subtle">LEGIT</span>
                  )}
                </td>
                <td>
                  <div className="d-flex align-items-center gap-2">
                    <div className="progress flex-grow-1" style={{ height: '6px', width: '60px' }}>
                      <div
                        className={`progress-bar ${tx.is_fraudulent ? 'bg-danger' : 'bg-primary'}`}
                        style={{ width: `${(tx.merchant_risk_score || 0)}%` }}
                      />
                    </div>
                    <span className="small">{tx.merchant_risk_score || '--'}</span>
                  </div>
                </td>
                <td className="small">{tx.fraud_type || '--'}</td>
                <td className="text-end">
                  <button className="btn btn-sm btn-outline-primary py-0 px-2" style={{ fontSize: '0.7rem' }}>Details</button>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
