import React, { useState } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { analyticsApi } from '../api';
import type { RecentAlert, TransactionSearchRequest, TransactionDetail } from '../types/api';
import {
  Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart
} from 'recharts';
import { Search, ChevronDown, ChevronRight, BarChart3 } from 'lucide-react';
import { ErrorBanner } from '../components/ErrorBanner';
import { DateRangePicker, KpiCard } from '../components';
import { useTenant } from '../hooks/useTenant';
import type { DateRange } from '../components/DateRangePicker';

export function Analytics() {
  const [daysFilter] = useState(30);
  const { tenantId } = useTenant();

  // New state for dynamic dashboard
  const [dateRange, setDateRange] = useState<DateRange>(() => {
    const end = new Date();
    const start = new Date();
    start.setDate(end.getDate() - 7);
    return {
      start: start.toISOString().split('T')[0],
      end: end.toISOString().split('T')[0],
    };
  });
  const [granularity, setGranularity] = useState<'hour' | 'day'>('day');

  // Fetch performance KPIs
  const kpisQuery = useQuery({
    queryKey: ['analytics', tenantId, 'kpis', dateRange, granularity],
    queryFn: () => analyticsApi.getKpis({
      start_time: dateRange.start,
      end_time: dateRange.end,
      group_by: granularity
    }),
    staleTime: 30000,
  });

  // Fetch volume timeseries
  const volumeQuery = useQuery({
    queryKey: ['analytics', tenantId, 'volume', dateRange, granularity],
    queryFn: () => analyticsApi.getVolume({
      start_time: dateRange.start,
      end_time: dateRange.end,
      granularity
    }),
    staleTime: 30000,
  });

  // Fetch overview metrics (legacy/static)
  const overviewQuery = useQuery({
    queryKey: ['analytics', tenantId, 'overview'],
    queryFn: () => analyticsApi.getOverview(daysFilter),
  });

  // Fetch recent alerts for FPR calculation
  const alertsQuery = useQuery({
    queryKey: ['analytics', tenantId, 'alerts'],
    queryFn: () => analyticsApi.getRecentAlerts(20),
  });

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

  return (
    <div className="page">
      <h2>Historical Analytics</h2>
      <p>Dataset overview and fraud trends</p>

      {/* KPI Dashboard Controls */}
      <div className="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-3">
        <DateRangePicker onChange={setDateRange} />
        <div className="btn-group btn-group-sm">
          <button
            className={`btn ${granularity === 'hour' ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => setGranularity('hour')}
          >
            Hourly
          </button>
          <button
            className={`btn ${granularity === 'day' ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => setGranularity('day')}
          >
            Daily
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="row g-3 mb-4">
        <div className="col-md">
          <KpiCard
            label="Total Decisions"
            value={kpisQuery.data?.total_decisions ?? 0}
            loading={kpisQuery.isLoading}
            error={kpisQuery.error}
            formatter={(val) => formatNumber(val as string | number | null | undefined)}
          />
        </div>
        <div className="col-md">
          <KpiCard
            label="Total Alerts"
            value={kpisQuery.data?.total_alerts ?? 0}
            loading={kpisQuery.isLoading}
            error={kpisQuery.error}
            formatter={(val) => formatNumber(val as string | number | null | undefined)}
          />
        </div>
        <div className="col-md">
          <KpiCard
            label="Alert Rate"
            value={kpisQuery.data?.alert_rate ?? 0}
            loading={kpisQuery.isLoading}
            error={kpisQuery.error}
            formatter={(val) => `${(Number(val) * 100).toFixed(1)}%`}
          />
        </div>
        <div className="col-md">
          <KpiCard
            label="Avg Risk Score"
            value={kpisQuery.data?.avg_score ?? 0}
            loading={kpisQuery.isLoading}
            error={kpisQuery.error}
            formatter={(val) => Number(val).toFixed(1)}
          />
        </div>
        <div className="col-md">
          <KpiCard
            label="Rules Fired"
            value={kpisQuery.data?.rules_fired_total ?? 0}
            loading={kpisQuery.isLoading}
            error={kpisQuery.error}
            formatter={(val) => formatNumber(val as string | number | null | undefined)}
          />
        </div>
      </div>

      {/* Volume Chart */}
      <div className="card shadow-sm border-0 mb-4">
        <div className="card-header bg-white border-bottom py-3 d-flex align-items-center gap-2">
          <BarChart3 size={18} className="text-primary" />
          <h3 className="card-title h6 fw-bold mb-0">Transaction & Alert Volume</h3>
        </div>
        <div className="card-body" style={{ height: 350 }}>
          {volumeQuery.isLoading ? (
            <div className="d-flex align-items-center justify-content-center h-100">
              <div className="spinner-border spinner-border-sm text-primary me-2" /> Loading volume data...
            </div>
          ) : volumeQuery.isError ? (
            <div className="text-danger p-4">Failed to load volume chart</div>
          ) : volumeQuery.data?.points && volumeQuery.data.points.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={volumeQuery.data.points}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <XAxis
                  dataKey="timestamp"
                  fontSize={10}
                  tickFormatter={(val) => {
                    const d = new Date(val);
                    return granularity === 'hour'
                      ? `${d.getHours()}:00`
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
              No volume data available for the selected range
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
  const [filters, setFilters] = useState<TransactionSearchRequest>({
    user_id: '',
    transaction_id: '',
    start_date: '',
    end_date: '',
    limit: 20,
    offset: 0,
    tenant_id: tenantId,
  });

  const searchQuery = useQuery({
    queryKey: ['analytics', tenantId, 'search', filters],
    queryFn: () => analyticsApi.searchTransactions({ ...filters, tenant_id: tenantId }),
    placeholderData: keepPreviousData,
  });

  const handlePageChange = (newOffset: number) => {
    setFilters((prev: TransactionSearchRequest) => ({ ...prev, offset: newOffset }));
  };
  const totalTransactions = Number(searchQuery.data?.total ?? 0);

  return (
    <div className="card shadow-sm border-0 mt-4 mb-5">
      <div className="card-header bg-white border-bottom py-3">
        <h3 className="card-title h6 fw-bold mb-0">Transaction Explorer</h3>
      </div>
      <div className="card-body">
        <TransactionFilters filters={filters} onChange={setFilters} />

        {searchQuery.isLoading && !searchQuery.isPlaceholderData ? (
          <div className="text-center p-5"><div className="spinner-border text-primary" /></div>
        ) : searchQuery.isError ? (
          <ErrorBanner error={searchQuery.error} title="Error loading transactions" className="alert alert-danger" />
        ) : (
          <>
            <TransactionTable data={searchQuery.data?.transactions || []} />

            {/* Pagination Controls */}
            <div className="d-flex justify-content-between align-items-center mt-3">
              <div className="small text-muted">
                Showing {totalTransactions === 0 ? 0 : filters.offset + 1} to {Math.min(filters.offset + (searchQuery.data?.transactions.length || 0), totalTransactions)} of {totalTransactions}
              </div>
              <div className="btn-group btn-group-sm">
                <button
                  className="btn btn-outline-secondary"
                  disabled={filters.offset === 0}
                  onClick={() => handlePageChange(Math.max(0, filters.offset - filters.limit))}
                >
                  Previous
                </button>
                <button
                  className="btn btn-outline-secondary"
                  disabled={(filters.offset + filters.limit) >= totalTransactions}
                  onClick={() => handlePageChange(filters.offset + filters.limit)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function TransactionFilters({ filters, onChange }: { filters: TransactionSearchRequest, onChange: (f: TransactionSearchRequest) => void }) {
  const [localFilters, setLocalFilters] = useState(filters);
  const [expanded, setExpanded] = useState(false);

  const applyFilters = (e: React.FormEvent) => {
    e.preventDefault();
    onChange({ ...localFilters, offset: 0 }); // Reset page on filter
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
