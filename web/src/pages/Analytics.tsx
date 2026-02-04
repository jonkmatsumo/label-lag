import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { analyticsApi } from '../api';
import type { RecentAlert, TransactionSearchRequest, TransactionDetail } from '../types/api';
import {
  Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart
} from 'recharts';
import { exportToCsv } from '../utils';
import { Download, Search, ChevronDown, ChevronRight, AlertTriangle, CheckCircle } from 'lucide-react';
import { ErrorBanner } from '../components/ErrorBanner';

export function Analytics() {
  const [daysFilter, setDaysFilter] = useState(30);

  // Fetch overview metrics
  const overviewQuery = useQuery({
    queryKey: ['analytics', 'overview'],
    queryFn: analyticsApi.getOverview,
  });

  // Fetch daily stats
  const dailyStatsQuery = useQuery({
    queryKey: ['analytics', 'daily-stats', daysFilter],
    queryFn: () => analyticsApi.getDailyStats(daysFilter),
  });

  // Sort stats by date ascending for chart
  const chartData = dailyStatsQuery.data?.stats ?
    [...dailyStatsQuery.data.stats].sort((a,b) => new Date(a.date).getTime() - new Date(b.date).getTime()) : [];

  // Fetch recent alerts for metric only
  const alertsQuery = useQuery({
    queryKey: ['analytics', 'alerts'],
    queryFn: () => analyticsApi.getRecentAlerts(20),
  });

  const formatNumber = (n: number) => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toLocaleString();
  };

  const formatPercent = (n: number) => `${(n * 100).toFixed(2)}%`;

  return (
    <div className="page">
      <h2>Historical Analytics</h2>
      <p>Dataset overview and fraud trends</p>

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
              <div className="metric-value">{formatNumber(overviewQuery.data.total_transactions)}</div>
            </div>
            <div className="metric-card shadow-sm border-0">
              <div className="metric-label">Fraud Count</div>
              <div className="metric-value text-danger">{formatNumber(overviewQuery.data.fraud_count)}</div>
            </div>
            <div className="metric-card shadow-sm border-0">
              <div className="metric-label">Fraud Rate</div>
              <div className="metric-value">{formatPercent(overviewQuery.data.fraud_rate)}</div>
            </div>
            <div className="metric-card shadow-sm border-0">
              <div className="metric-label">Est. FPR</div>
              <div className="metric-value text-warning">
                {alertsQuery.data?.alerts && overviewQuery.data ?
                  `${((alertsQuery.data.alerts.filter((a: RecentAlert) => a.score >= 80).length / Math.max(overviewQuery.data.total_transactions * 0.05, 1)) * 100).toFixed(1)}%`
                  : '--'}
              </div>
              <div className="small text-muted mt-1" style={{fontSize: '0.7em'}}>False Positive Rate (Est)</div>
            </div>
          </div>
        ) : null}
        {overviewQuery.data && (
          <div className="date-range-info px-4 pb-3 small text-muted">
            Data range: {new Date(overviewQuery.data.min_timestamp).toLocaleDateString()} - {new Date(overviewQuery.data.max_timestamp).toLocaleDateString()}
          </div>
        )}
      </div>

      <div className="row mt-4">
        {/* Transaction Volume & Trends Chart */}
        <div className="col-lg-8">
          <div className="card h-100 shadow-sm border-0">
            <div className="card-header bg-white border-bottom py-3">
               <h3 className="card-title h6 fw-bold mb-0">Transaction Volume & Fraud Trends</h3>
            </div>
            <div className="card-body" style={{ height: 350 }}>
              {dailyStatsQuery.isLoading ? <div className="loading">Loading charts...</div> : (
                 chartData.length > 0 ? (
                   <ResponsiveContainer width="100%" height="100%">
                     <ComposedChart data={chartData}>
                       <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                       <XAxis
                         dataKey="date"
                         tickFormatter={d => new Date(d).toLocaleDateString(undefined, {month:'short', day:'numeric'})}
                         fontSize={10}
                       />
                       <YAxis yAxisId="left" orientation="left" stroke="#8884d8" fontSize={10} />
                       <YAxis yAxisId="right" orientation="right" stroke="#ff7300" fontSize={10} />
                       <Tooltip labelFormatter={l => new Date(l).toLocaleDateString()}/>
                       <Legend wrapperStyle={{fontSize: '10px'}} />
                       <Bar yAxisId="left" dataKey="total_transactions" name="Volume" fill="#8884d8" barSize={20} radius={[4, 4, 0, 0]} />
                       <Line yAxisId="right" type="monotone" dataKey="fraud_transactions" name="Fraud Count" stroke="#ff7300" strokeWidth={2} dot={{r: 3}} />
                     </ComposedChart>
                   </ResponsiveContainer>
                 ) : <div className="text-center text-muted p-5">No data for charts</div>
              )}
            </div>
          </div>
        </div>

        {/* Amount Distribution Placeholder */}
        <div className="col-lg-4">
          <div className="card h-100 shadow-sm border-0">
            <div className="card-header bg-white border-bottom py-3">
               <h3 className="card-title h6 fw-bold mb-0">Amount Distribution</h3>
            </div>
            <div className="card-body" style={{ height: 350 }}>
              <AmountDistributionChart />
            </div>
          </div>
        </div>
      </div>

      {/* Daily Stats Table */}
      <div className="card shadow-sm border-0 mt-4">
        <div className="card-header bg-white border-bottom py-3 d-flex justify-content-between align-items-center">
          <h3 className="card-title h6 fw-bold mb-0">Daily Statistics</h3>
          <div className="card-actions">
            <button
              className="btn btn-secondary btn-sm me-2"
              onClick={() => dailyStatsQuery.data?.stats && exportToCsv(dailyStatsQuery.data.stats, 'daily_stats.csv')}
              disabled={!dailyStatsQuery.data?.stats?.length}
              style={{ marginRight: '0.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}
            >
              <Download size={14} /> Export
            </button>
            <select
              className="form-select form-select-sm d-inline-block w-auto"
              value={daysFilter}
              onChange={(e) => setDaysFilter(parseInt(e.target.value))}
            >
              <option value={7}>Last 7 days</option>
              <option value={14}>Last 14 days</option>
              <option value={30}>Last 30 days</option>
              <option value={60}>Last 60 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </div>
        </div>
        {dailyStatsQuery.isLoading ? (
          <div className="loading p-4">Loading daily stats...</div>
        ) : dailyStatsQuery.isError ? (
          <ErrorBanner error={dailyStatsQuery.error} title="Failed to load daily stats" className="m-4 alert alert-danger" />
        ) : dailyStatsQuery.data?.stats && dailyStatsQuery.data.stats.length > 0 ? (
          <div className="table-responsive">
            <table className="table table-hover align-middle mb-0">
              <thead className="table-light">
                <tr>
                  <th className="ps-4">Date</th>
                  <th className="text-end">Transactions</th>
                  <th className="text-end">Fraud</th>
                  <th className="text-end">Fraud Rate</th>
                  <th className="text-end pe-4">Total Amount</th>
                </tr>
              </thead>
              <tbody>
                {dailyStatsQuery.data.stats.map((stat) => (
                  <tr key={stat.date}>
                    <td className="ps-4 small">{new Date(stat.date).toLocaleDateString()}</td>
                    <td className="text-end small">{stat.total_transactions.toLocaleString()}</td>
                    <td className="text-end small">{stat.fraud_transactions.toLocaleString()}</td>
                    <td className="text-end small">
                      <span className={`badge rounded-pill ${stat.fraud_rate > 0.05 ? 'bg-danger' : 'bg-light text-dark border'}`}>
                        {formatPercent(stat.fraud_rate)}
                      </span>
                    </td>
                    <td className="text-end pe-4 small font-monospace">${stat.total_amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state text-center py-5 opacity-50">
            <div className="display-4">📊</div>
            <p className="mt-2">No transaction data found for the selected period</p>
          </div>
        )}
      </div>

      <TransactionExplorer />
    </div>
  );
}

function AmountDistributionChart() {
  const distQuery = useQuery({
    queryKey: ['analytics', 'dist'],
    queryFn: () => analyticsApi.searchTransactions({ limit: 1000 }),
    staleTime: 300000
  });

  const chartData = useMemo(() => {
    if (!distQuery.data?.transactions || distQuery.data.transactions.length === 0) return [];

    const txns = distQuery.data.transactions;
    const amounts = txns.map(t => t.amount);
    const min = Math.min(...amounts);
    const max = Math.max(...amounts);

    if (min === max) return [];

    const bins = 20;
    const binSize = (max - min) / bins;

    const binData = Array.from({ length: bins }, (_, i) => ({
      range: `${(min + i * binSize).toFixed(0)}-${(min + (i + 1) * binSize).toFixed(0)}`,
      min: min + i * binSize,
      fraud: 0,
      legit: 0
    }));

    txns.forEach(t => {
      const binIdx = Math.min(Math.floor((t.amount - min) / binSize), bins - 1);
      if (binIdx >= 0) {
        if (t.is_fraudulent || t.is_fraud) binData[binIdx].fraud++;
        else binData[binIdx].legit++;
      }
    });

    return binData;
  }, [distQuery.data]);

  if (distQuery.isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary"/></div>;
  if (chartData.length === 0) return <div className="text-center p-5 text-muted">No data available</div>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
        <XAxis dataKey="range" fontSize={10} tickFormatter={(val) => val.split('-')[0]} />
        <YAxis fontSize={10} />
        <Tooltip />
        <Legend wrapperStyle={{fontSize: '10px'}} />
        <Bar dataKey="legit" name="Legitimate" stackId="a" fill="#4caf50" />
        <Bar dataKey="fraud" name="Fraudulent" stackId="a" fill="#f44336" />
      </BarChart>
    </ResponsiveContainer>
  );
}

function TransactionExplorer() {
  const [filters, setFilters] = useState<TransactionSearchRequest>({
    limit: 20,
    offset: 0
  });

  const searchQuery = useQuery({
    queryKey: ['analytics', 'search', filters],
    queryFn: () => analyticsApi.searchTransactions(filters),
    keepPreviousData: true
  });

  const handlePageChange = (newOffset: number) => {
    setFilters(prev => ({ ...prev, offset: newOffset }));
  };

  return (
    <div className="card shadow-sm border-0 mt-4 mb-5">
      <div className="card-header bg-white border-bottom py-3">
        <h3 className="card-title h6 fw-bold mb-0">Transaction Explorer</h3>
      </div>
      <div className="card-body">
        <TransactionFilters filters={filters} onChange={setFilters} />

        {searchQuery.isLoading && !searchQuery.isPlaceholderData ? (
          <div className="text-center p-5"><div className="spinner-border text-primary"/></div>
        ) : searchQuery.isError ? (
          <ErrorBanner error={searchQuery.error} title="Error loading transactions" className="alert alert-danger" />
        ) : (
          <>
            <TransactionTable data={searchQuery.data?.transactions || []} />

            {/* Pagination Controls */}
            <div className="d-flex justify-content-between align-items-center mt-3">
              <div className="small text-muted">
                Showing {filters.offset! + 1} to {Math.min(filters.offset! + (searchQuery.data?.transactions.length || 0), searchQuery.data?.total || 0)} of {searchQuery.data?.total || 0}
              </div>
              <div className="btn-group btn-group-sm">
                <button
                  className="btn btn-outline-secondary"
                  disabled={filters.offset === 0}
                  onClick={() => handlePageChange(Math.max(0, filters.offset! - filters.limit!))}
                >
                  Previous
                </button>
                <button
                  className="btn btn-outline-secondary"
                  disabled={(filters.offset! + filters.limit!) >= (searchQuery.data?.total || 0)}
                  onClick={() => handlePageChange(filters.offset! + filters.limit!)}
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
            onChange={e => setLocalFilters({...localFilters, user_id: e.target.value || undefined})}
            placeholder="Search User..."
          />
        </div>
        <div className="col-md-3">
          <label className="form-label small fw-bold">Transaction ID</label>
          <input
            type="text" className="form-control form-control-sm"
            value={localFilters.transaction_id || ''}
            onChange={e => setLocalFilters({...localFilters, transaction_id: e.target.value || undefined})}
            placeholder="Search Txn..."
          />
        </div>
        <div className="col-md-2">
          <label className="form-label small fw-bold">Status</label>
          <select
            className="form-select form-select-sm"
            value={localFilters.is_fraudulent === undefined ? '' : String(localFilters.is_fraudulent)}
            onChange={e => {
              const val = e.target.value;
              setLocalFilters({...localFilters, is_fraudulent: val === '' ? undefined : val === 'true'});
            }}
          >
            <option value="">All</option>
            <option value="true">Fraud</option>
            <option value="false">Legitimate</option>
          </select>
        </div>
        <div className="col-md-2 d-flex align-items-end">
          <button type="submit" className="btn btn-primary btn-sm w-100 fw-bold">
            <Search size={14} className="me-1" /> Search
          </button>
        </div>
        <div className="col-md-2 d-flex align-items-end">
          <button type="button" className="btn btn-link btn-sm text-decoration-none" onClick={() => setExpanded(!expanded)}>
            {expanded ? <ChevronDown size={14}/> : <ChevronRight size={14}/>} Advanced
          </button>
        </div>
      </div>

      {expanded && (
        <div className="row g-3 mt-2 pt-2 border-top">
          <div className="col-md-3">
            <label className="form-label small text-muted">Min Amount</label>
            <input type="number" className="form-control form-control-sm" value={localFilters.min_amount || ''} onChange={e => setLocalFilters({...localFilters, min_amount: e.target.value ? parseFloat(e.target.value) : undefined})} />
          </div>
          <div className="col-md-3">
            <label className="form-label small text-muted">Max Amount</label>
            <input type="number" className="form-control form-control-sm" value={localFilters.max_amount || ''} onChange={e => setLocalFilters({...localFilters, max_amount: e.target.value ? parseFloat(e.target.value) : undefined})} />
          </div>
          <div className="col-md-3">
            <label className="form-label small text-muted">Start Date</label>
            <input type="date" className="form-control form-control-sm" value={localFilters.start_date || ''} onChange={e => setLocalFilters({...localFilters, start_date: e.target.value || undefined})} />
          </div>
          <div className="col-md-3">
            <label className="form-label small text-muted">End Date</label>
            <input type="date" className="form-control form-control-sm" value={localFilters.end_date || ''} onChange={e => setLocalFilters({...localFilters, end_date: e.target.value || undefined})} />
          </div>
        </div>
      )}
    </form>
  );
}

function TransactionTable({ data }: { data: TransactionDetail[] }) {
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  if (data.length === 0) return <div className="text-center p-5 text-muted">No transactions found matching criteria.</div>;

  return (
    <div className="table-responsive">
      <table className="table table-hover align-middle mb-0 small">
        <thead className="table-light">
          <tr>
            <th style={{width: '30px'}}></th>
            <th>Time</th>
            <th>Transaction ID</th>
            <th>User</th>
            <th className="text-end">Amount</th>
            <th className="text-center">Status</th>
            <th className="text-center">Score</th>
          </tr>
        </thead>
        <tbody>
          {data.map(txn => (
            <React.Fragment key={txn.record_id || txn.id}>
              <tr
                onClick={() => setExpandedRow(expandedRow === (txn.record_id || txn.id) ? null : (txn.record_id || txn.id))}
                className={expandedRow === (txn.record_id || txn.id) ? 'table-active' : ''}
                style={{cursor: 'pointer'}}
              >
                <td className="text-center text-muted">
                  {expandedRow === (txn.record_id || txn.id) ? <ChevronDown size={14}/> : <ChevronRight size={14}/>}
                </td>
                <td className="text-muted">{new Date(txn.created_at || txn.timestamp).toLocaleString()}</td>
                <td className="font-monospace">{txn.record_id || txn.id}</td>
                <td>{txn.user_id}</td>
                <td className="text-end font-monospace">${txn.amount.toFixed(2)}</td>
                <td className="text-center">
                  {txn.is_fraudulent || txn.is_fraud ? (
                    <span className="badge bg-danger rounded-pill"><AlertTriangle size={10} className="me-1"/>Fraud</span>
                  ) : (
                    <span className="badge bg-light text-success border rounded-pill"><CheckCircle size={10} className="me-1"/>OK</span>
                  )}
                </td>
                <td className="text-center">
                  {txn.merchant_risk_score !== undefined ? (
                    <span className={`badge ${txn.merchant_risk_score > 50 ? 'bg-warning text-dark' : 'bg-light text-dark border'}`}>
                      {txn.merchant_risk_score}
                    </span>
                  ) : '-'}
                </td>
              </tr>
              {expandedRow === (txn.record_id || txn.id) && (
                <tr>
                  <td colSpan={7} className="p-0">
                    <div className="p-3 bg-light border-bottom">
                      <div className="row g-3">
                        <div className="col-md-4">
                          <h6 className="fw-bold mb-2 text-muted text-uppercase x-small">Risk Factors</h6>
                          <ul className="list-unstyled mb-0">
                            <li><strong>Velocity (24h):</strong> {txn.velocity_24h ?? 'N/A'}</li>
                            <li><strong>Amt/Avg (30d):</strong> {txn.amount_to_avg_ratio_30d?.toFixed(2) ?? 'N/A'}</li>
                            <li><strong>Balance Volatility:</strong> {txn.balance_volatility_z_score?.toFixed(2) ?? 'N/A'}</li>
                          </ul>
                        </div>
                        <div className="col-md-4">
                          <h6 className="fw-bold mb-2 text-muted text-uppercase x-small">Context</h6>
                          <ul className="list-unstyled mb-0">
                            <li><strong>Fraud Type:</strong> {txn.fraud_type || 'None'}</li>
                            <li><strong>Off Hours:</strong> {txn.is_off_hours_txn ? 'Yes' : 'No'}</li>
                            <li><strong>Merchant Score:</strong> {txn.merchant_risk_score ?? 'N/A'}</li>
                          </ul>
                        </div>
                        <div className="col-md-4">
                          <h6 className="fw-bold mb-2 text-muted text-uppercase x-small">Raw Data</h6>
                          <pre className="mb-0 bg-white p-2 border rounded" style={{fontSize: '0.7em', maxHeight: '100px', overflowY: 'auto'}}>
                            {JSON.stringify(txn, null, 2)}
                          </pre>
                        </div>
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
