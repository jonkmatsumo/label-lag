import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { analyticsApi } from '../api';
import { 
  Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart 
} from 'recharts';
import { exportToCsv } from '../utils';
import { Download } from 'lucide-react';

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

  // Fetch recent alerts
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
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Dataset Overview</h3>
        </div>
        {overviewQuery.isLoading ? (
          <div className="loading">Loading overview metrics...</div>
        ) : overviewQuery.isError ? (
          <div className="alert alert-error">
            Failed to load overview: {overviewQuery.error?.message}
          </div>
        ) : overviewQuery.data ? (
          <div className="metrics-grid">
            <div className="metric-card shadow-sm border-0">
              <div className="metric-label">Total Records</div>
              <div className="metric-value">{formatNumber(overviewQuery.data.total_records)}</div>
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
                {alertsQuery.data?.alerts ? 
                  `${((alertsQuery.data.alerts.filter(a => a.score >= 80).length / Math.max(overviewQuery.data.total_records * 0.05, 1)) * 100).toFixed(1)}%` 
                  : '--'}
              </div>
              <div className="small text-muted mt-1" style={{fontSize: '0.7em'}}>False Positive Rate (Est)</div>
            </div>
          </div>
        ) : null}
        {overviewQuery.data && (
          <div className="date-range-info mt-3 small text-muted">
            Data range: {new Date(overviewQuery.data.min_timestamp).toLocaleDateString()} - {new Date(overviewQuery.data.max_timestamp).toLocaleDateString()}
          </div>
        )}
      </div>

      <div className="row mt-4">
        {/* Transaction Volume & Trends Chart */}
        <div className="col-lg-8">
          <div className="card h-100 shadow-sm border-0">
            <div className="card-header bg-white">
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
            <div className="card-header bg-white">
               <h3 className="card-title h6 fw-bold mb-0">Amount Distribution</h3>
            </div>
            <div className="card-body d-flex flex-column justify-content-center align-items-center text-muted text-center p-4">
              <div className="display-4 opacity-25 mb-3">📊</div>
              <p className="small mb-0">Visualization of transaction amounts by fraud status (Restoring MVP parity)</p>
            </div>
          </div>
        </div>
      </div>

      {/* Daily Stats Table */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-header">
          <h3 className="card-title">Daily Statistics</h3>
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
              className="form-input"
              value={daysFilter}
              onChange={(e) => setDaysFilter(parseInt(e.target.value))}
              style={{ width: 'auto' }}
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
          <div className="loading">Loading daily stats...</div>
        ) : dailyStatsQuery.isError ? (
          <div className="alert alert-error">
            Failed to load daily stats: {dailyStatsQuery.error?.message}
          </div>
        ) : dailyStatsQuery.data?.stats && dailyStatsQuery.data.stats.length > 0 ? (
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th style={{ textAlign: 'right' }}>Transactions</th>
                  <th style={{ textAlign: 'right' }}>Fraud</th>
                  <th style={{ textAlign: 'right' }}>Fraud Rate</th>
                  <th style={{ textAlign: 'right' }}>Total Amount</th>
                </tr>
              </thead>
              <tbody>
                {dailyStatsQuery.data.stats.map((stat) => (
                  <tr key={stat.date}>
                    <td>{new Date(stat.date).toLocaleDateString()}</td>
                    <td style={{ textAlign: 'right' }}>{stat.total_transactions.toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>{stat.fraud_transactions.toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      <span className={stat.fraud_rate > 0.05 ? 'text-danger' : ''}>
                        {formatPercent(stat.fraud_rate)}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>${stat.total_amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">📊</div>
            <div className="empty-state-title">No data available</div>
            <p>No transaction data found for the selected period</p>
          </div>
        )}
      </div>

      {/* Recent Alerts */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-header">
          <h3 className="card-title">Recent High-Risk Alerts</h3>
        </div>
        {alertsQuery.isLoading ? (
          <div className="loading">Loading alerts...</div>
        ) : alertsQuery.isError ? (
          <div className="alert alert-error">
            Failed to load alerts: {alertsQuery.error?.message}
          </div>
        ) : alertsQuery.data?.alerts && alertsQuery.data.alerts.length > 0 ? (
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>User</th>
                  <th style={{ textAlign: 'right' }}>Amount</th>
                  <th style={{ textAlign: 'center' }}>Score</th>
                  <th>Matched Rules</th>
                </tr>
              </thead>
              <tbody>
                {alertsQuery.data.alerts.map((alert) => (
                  <tr key={alert.id}>
                    <td>{new Date(alert.created_at).toLocaleString()}</td>
                    <td><code>{alert.user_id}</code></td>
                    <td style={{ textAlign: 'right' }}>${alert.amount.toFixed(2)}</td>
                    <td style={{ textAlign: 'center' }}>
                      <span className={`score-badge ${alert.score >= 80 ? 'score-high' : alert.score >= 50 ? 'score-medium' : 'score-low'}`}>
                        {alert.score}
                      </span>
                    </td>
                    <td>
                      {alert.matched_rules.length > 0 ? (
                        <div className="rule-tags">
                          {alert.matched_rules.slice(0, 3).map((rule, i) => (
                            <span key={i} className="rule-tag">{rule}</span>
                          ))}
                          {alert.matched_rules.length > 3 && (
                            <span className="rule-tag more">+{alert.matched_rules.length - 3}</span>
                          )}
                        </div>
                      ) : (
                        <span className="text-muted">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">🔔</div>
            <div className="empty-state-title">No recent alerts</div>
            <p>No high-risk transactions detected recently</p>
          </div>
        )}
      </div>
    </div>
  );
}