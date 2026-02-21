import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { AreaChart, Area, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { jobsApi } from '../api';
import { useTenant } from '../hooks/useTenant';
import { useCursorPagination, type CursorPage } from '../hooks/useCursorPagination';
import { CursorPaginationControls } from '../components/CursorPaginationControls';
import { ErrorBanner } from '../components/ErrorBanner';
import type { Job } from '../types/api';
import { Activity, CheckCircle, XCircle } from 'lucide-react';

const JOB_STATUSES = ['', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'] as const;
const JOB_TYPES = ['', 'TRAINING', 'BACKTEST', 'DATA_GENERATION', 'DEPLOYMENT'] as const;
const PAGE_SIZE = 25;

export function Jobs() {
  const { tenantId } = useTenant();
  const [statusFilter, setStatusFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');

  const filters = {
    ...(statusFilter ? { status: statusFilter } : {}),
    ...(typeFilter ? { job_type: typeFilter } : {}),
  };

  const pagination = useCursorPagination<Job>({
    queryKeyBase: ['jobs', tenantId],
    fetchPage: async ({ cursor, limit }): Promise<CursorPage<Job>> => {
      const resp = await jobsApi.list({
        ...filters,
        limit,
        cursor,
      });
      return {
        items: resp.jobs ?? [],
        nextCursor: resp.pagination?.next_cursor,
        total: resp.pagination?.total ? parseInt(String(resp.pagination.total)) : 0,
      };
    },
    limit: PAGE_SIZE,
    filters,
  });

  return (
    <div className="container-fluid py-4">
      <header className="mb-4 d-flex justify-content-between align-items-center">
        <div>
          <h1 className="display-6 fw-bold text-primary">Jobs</h1>
          <p className="text-muted">View and manage background jobs.</p>
        </div>
      </header>

      <JobsSummaryStrip />

      <div className="card shadow-sm border-0">
        <div className="card-header bg-white py-3">
          <div className="row g-3 align-items-end">
            <div className="col-auto">
              <label className="form-label small fw-bold mb-1">Status</label>
              <select
                className="form-select form-select-sm"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="">All Statuses</option>
                {JOB_STATUSES.filter(Boolean).map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
            <div className="col-auto">
              <label className="form-label small fw-bold mb-1">Type</label>
              <select
                className="form-select form-select-sm"
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
              >
                <option value="">All Types</option>
                {JOB_TYPES.filter(Boolean).map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="card-body p-0">
          {pagination.isLoading ? (
            <div className="text-center p-5">
              <div className="spinner-border text-primary" />
            </div>
          ) : pagination.isError ? (
            <div className="p-4">
              <ErrorBanner error={pagination.error} title="Failed to load jobs" />
            </div>
          ) : pagination.data.length === 0 ? (
            <div className="text-center p-5 text-muted">No jobs found</div>
          ) : (
            <div className="table-responsive">
              <table className="table table-hover align-middle mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Job ID</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {pagination.data.map((job) => (
                    <tr key={job.job_id}>
                      <td>
                        <Link
                          to={`/jobs/${job.job_id}`}
                          className="font-monospace text-decoration-none small"
                        >
                          {job.job_id.slice(0, 12)}...
                        </Link>
                      </td>
                      <td>
                        <span className="badge bg-light text-dark border">
                          {job.job_type}
                        </span>
                      </td>
                      <td>
                        <JobStatusBadge status={job.status} />
                      </td>
                      <td className="small text-muted">
                        {job.created_at ? new Date(job.created_at).toLocaleString() : '-'}
                      </td>
                      <td className="small text-muted">
                        {formatDuration(job.created_at, job.ended_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {pagination.data.length > 0 && (
          <div className="card-footer bg-white border-top">
            <CursorPaginationControls
              itemCount={pagination.data.length}
              total={pagination.total}
              hasNextPage={pagination.hasNextPage}
              onLoadMore={pagination.loadNext}
              isFetching={pagination.isFetching}
              pageSize={PAGE_SIZE}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function JobsSummaryStrip() {
  const { tenantId } = useTenant();
  const summaryQuery = useQuery({
    queryKey: ['jobs', tenantId, 'summary'],
    queryFn: () => jobsApi.getSummary()
  });

  if (summaryQuery.isLoading) return <div className="p-3 border rounded mb-4 bg-light text-center small">Loading summary...</div>;
  if (!summaryQuery.data || summaryQuery.data.summaries.length === 0) return null;

  const summaries = summaryQuery.data?.summaries ?? [];
  const total = summaries.reduce((acc: number, s) => acc + (Number(s.total_jobs) || 0), 0);
  const failed = summaries.reduce((acc: number, s) => acc + (Number(s.failed_jobs) || 0), 0);
  const successRate = total > 0 ? ((total - failed) / total) * 100 : 100;

  return (
    <div className="row g-3 mb-4">
      <div className="col-lg-3">
        <div className="card h-100 border-0 shadow-sm bg-primary text-white">
          <div className="card-body d-flex flex-column justify-content-center">
            <div className="d-flex align-items-center mb-2 opacity-75">
              <Activity size={16} className="me-2" />
              <span className="small text-uppercase fw-bold">Total Throughput</span>
            </div>
            <div className="h2 mb-0 fw-bold">{total.toLocaleString()}</div>
            <div className="small opacity-75 mt-1">Jobs processed today</div>
          </div>
        </div>
      </div>

      <div className="col-lg-2">
        <div className="card h-100 border-0 shadow-sm">
          <div className="card-body d-flex flex-column justify-content-center">
            <div className="d-flex align-items-center mb-2 text-success">
              <CheckCircle size={16} className="me-2" />
              <span className="small text-uppercase fw-bold">Success Rate</span>
            </div>
            <div className="h3 mb-0 fw-bold">{successRate.toFixed(1)}%</div>
          </div>
        </div>
      </div>

      <div className="col-lg-2">
        <div className="card h-100 border-0 shadow-sm">
          <div className="card-body d-flex flex-column justify-content-center">
            <div className="d-flex align-items-center mb-2 text-danger">
              <XCircle size={16} className="me-2" />
              <span className="small text-uppercase fw-bold">Failures</span>
            </div>
            <div className="h3 mb-0 fw-bold">{failed.toLocaleString()}</div>
          </div>
        </div>
      </div>

      <div className="col-lg-5">
        <div className="card h-100 border-0 shadow-sm">
          <div className="card-body p-2 d-flex align-items-center">
            <div style={{ width: '100%', height: 80 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={summaryQuery.data.summaries}>
                  <defs>
                    <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0d6efd" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#0d6efd" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                  <Tooltip
                    labelFormatter={(label) => new Date(label).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    contentStyle={{ fontSize: '12px', borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="total_jobs"
                    stroke="#0d6efd"
                    fillOpacity={1}
                    fill="url(#colorTotal)"
                    strokeWidth={2}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function JobStatusBadge({ status }: { status: string }) {
  const classes: Record<string, string> = {
    QUEUED: 'bg-info-subtle text-info border-info-subtle',
    RUNNING: 'bg-primary-subtle text-primary border-primary-subtle',
    COMPLETED: 'bg-success-subtle text-success border-success-subtle',
    FAILED: 'bg-danger-subtle text-danger border-danger-subtle',
    CANCELLED: 'bg-warning-subtle text-warning border-warning-subtle',
  };

  return (
    <span className={`badge border ${classes[status] ?? 'bg-secondary-subtle text-secondary'}`}>
      {status}
    </span>
  );
}

function formatDuration(start?: string | Date, end?: string | Date): string {
  if (!start) return '--';
  const startTime = new Date(start).getTime();
  const endTime = end ? new Date(end).getTime() : Date.now(); // Calculate duration so far if running? Or just '--'?
  // If end is undefined, it might be running. But the original code was: if (!end) return '--';
  if (!end) return '--';

  const ms = endTime - startTime;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}
