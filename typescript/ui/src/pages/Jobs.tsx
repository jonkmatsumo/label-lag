import { useState } from 'react';
import { Link } from 'react-router-dom';
import { jobsApi } from '../api';
import { useTenant } from '../hooks/useTenant';
import { useCursorPagination, type CursorPage } from '../hooks/useCursorPagination';
import { CursorPaginationControls } from '../components/CursorPaginationControls';
import { ErrorBanner } from '../components/ErrorBanner';
import type { Job } from '../types/api';

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
      <header className="mb-4">
        <h1 className="display-6 fw-bold text-primary">Jobs</h1>
        <p className="text-muted">View and manage background jobs.</p>
      </header>

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
