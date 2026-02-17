import { useState } from 'react';
import { EmptyState } from '../../components/EmptyState';
import { Dumbbell } from 'lucide-react';
import { Link } from 'react-router-dom';
import { trainingApi } from '../../api';
import { useTenant } from '../../hooks/useTenant';
import { useCursorPagination, type CursorPage } from '../../hooks/useCursorPagination';
import { CursorPaginationControls } from '../../components/CursorPaginationControls';
import { ErrorBanner } from '../../components/ErrorBanner';
import type { TrainingRun } from '../../types/api';

const PAGE_SIZE = 25;

const STATUSES = ['', 'RUNNING', 'COMPLETED', 'FAILED'];

export function TrainingRunsPage() {
    const { tenantId } = useTenant();
    const [modelNameFilter, setModelNameFilter] = useState('');
    const [statusFilter, setStatusFilter] = useState('');

    const filters = {
        ...(modelNameFilter ? { model_name: modelNameFilter } : {}),
        ...(statusFilter ? { status: statusFilter } : {}),
    };

    const pagination = useCursorPagination<TrainingRun>({
        queryKeyBase: ['training-runs', tenantId],
        fetchPage: async ({ cursor, limit }): Promise<CursorPage<TrainingRun>> => {
            const resp = await trainingApi.list({
                ...filters,
                limit,
                cursor,
            });
            return {
                items: resp.runs ?? [],
                nextCursor: resp.pagination?.next_cursor,
                total: resp.pagination?.total ? parseInt(resp.pagination.total, 10) : undefined,
            };
        },
        limit: PAGE_SIZE,
        filters,
    });

    return (
        <div className="container-fluid py-4">
            <header className="mb-4">
                <h1 className="display-6 fw-bold text-primary">Training Runs</h1>
                <p className="text-muted">Monitor model training history and performance.</p>
            </header>

            <div className="card shadow-sm border-0">
                <div className="card-header bg-white py-3">
                    <div className="row g-3 align-items-end">
                        <div className="col-auto">
                            <label className="form-label small fw-bold mb-1">Model Name</label>
                            <input
                                type="text"
                                className="form-control form-control-sm"
                                placeholder="Filter by Model Name"
                                value={modelNameFilter}
                                onChange={(e) => setModelNameFilter(e.target.value)}
                            />
                        </div>
                        <div className="col-auto">
                            <label className="form-label small fw-bold mb-1">Status</label>
                            <select
                                className="form-select form-select-sm"
                                value={statusFilter}
                                onChange={(e) => setStatusFilter(e.target.value)}
                            >
                                <option value="">All Statuses</option>
                                {STATUSES.filter(Boolean).map((s) => (
                                    <option key={s} value={s}>{s}</option>
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
                            <ErrorBanner error={pagination.error} title="Failed to load training runs" />
                        </div>
                    ) : pagination.data.length === 0 ? (
                        <EmptyState
                            title="No training runs found"
                            description="Start a new training run or adjust filters."
                            icon={<Dumbbell size={48} />}
                        />
                    ) : (
                        <div className="table-responsive">
                            <table className="table table-hover align-middle mb-0">
                                <thead className="table-light">
                                    <tr>
                                        <th>Run ID</th>
                                        <th>Model Name</th>
                                        <th>Version</th>
                                        <th>Status</th>
                                        <th>Created At</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {pagination.data.map((run) => (
                                        <tr key={run.run_id}>
                                            <td>
                                                <Link
                                                    to={`/training-runs/${run.run_id}`}
                                                    className="font-monospace text-decoration-none small"
                                                >
                                                    {run.run_id.slice(0, 12)}...
                                                </Link>
                                            </td>
                                            <td>{run.model_name}</td>
                                            <td>
                                                <span className="badge bg-light text-dark border">
                                                    {run.mlflow_run_id?.slice(0, 7) ?? 'v1'}
                                                </span>
                                            </td>
                                            <td>
                                                <StatusBadge status={run.status} />
                                            </td>
                                            <td className="small text-muted">
                                                {run.started_at ? new Date(run.started_at).toLocaleString() : '-'}
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

function StatusBadge({ status }: { status: string }) {
    const classes: Record<string, string> = {
        COMPLETED: 'bg-success-subtle text-success border-success-subtle',
        FAILED: 'bg-danger-subtle text-danger border-danger-subtle',
        RUNNING: 'bg-primary-subtle text-primary border-primary-subtle',
    };

    return (
        <span className={`badge border ${classes[status] ?? 'bg-secondary-subtle text-secondary'}`}>
            {status}
        </span>
    );
}
