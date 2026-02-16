import { useState } from 'react';
import { Link } from 'react-router-dom';
import { modelVersionsApi } from '../../api';
import { useCursorPagination, type CursorPage } from '../../hooks/useCursorPagination';
import { CursorPaginationControls } from '../../components/CursorPaginationControls';
import { ErrorBanner } from '../../components/ErrorBanner';
import type { ModelVersion } from '../../types/api';

const PAGE_SIZE = 25;

export function ModelVersionsPage() {
    const [modelNameFilter, setModelNameFilter] = useState('');

    const filters = {
        ...(modelNameFilter ? { model_name: modelNameFilter } : {}),
    };

    const pagination = useCursorPagination<ModelVersion>({
        queryKeyBase: ['model-versions'], // Model versions are global or tenant-scoped? The API uses tenant header implicitly.
        fetchPage: async ({ cursor, limit }): Promise<CursorPage<ModelVersion>> => {
            const resp = await modelVersionsApi.list({
                ...filters,
                limit,
                cursor,
            });
            return {
                items: resp.versions ?? [],
                nextCursor: resp.pagination?.next_cursor,
                total: resp.pagination?.total,
            };
        },
        limit: PAGE_SIZE,
        filters,
    });

    return (
        <div className="container-fluid py-4">
            <header className="mb-4">
                <h1 className="display-6 fw-bold text-primary">Model Versions</h1>
                <p className="text-muted">Registry of trained model versions.</p>
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
                    </div>
                </div>

                <div className="card-body p-0">
                    {pagination.isLoading ? (
                        <div className="text-center p-5">
                            <div className="spinner-border text-primary" />
                        </div>
                    ) : pagination.isError ? (
                        <div className="p-4">
                            <ErrorBanner error={pagination.error} title="Failed to load model versions" />
                        </div>
                    ) : pagination.data.length === 0 ? (
                        <div className="text-center p-5 text-muted">No model versions found</div>
                    ) : (
                        <div className="table-responsive">
                            <table className="table table-hover align-middle mb-0">
                                <thead className="table-light">
                                    <tr>
                                        <th>Version</th>
                                        <th>Model Name</th>
                                        <th>Status</th>
                                        <th>Created At</th>
                                        <th>Deployed At</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {pagination.data.map((v) => (
                                        <tr key={v.version}>
                                            <td>
                                                <Link
                                                    to={`/models/${encodeURIComponent(v.version)}`}
                                                    className="font-monospace text-decoration-none small"
                                                >
                                                    {v.version}
                                                </Link>
                                            </td>
                                            <td>{v.model_name}</td>
                                            <td>
                                                <StatusBadge status={v.status} />
                                            </td>
                                            <td className="small text-muted">
                                                {new Date(v.created_at).toLocaleString()}
                                            </td>
                                            <td className="small text-muted">
                                                {v.deployed_at ? new Date(v.deployed_at).toLocaleString() : '--'}
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
        READY: 'bg-success-subtle text-success border-success-subtle',
        ARCHIVED: 'bg-secondary-subtle text-secondary border-secondary-subtle',
        FAILED: 'bg-danger-subtle text-danger border-danger-subtle',
        TRAINING: 'bg-primary-subtle text-primary border-primary-subtle',
    };

    return (
        <span className={`badge border ${classes[status] ?? 'bg-light text-dark border'}`}>
            {status}
        </span>
    );
}
