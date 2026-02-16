import { useState } from 'react';
import { EmptyState } from '../../components/EmptyState';
import { Scale } from 'lucide-react';
import { Link } from 'react-router-dom';
import { decisionsApi } from '../../api';
import { useTenant } from '../../context/TenantContext'; // Corrected import path
import { useCursorPagination, type CursorPage } from '../../hooks/useCursorPagination'; // Corrected import path
import { CursorPaginationControls } from '../../components/CursorPaginationControls'; // Corrected import path
import { ErrorBanner } from '../../components/ErrorBanner'; // Corrected import path
import type { Decision } from '../../types/api';

const PAGE_SIZE = 25;

export function DecisionsPage() {
    const { tenantId } = useTenant();
    const [userIdFilter, setUserIdFilter] = useState('');
    const [decisionFilter, setDecisionFilter] = useState('');

    const filters = {
        ...(userIdFilter ? { user_id: userIdFilter } : {}),
        ...(decisionFilter ? { decision: decisionFilter } : {}),
    };

    const pagination = useCursorPagination<Decision>({
        queryKeyBase: ['decisions', tenantId],
        fetchPage: async ({ cursor, limit }): Promise<CursorPage<Decision>> => {
            const resp = await decisionsApi.list({
                ...filters,
                limit,
                cursor,
            });
            return {
                items: resp.decisions ?? [],
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
                <h1 className="display-6 fw-bold text-primary">Decisions</h1>
                <p className="text-muted">Explore and audit decision history.</p>
            </header>

            <div className="card shadow-sm border-0">
                <div className="card-header bg-white py-3">
                    <div className="row g-3 align-items-end">
                        <div className="col-auto">
                            <label className="form-label small fw-bold mb-1">User ID</label>
                            <input
                                type="text"
                                className="form-control form-control-sm"
                                placeholder="Filter by User ID"
                                value={userIdFilter}
                                onChange={(e) => setUserIdFilter(e.target.value)}
                            />
                        </div>
                        <div className="col-auto">
                            <label className="form-label small fw-bold mb-1">Decision</label>
                            <select
                                className="form-select form-select-sm"
                                value={decisionFilter}
                                onChange={(e) => setDecisionFilter(e.target.value)}
                            >
                                <option value="">All Decisions</option>
                                <option value="APPROVE">APPROVE</option>
                                <option value="DECLINE">DECLINE</option>
                                <option value="MANUAL_REVIEW">MANUAL_REVIEW</option>
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
                            <ErrorBanner error={pagination.error} title="Failed to load decisions" />
                        </div>
                    ) : pagination.data.length === 0 ? (
                        <EmptyState
                            title="No decisions found"
                            description="Adjust filters or check back later."
                            icon={<Scale size={48} />}
                        />
                    ) : (
                        <div className="table-responsive">
                            <table className="table table-hover align-middle mb-0">
                                <thead className="table-light">
                                    <tr>
                                        <th>Decision ID</th>
                                        <th>User ID</th>
                                        <th>Result</th>
                                        <th>Score</th>
                                        <th>Timestamp</th>
                                        <th>Model</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {pagination.data.map((d) => (
                                        <tr key={d.decision_id}>
                                            <td>
                                                <Link
                                                    to={`/decisions/${d.decision_id}`}
                                                    className="font-monospace text-decoration-none small"
                                                >
                                                    {d.decision_id.slice(0, 12)}...
                                                </Link>
                                            </td>
                                            <td className="font-monospace small">{d.user_id}</td>
                                            <td>
                                                <DecisionBadge decision={d.decision} />
                                            </td>
                                            <td className="font-monospace">{(d.score * 100).toFixed(1)}</td>
                                            <td className="small text-muted">
                                                {new Date(d.timestamp).toLocaleString()}
                                            </td>
                                            <td className="small text-muted">{d.model_version}</td>
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

function DecisionBadge({ decision }: { decision: string }) {
    const classes: Record<string, string> = {
        APPROVE: 'bg-success-subtle text-success border-success-subtle',
        DECLINE: 'bg-danger-subtle text-danger border-danger-subtle',
        MANUAL_REVIEW: 'bg-warning-subtle text-warning border-warning-subtle',
    };

    return (
        <span className={`badge border ${classes[decision] ?? 'bg-secondary-subtle text-secondary'}`}>
            {decision}
        </span>
    );
}
