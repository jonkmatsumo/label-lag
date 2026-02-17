import { useState } from 'react';
import { EmptyState } from '../../components/EmptyState';
import { FileBarChart } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { profilesApi } from '../../api';
import { useCursorPagination, type CursorPage } from '../../hooks/useCursorPagination';
import { CursorPaginationControls } from '../../components/CursorPaginationControls';
import { ErrorBanner } from '../../components/ErrorBanner';
import type { DatasetProfile } from '../../types/api';

const PAGE_SIZE = 25;

export function ProfilesPage() {
    const navigate = useNavigate();
    const [selectedProfiles, setSelectedProfiles] = useState<string[]>([]);

    const pagination = useCursorPagination<DatasetProfile>({
        queryKeyBase: ['dataset-profiles'],
        fetchPage: async ({ cursor, limit }): Promise<CursorPage<DatasetProfile>> => {
            const resp = await profilesApi.list({
                limit,
                cursor,
            });
            return {
                items: resp.profiles ?? [],
                nextCursor: resp.pagination?.next_cursor,
                total: resp.pagination?.total ? parseInt(resp.pagination.total) : 0,
            };
        },
        limit: PAGE_SIZE,
    });

    const toggleSelection = (id: string) => {
        setSelectedProfiles(prev => {
            if (prev.includes(id)) return prev.filter(p => p !== id);
            if (prev.length >= 2) return [prev[1], id]; // Keep max 2
            return [...prev, id];
        });
    };

    const handleCompare = () => {
        if (selectedProfiles.length === 2) {
            navigate(`/dataset/profiles/compare?base_id=${selectedProfiles[0]}&target_id=${selectedProfiles[1]}`);
        }
    };

    return (
        <div className="container-fluid py-4">
            <header className="mb-4 d-flex justify-content-between align-items-center">
                <div>
                    <h1 className="display-6 fw-bold text-primary">Dataset Profiles</h1>
                    <p className="text-muted">History of dataset snapshots and statistics.</p>
                </div>
                <div className="d-flex gap-2">
                    {pagination.data.length > 0 && (
                        <Link to={`/dataset/profiles/${pagination.data[0].profile_id}`} className="btn btn-outline-primary">
                            View Latest
                        </Link>
                    )}
                    <button
                        className="btn btn-primary"
                        onClick={handleCompare}
                        disabled={selectedProfiles.length !== 2}
                    >
                        Compare ({selectedProfiles.length}/2)
                    </button>
                </div>
            </header>

            <div className="card shadow-sm border-0">
                <div className="card-body p-0">
                    {pagination.isLoading ? (
                        <div className="text-center p-5">
                            <div className="spinner-border text-primary" />
                        </div>
                    ) : pagination.isError ? (
                        <div className="p-4">
                            <ErrorBanner error={pagination.error} title="Failed to load profiles" />
                        </div>
                    ) : pagination.data.length === 0 ? (
                        <EmptyState
                            title="No profiles found"
                            description="Generate a dataset profile to see it here."
                            icon={<FileBarChart size={48} />}
                        />
                    ) : (
                        <div className="table-responsive">
                            <table className="table table-hover align-middle mb-0">
                                <thead className="table-light">
                                    <tr>
                                        <th style={{ width: '40px' }}></th>
                                        <th>Profile ID</th>
                                        <th>Created At</th>
                                        <th>Rows</th>
                                        <th>Columns</th>
                                        <th>Size</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {pagination.data.map((p) => (
                                        <tr key={p.profile_id} className={selectedProfiles.includes(p.profile_id) ? 'table-active' : ''}>
                                            <td>
                                                <input
                                                    type="checkbox"
                                                    className="form-check-input"
                                                    checked={selectedProfiles.includes(p.profile_id)}
                                                    onChange={() => toggleSelection(p.profile_id)}
                                                />
                                            </td>
                                            <td>
                                                <Link
                                                    to={`/dataset/profiles/${p.profile_id}`}
                                                    className="font-monospace text-decoration-none small"
                                                >
                                                    {p.profile_id.slice(0, 12)}...
                                                </Link>
                                            </td>
                                            <td className="small text-muted">
                                                {p.computed_at ? new Date(p.computed_at).toLocaleString() : '--'}
                                            </td>
                                            <td>{parseInt(p.record_count || '0').toLocaleString()}</td>
                                            <td>{p.feature_profiles?.length || 0}</td>
                                            <td className="small text-muted">
                                                {p.size_bytes ? (p.size_bytes / 1024 / 1024).toFixed(2) + ' MB' : '--'}
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
