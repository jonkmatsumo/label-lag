import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { profilesApi } from '../../api';

export function ProfileDetail() {
    const { id } = useParams<{ id: string }>();

    const { data: profile, isLoading, error } = useQuery({
        queryKey: ['dataset-profile', id],
        queryFn: () => profilesApi.get(id!),
        enabled: !!id,
    });

    const { data: summary } = useQuery({
        queryKey: ['dataset-summary', id],
        queryFn: () => profilesApi.getSummary(id!),
        enabled: !!id,
    });

    if (isLoading) {
        return (
            <div className="text-center p-5">
                <div className="spinner-border text-primary" />
            </div>
        );
    }

    if (error || !profile) {
        return (
            <div className="alert alert-danger m-4">
                Failed to load profile details.
            </div>
        );
    }

    return (
        <div className="container-fluid py-4">
            <header className="mb-4">
                <h1 className="display-6 fw-bold text-primary">Dataset Profile</h1>
                <p className="font-monospace text-muted">{id}</p>
            </header>

            <div className="row g-4">
                <div className="col-md-4">
                    <div className="card shadow-sm h-100">
                        <div className="card-header bg-white">
                            <h5 className="mb-0">Overview</h5>
                        </div>
                        <div className="card-body">
                            <dl className="row mb-0">
                                <dt className="col-sm-5">Created</dt>
                                <dd className="col-sm-7 small">
                                    {new Date(profile.computed_at || 0).toLocaleString()}
                                </dd>

                                <dt className="col-sm-5">Rows</dt>
                                <dd className="col-sm-7">{(profile.record_count || 0).toLocaleString()}</dd>

                                <dt className="col-sm-5">Columns</dt>
                                <dd className="col-sm-7">{profile.feature_profiles?.length || 0}</dd>

                                <dt className="col-sm-5">Size</dt>
                                <dd className="col-sm-7">
                                    {profile.size_bytes ? (profile.size_bytes / 1024 / 1024).toFixed(2) + ' MB' : '--'}
                                </dd>
                            </dl>
                        </div>
                    </div>
                </div>

                <div className="col-md-8">
                    <div className="card shadow-sm h-100">
                        <div className="card-header bg-white">
                            <h5 className="mb-0">Column Statistics</h5>
                        </div>
                        <div className="card-body p-0">
                            {summary ? (
                                <div className="table-responsive" style={{ maxHeight: '400px' }}>
                                    <table className="table table-hover table-sm mb-0">
                                        <thead className="table-light sticky-top">
                                            <tr>
                                                <th>Column</th>
                                                <th>Type</th>
                                                <th>Nulls</th>
                                                <th>Distinct</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {summary.profile?.feature_profiles.map((stats) => (
                                                <tr key={stats.name}>
                                                    <td className="font-monospace small">{stats.name}</td>
                                                    <td><code>{stats.type}</code></td>
                                                    <td>{(stats.null_rate * 100).toFixed(1)}%</td>
                                                    <td>{stats.mean.toFixed(2)}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            ) : (
                                <div className="p-4 text-center text-muted">
                                    No detailed statistics available
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
