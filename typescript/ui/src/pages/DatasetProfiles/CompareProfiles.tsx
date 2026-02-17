import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { profilesApi } from '../../api';
import { ErrorBanner } from '../../components/ErrorBanner';

export function CompareProfiles() {
    const [searchParams] = useSearchParams();
    const baseId = searchParams.get('base_id');
    const targetId = searchParams.get('target_id');

    const { data, isLoading, error } = useQuery({
        queryKey: ['compare-profiles', baseId, targetId],
        queryFn: () => profilesApi.compare(baseId!, targetId!),
        enabled: !!baseId && !!targetId,
    });

    if (!baseId || !targetId) {
        return (
            <div className="alert alert-warning m-4">
                Select two profiles to compare.
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="text-center p-5">
                <div className="spinner-border text-primary" />
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="p-4">
                <ErrorBanner error={error} title="Failed to compare profiles" />
            </div>
        );
    }

    return (
        <div className="container-fluid py-4">
            <header className="mb-4">
                <h1 className="display-6 fw-bold text-primary">Compare Profiles</h1>
                <div className="d-flex gap-4 text-muted small">
                    <div>Base: <span className="font-monospace text-dark">{baseId}</span></div>
                    <div>Target: <span className="font-monospace text-dark">{targetId}</span></div>
                </div>
            </header>

            <div className="card shadow-sm border-0">
                <div className="card-header bg-white py-3">
                    <h5 className="mb-0">Feature PSI Analysis</h5>
                </div>
                <div className="card-body p-0">
                    <div className="table-responsive">
                        <table className="table table-hover align-middle mb-0">
                            <thead className="table-light">
                                <tr>
                                    <th>Feature</th>
                                    <th>PSI Score</th>
                                    <th>Severity</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.drift.map((f: { feature_name: string; psi: number; drift_severity: string }) => (
                                    <tr key={f.feature_name}>
                                        <td className="font-monospace small">{f.feature_name}</td>
                                        <td>{f.psi.toFixed(4)}</td>
                                        <td>
                                            <SeverityBadge severity={f.drift_severity as 'low' | 'medium' | 'high'} />
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}

function SeverityBadge({ severity }: { severity: 'low' | 'medium' | 'high' }) {
    const classes = {
        low: 'bg-success-subtle text-success border-success-subtle',
        medium: 'bg-warning-subtle text-warning border-warning-subtle',
        high: 'bg-danger-subtle text-danger border-danger-subtle',
    };
    return (
        <span className={`badge border ${classes[severity]}`}>
            {severity.toUpperCase()}
        </span>
    );
}
