import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { decisionsApi } from '../../api';

export function DecisionDetail() {
    const { id } = useParams<{ id: string }>();

    const { data: decision, isLoading, error } = useQuery({
        queryKey: ['decision', id],
        queryFn: () => decisionsApi.get(id!),
        enabled: !!id,
    });

    const { data: trace } = useQuery({
        queryKey: ['decision-trace', id],
        queryFn: () => decisionsApi.getTrace(id!),
        enabled: !!id,
    });

    if (isLoading) {
        return (
            <div className="text-center p-5">
                <div className="spinner-border text-primary" />
            </div>
        );
    }

    if (error || !decision) {
        return (
            <div className="alert alert-danger m-4">
                Failed to load decision details.
            </div>
        );
    }

    return (
        <div className="container-fluid py-4">
            <header className="mb-4">
                <h1 className="display-6 fw-bold text-primary">Decision Details</h1>
                <p className="font-monospace text-muted">{id}</p>
            </header>

            <div className="row g-4">
                <div className="col-md-6">
                    <div className="card shadow-sm h-100">
                        <div className="card-header bg-white">
                            <h5 className="mb-0">Overview</h5>
                        </div>
                        <div className="card-body">
                            <dl className="row mb-0">
                                <dt className="col-sm-4">User ID</dt>
                                <dd className="col-sm-8 font-monospace">{decision.user_id}</dd>

                                <dt className="col-sm-4">Result</dt>
                                <dd className="col-sm-8">{decision.decision}</dd>

                                <dt className="col-sm-4">Score</dt>
                                <dd className="col-sm-8">{(decision.score * 100).toFixed(1)}</dd>

                                <dt className="col-sm-4">Timestamp</dt>
                                <dd className="col-sm-8">{new Date(decision.timestamp).toLocaleString()}</dd>

                                <dt className="col-sm-4">Model Version</dt>
                                <dd className="col-sm-8">{decision.model_version}</dd>
                            </dl>
                        </div>
                    </div>
                </div>

                <div className="col-md-6">
                    <div className="card shadow-sm h-100">
                        <div className="card-header bg-white">
                            <h5 className="mb-0">Features</h5>
                        </div>
                        <div className="card-body">
                            <pre className="bg-light p-3 rounded small mb-0" style={{ maxHeight: '300px', overflow: 'auto' }}>
                                {decision.features_json ? JSON.stringify(JSON.parse(decision.features_json), null, 2) : 'No features data'}
                            </pre>
                        </div>
                    </div>
                </div>

                {trace && (
                    <div className="col-12">
                        <div className="card shadow-sm">
                            <div className="card-header bg-white">
                                <h5 className="mb-0">Decision Trace</h5>
                            </div>
                            <div className="card-body">
                                <pre className="bg-light p-3 rounded small mb-0" style={{ maxHeight: '400px', overflow: 'auto' }}>
                                    {JSON.stringify(trace, null, 2)}
                                </pre>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
