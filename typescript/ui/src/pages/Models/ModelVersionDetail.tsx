import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { modelVersionsApi } from '../../api';

export function ModelVersionDetail() {
    const { version } = useParams<{ version: string }>();

    const { data: model, isLoading, error } = useQuery({
        queryKey: ['model-version', version],
        queryFn: () => modelVersionsApi.get(version!),
        enabled: !!version,
    });

    if (isLoading) {
        return (
            <div className="text-center p-5">
                <div className="spinner-border text-primary" />
            </div>
        );
    }

    if (error || !model) {
        return (
            <div className="alert alert-danger m-4">
                Failed to load model version details.
            </div>
        );
    }

    return (
        <div className="container-fluid py-4">
            <header className="mb-4">
                <h1 className="display-6 fw-bold text-primary">Model Version Details</h1>
                <p className="font-monospace text-muted">{version}</p>
            </header>

            <div className="row g-4">
                <div className="col-md-6">
                    <div className="card shadow-sm h-100">
                        <div className="card-header bg-white">
                            <h5 className="mb-0">Overview</h5>
                        </div>
                        <div className="card-body">
                            <dl className="row mb-0">
                                <dt className="col-sm-4">Model Name</dt>
                                <dd className="col-sm-8">{model.model_name}</dd>

                                <dt className="col-sm-4">Status</dt>
                                <dd className="col-sm-8">{model.status}</dd>

                                <dt className="col-sm-4">Created</dt>
                                <dd className="col-sm-8">{model.started_at ? new Date(model.started_at).toLocaleString() : '-'}</dd>

                                {model.ended_at && (
                                    <>
                                        <dt className="col-sm-4">Completed</dt>
                                        <dd className="col-sm-8">{new Date(model.ended_at).toLocaleString()}</dd>
                                    </>
                                )}
                            </dl>
                        </div>
                    </div>
                </div>

                <div className="col-md-6">
                    <div className="card shadow-sm h-100">
                        <div className="card-header bg-white">
                            <h5 className="mb-0">Metrics</h5>
                        </div>
                        <div className="card-body">
                            <pre className="bg-light p-3 rounded small mb-0" style={{ maxHeight: '300px', overflow: 'auto' }}>
                                {model.metrics_json ? JSON.stringify(JSON.parse(model.metrics_json), null, 2) : 'No metrics data'}
                            </pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
