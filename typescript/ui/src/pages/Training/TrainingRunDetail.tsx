import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { trainingApi, monitoringApi } from '../../api'; // monitoringApi for series
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';

const METRIC_WINDOW_DAYS = 90;

export function TrainingRunDetail() {
    const { id } = useParams<{ id: string }>();
    const [selectedMetric, setSelectedMetric] = useState<string>('');

    const { data: run, isLoading: isRunLoading, error: runError } = useQuery({
        queryKey: ['training-run', id],
        queryFn: () => trainingApi.get(id!),
        enabled: !!id,
    });

    // Extract available metrics from run.metrics_json
    // Extract available metrics from run.metrics_json
    let availableMetrics: string[] = [];
    if (run?.metrics_json) {
        try {
            const metrics = JSON.parse(run.metrics_json);
            availableMetrics = Object.keys(metrics);
        } catch {
            // ignore
        }
    }

    // Derived state for metric selection (default to first available)
    const metricToUse = selectedMetric || (availableMetrics.length > 0 ? availableMetrics[0] : '');

    const { data: seriesData, isLoading: isSeriesLoading } = useQuery({
        queryKey: ['metric-series', metricToUse, id],
        queryFn: () => {
            const endDate = new Date();
            const startDate = new Date();
            startDate.setDate(endDate.getDate() - METRIC_WINDOW_DAYS);

            return monitoringApi.getSeries({
                metric: metricToUse,
                start_date: startDate.toISOString(),
                end_date: endDate.toISOString(),
                tags: { run_id: id! },
            });
        },
        enabled: !!id && !!metricToUse,
    });

    if (isRunLoading) {
        return (
            <div className="text-center p-5">
                <div className="spinner-border text-primary" />
            </div>
        );
    }

    if (runError || !run) {
        return (
            <div className="alert alert-danger m-4">
                Failed to load training run details.
            </div>
        );
    }

    return (
        <div className="container-fluid py-4">
            <header className="mb-4">
                <h1 className="display-6 fw-bold text-primary">Training Run Details</h1>
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
                                <dt className="col-sm-5">Model Name</dt>
                                <dd className="col-sm-7">{run.model_name}</dd>

                                <dt className="col-sm-5">Version</dt>
                                <dd className="col-sm-7">
                                    <span className="badge bg-light text-dark border">
                                        {run.version}
                                    </span>
                                </dd>

                                <dt className="col-sm-5">Status</dt>
                                <dd className="col-sm-7">{run.status}</dd>

                                <dt className="col-sm-5">Created</dt>
                                <dd className="col-sm-7 small">
                                    {new Date(run.created_at).toLocaleString()}
                                </dd>

                                {run.started_at && (
                                    <>
                                        <dt className="col-sm-5">Started</dt>
                                        <dd className="col-sm-7 small">
                                            {new Date(run.started_at).toLocaleString()}
                                        </dd>
                                    </>
                                )}

                                {run.completed_at && (
                                    <>
                                        <dt className="col-sm-5">Completed</dt>
                                        <dd className="col-sm-7 small">
                                            {new Date(run.completed_at).toLocaleString()}
                                        </dd>
                                    </>
                                )}
                            </dl>
                        </div>
                    </div>
                </div>

                <div className="col-md-8">
                    <div className="card shadow-sm h-100">
                        <div className="card-header bg-white d-flex justify-content-between align-items-center">
                            <h5 className="mb-0">Metric Series</h5>
                            <div className="col-auto">
                                <select
                                    className="form-select form-select-sm"
                                    value={metricToUse}
                                    onChange={(e) => setSelectedMetric(e.target.value)}
                                    disabled={availableMetrics.length === 0}
                                >
                                    <option value="">Select Metric</option>
                                    {availableMetrics.map(m => (
                                        <option key={m} value={m}>{m}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                        <div className="card-body" style={{ height: '300px' }}>
                            {isSeriesLoading ? (
                                <div className="d-flex justify-content-center align-items-center h-100">
                                    <div className="spinner-border text-primary" />
                                </div>
                            ) : seriesData?.series && seriesData.series.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={seriesData.series}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis
                                            dataKey="timestamp"
                                            tickFormatter={(t) => new Date(t).toLocaleDateString()}
                                        />
                                        <YAxis />
                                        <Tooltip
                                            labelFormatter={(t) => new Date(t).toLocaleString()}
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey="value"
                                            stroke="#8884d8"
                                            dot={false}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="d-flex justify-content-center align-items-center h-100 text-muted">
                                    No series data available for selected metric
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="col-md-6">
                    <div className="card shadow-sm">
                        <div className="card-header bg-white">
                            <h5 className="mb-0">Parameters</h5>
                        </div>
                        <div className="card-body">
                            <pre className="bg-light p-3 rounded small mb-0" style={{ maxHeight: '300px', overflow: 'auto' }}>
                                {run.params_json ? JSON.stringify(JSON.parse(run.params_json), null, 2) : 'No parameters'}
                            </pre>
                        </div>
                    </div>
                </div>

                <div className="col-md-6">
                    <div className="card shadow-sm">
                        <div className="card-header bg-white">
                            <h5 className="mb-0">Metrics Summary</h5>
                        </div>
                        <div className="card-body">
                            <pre className="bg-light p-3 rounded small mb-0" style={{ maxHeight: '300px', overflow: 'auto' }}>
                                {run.metrics_json ? JSON.stringify(JSON.parse(run.metrics_json), null, 2) : 'No metrics'}
                            </pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
