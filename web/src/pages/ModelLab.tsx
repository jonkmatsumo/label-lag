import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { modelApi, healthApi } from '../api';
import type { TrainRequest, TrainResponse, DeployResponse } from '../types/api';

export function ModelLab() {
  const [trainForm, setTrainForm] = useState<TrainRequest>({
    name: '',
    test_size: 0.2,
    random_seed: 42,
  });
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [deployResult, setDeployResult] = useState<DeployResponse | null>(null);

  // Fetch current model status
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.getHealth,
    refetchInterval: 30000,
  });

  // Training mutation
  const trainMutation = useMutation({
    mutationFn: modelApi.train,
    onSuccess: (data) => {
      setTrainResult(data);
    },
  });

  // Deploy mutation
  const deployMutation = useMutation({
    mutationFn: modelApi.deploy,
    onSuccess: (data) => {
      setDeployResult(data);
      // Refresh health status after deploy
      healthQuery.refetch();
    },
  });

  const handleTrainSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setTrainResult(null);
    trainMutation.mutate(trainForm);
  };

  const handleTrainInputChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const { name, value } = e.target;
    setTrainForm((prev) => ({
      ...prev,
      [name]:
        name === 'test_size' || name === 'random_seed'
          ? parseFloat(value) || 0
          : value,
    }));
  };

  const handleDeploy = () => {
    if (trainResult?.run_id) {
      setDeployResult(null);
      deployMutation.mutate({ run_id: trainResult.run_id });
    }
  };

  return (
    <div className="page">
      <h2>Model Lab</h2>
      <p>Model training and lifecycle management</p>

      {/* Current Model Status */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Current Production Model</h3>
        </div>
        {healthQuery.isLoading ? (
          <div className="loading">Loading model status...</div>
        ) : healthQuery.isError ? (
          <div className="alert alert-error">
            Failed to fetch model status
          </div>
        ) : healthQuery.data ? (
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Model Loaded</div>
              <div className="metric-value">
                {healthQuery.data.model_loaded ? 'Yes' : 'No'}
              </div>
            </div>
            {healthQuery.data.model_version && (
              <div className="metric-card">
                <div className="metric-label">Model Version</div>
                <div className="metric-value">
                  {healthQuery.data.model_version}
                </div>
              </div>
            )}
            <div className="metric-card">
              <div className="metric-label">API Status</div>
              <div className="metric-value">
                <span
                  className={`status-badge ${
                    healthQuery.data.status === 'healthy'
                      ? 'status-published'
                      : 'status-pending'
                  }`}
                >
                  {healthQuery.data.status}
                </span>
              </div>
            </div>
          </div>
        ) : null}
      </div>

      <div className="model-lab-layout">
        {/* Training Form */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Train New Model</h3>
          </div>
          <form onSubmit={handleTrainSubmit}>
            <div className="form-group">
              <label className="form-label" htmlFor="name">
                Experiment Name (optional)
              </label>
              <input
                type="text"
                id="name"
                name="name"
                className="form-input"
                value={trainForm.name}
                onChange={handleTrainInputChange}
                placeholder="e.g., fraud-model-v2"
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="test_size">
                Test Size (0-1)
              </label>
              <input
                type="number"
                id="test_size"
                name="test_size"
                className="form-input"
                value={trainForm.test_size}
                onChange={handleTrainInputChange}
                min="0.1"
                max="0.5"
                step="0.05"
              />
              <small className="form-hint">
                Fraction of data reserved for testing (recommended: 0.2)
              </small>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="random_seed">
                Random Seed
              </label>
              <input
                type="number"
                id="random_seed"
                name="random_seed"
                className="form-input"
                value={trainForm.random_seed}
                onChange={handleTrainInputChange}
                min="0"
              />
              <small className="form-hint">
                Set for reproducible results
              </small>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={trainMutation.isPending}
            >
              {trainMutation.isPending ? 'Training...' : 'Start Training'}
            </button>

            {trainMutation.isPending && (
              <div className="alert alert-info" style={{ marginTop: '1rem' }}>
                Training in progress. This may take several minutes...
              </div>
            )}

            {trainMutation.isError && (
              <div className="alert alert-error" style={{ marginTop: '1rem' }}>
                Training failed: {trainMutation.error?.message}
              </div>
            )}
          </form>
        </div>

        {/* Training Result */}
        {trainResult && (
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Training Result</h3>
            </div>
            <div
              className={`alert ${
                trainResult.status === 'success'
                  ? 'alert-success'
                  : 'alert-error'
              }`}
            >
              {trainResult.message}
            </div>

            {trainResult.run_id && (
              <div className="result-metadata" style={{ marginTop: '1rem' }}>
                <div className="metadata-item">
                  <span className="metadata-label">Run ID:</span>
                  <code>{trainResult.run_id}</code>
                </div>
              </div>
            )}

            {trainResult.metrics && Object.keys(trainResult.metrics).length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <h4 style={{ fontSize: '0.875rem', marginBottom: '0.75rem' }}>
                  Metrics
                </h4>
                <div className="metrics-grid">
                  {Object.entries(trainResult.metrics).map(([key, value]) => (
                    <div className="metric-card" key={key}>
                      <div className="metric-label">{formatMetricName(key)}</div>
                      <div className="metric-value">
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {trainResult.status === 'success' && trainResult.run_id && (
              <div className="form-actions">
                <button
                  className="btn btn-primary"
                  onClick={handleDeploy}
                  disabled={deployMutation.isPending}
                >
                  {deployMutation.isPending
                    ? 'Deploying...'
                    : 'Deploy to Production'}
                </button>
              </div>
            )}

            {deployMutation.isError && (
              <div className="alert alert-error" style={{ marginTop: '1rem' }}>
                Deployment failed: {deployMutation.error?.message}
              </div>
            )}
          </div>
        )}

        {/* Deployment Result */}
        {deployResult && (
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Deployment Result</h3>
            </div>
            <div
              className={`alert ${
                deployResult.status === 'success'
                  ? 'alert-success'
                  : 'alert-error'
              }`}
            >
              {deployResult.message}
            </div>
            {deployResult.model_version && (
              <div className="result-metadata" style={{ marginTop: '1rem' }}>
                <div className="metadata-item">
                  <span className="metadata-label">Deployed Version:</span>
                  <code>{deployResult.model_version}</code>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function formatMetricName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
