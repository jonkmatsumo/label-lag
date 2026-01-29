import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { signalApi, healthApi } from '../api';
import type { SignalRequest, SignalResponse, HealthResponse } from '../types/api';

function getScoreClass(score: number): string {
  if (score < 30) return 'score-low';
  if (score < 70) return 'score-medium';
  return 'score-high';
}

export function LiveScoring() {
  const [formData, setFormData] = useState<SignalRequest>({
    user_id: '',
    amount: 0,
    currency: 'USD',
    client_transaction_id: '',
  });

  const [result, setResult] = useState<SignalResponse | null>(null);

  // Fetch model health status
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.getHealth,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Mutation for evaluating signals
  const evaluateMutation = useMutation({
    mutationFn: signalApi.evaluate,
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    evaluateMutation.mutate(formData);
  };

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'amount' ? parseFloat(value) || 0 : value,
    }));
  };

  const generateTransactionId = () => {
    const id = `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setFormData((prev) => ({ ...prev, client_transaction_id: id }));
  };

  return (
    <div className="page">
      <h2>Live Scoring</h2>
      <p>Real-time transaction risk assessment</p>

      {/* Model Status */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Model Status</h3>
        </div>
        {healthQuery.isLoading ? (
          <div className="loading">Loading model status...</div>
        ) : healthQuery.isError ? (
          <div className="alert alert-error">
            Failed to fetch model status: {healthQuery.error?.message}
          </div>
        ) : healthQuery.data ? (
          <ModelStatus health={healthQuery.data} />
        ) : null}
      </div>

      <div className="scoring-layout">
        {/* Input Form */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Transaction Details</h3>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label" htmlFor="user_id">
                User ID
              </label>
              <input
                type="text"
                id="user_id"
                name="user_id"
                className="form-input"
                value={formData.user_id}
                onChange={handleInputChange}
                placeholder="e.g., user_12345"
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="amount">
                Amount
              </label>
              <input
                type="number"
                id="amount"
                name="amount"
                className="form-input"
                value={formData.amount}
                onChange={handleInputChange}
                min="0"
                step="0.01"
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="currency">
                Currency
              </label>
              <select
                id="currency"
                name="currency"
                className="form-input"
                value={formData.currency}
                onChange={handleInputChange}
              >
                <option value="USD">USD</option>
                <option value="EUR">EUR</option>
                <option value="GBP">GBP</option>
                <option value="CAD">CAD</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="client_transaction_id">
                Transaction ID
              </label>
              <div className="input-with-button">
                <input
                  type="text"
                  id="client_transaction_id"
                  name="client_transaction_id"
                  className="form-input"
                  value={formData.client_transaction_id}
                  onChange={handleInputChange}
                  placeholder="e.g., txn_abc123"
                  required
                />
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={generateTransactionId}
                >
                  Generate
                </button>
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={evaluateMutation.isPending}
            >
              {evaluateMutation.isPending ? 'Evaluating...' : 'Evaluate Risk'}
            </button>
          </form>

          {evaluateMutation.isError && (
            <div className="alert alert-error" style={{ marginTop: '1rem' }}>
              Error: {evaluateMutation.error?.message}
            </div>
          )}
        </div>

        {/* Results */}
        {result && (
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Risk Assessment</h3>
            </div>
            <RiskResult result={result} />
          </div>
        )}
      </div>
    </div>
  );
}

function ModelStatus({ health }: { health: HealthResponse }) {
  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div className="metric-label">Status</div>
        <div className="metric-value">
          <span
            className={`status-badge ${
              health.status === 'healthy'
                ? 'status-published'
                : health.status === 'degraded'
                ? 'status-pending'
                : 'status-rejected'
            }`}
          >
            {health.status}
          </span>
        </div>
      </div>
      <div className="metric-card">
        <div className="metric-label">Model Loaded</div>
        <div className="metric-value">
          {health.model_loaded ? 'Yes' : 'No'}
        </div>
      </div>
      {health.model_version && (
        <div className="metric-card">
          <div className="metric-label">Model Version</div>
          <div className="metric-value">{health.model_version}</div>
        </div>
      )}
      {health.uptime_seconds !== undefined && (
        <div className="metric-card">
          <div className="metric-label">Uptime</div>
          <div className="metric-value">
            {formatUptime(health.uptime_seconds)}
          </div>
        </div>
      )}
    </div>
  );
}

function RiskResult({ result }: { result: SignalResponse }) {
  return (
    <div>
      {/* Main Score */}
      <div className="score-display">
        <div className="score-label">Risk Score</div>
        <div className={`score-value ${getScoreClass(result.score)}`}>
          {result.score}
        </div>
      </div>

      {/* Metadata */}
      <div className="result-metadata">
        <div className="metadata-item">
          <span className="metadata-label">Request ID:</span>
          <code>{result.request_id}</code>
        </div>
        <div className="metadata-item">
          <span className="metadata-label">Model Version:</span>
          <code>{result.model_version}</code>
        </div>
      </div>

      {/* Risk Components */}
      {result.risk_components.length > 0 && (
        <div className="risk-components">
          <h4>Risk Components</h4>
          <table className="table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Score</th>
                <th>Weight</th>
              </tr>
            </thead>
            <tbody>
              {result.risk_components.map((component, index) => (
                <tr key={index}>
                  <td>{component.name}</td>
                  <td>{component.score.toFixed(2)}</td>
                  <td>{(component.weight * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Matched Rules */}
      {result.matched_rules.length > 0 && (
        <div className="matched-rules">
          <h4>Matched Rules</h4>
          <table className="table">
            <thead>
              <tr>
                <th>Rule</th>
                <th>Action</th>
                <th>Adjustment</th>
              </tr>
            </thead>
            <tbody>
              {result.matched_rules.map((rule, index) => (
                <tr key={index}>
                  <td>{rule.name}</td>
                  <td>
                    <span className="status-badge status-pending">
                      {rule.action}
                    </span>
                  </td>
                  <td>
                    {rule.score_adjustment !== undefined
                      ? (rule.score_adjustment > 0 ? '+' : '') +
                        rule.score_adjustment
                      : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Shadow Matched Rules */}
      {result.shadow_matched_rules && result.shadow_matched_rules.length > 0 && (
        <div className="shadow-rules">
          <h4>Shadow Rules (Testing)</h4>
          <table className="table">
            <thead>
              <tr>
                <th>Rule</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {result.shadow_matched_rules.map((rule, index) => (
                <tr key={index}>
                  <td>{rule.name}</td>
                  <td>
                    <span className="status-badge status-draft">
                      {rule.action}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}
