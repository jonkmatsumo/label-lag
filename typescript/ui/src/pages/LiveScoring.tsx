import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { signalApi, healthApi } from '../api';
import type { SignalRequest, SignalResponse, HealthResponse } from '../types/api';
import { Clock, Cpu, AlertTriangle, Shield, ChevronDown, ChevronRight } from 'lucide-react';
import { ErrorBanner } from '../components/ErrorBanner';
import { useTenant } from '../context/TenantContext';

export function LiveScoring() {
  const [formData, setFormData] = useState<SignalRequest>({
    user_id: '',
    amount: 0,
    currency: 'USD',
    client_transaction_id: '',
  });

  const [result, setResult] = useState<SignalResponse | null>(null);
  const { tenantId } = useTenant();

  // Fetch model health status
  const healthQuery = useQuery({
    queryKey: ['health', tenantId],
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
          <ErrorBanner error={healthQuery.error} title="Failed to fetch model status" className="alert alert-danger m-3" />
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
            <div style={{ marginTop: '1rem' }}>
              <ErrorBanner error={evaluateMutation.error} title="Evaluation Error" />
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
  const [showRaw, setShowRaw] = useState(false);

  if (!result) return null;

  return (
    <div className="space-y-4">
      {/* Risk Gauge and Core Info */}
      <div className="text-center py-4 bg-light rounded-3 mb-4">
        <div className={`display-1 fw-bold mb-0 ${result.score >= 80 ? 'text-danger' : result.score >= 30 ? 'text-warning' : 'text-success'}`}>
          {result.score ?? 0}
        </div>
        <div className="text-muted small text-uppercase fw-bold tracking-wider mb-2">Risk Score</div>
        <div className={`badge rounded-pill px-3 py-2 ${result.risk_label === 'HIGH' ? 'bg-danger' : result.risk_label === 'MEDIUM' ? 'bg-warning text-dark' : 'bg-success'}`}>
          {result.risk_label ?? 'UNKNOWN'} RISK
        </div>

        <div className="mt-3 d-flex justify-content-center gap-4 text-muted small">
          <div className="d-flex align-items-center">
            <Clock size={14} className="me-1" /> {(result.latency_ms ?? 0).toFixed(1)}ms
          </div>
          <div className="d-flex align-items-center">
            <Cpu size={14} className="me-1" /> {result.model_version ?? 'unknown'}
          </div>
        </div>
      </div>

      {/* Risk Components */}
      <div className="risk-components mb-4">
        <h6 className="fw-bold mb-3 small text-uppercase text-muted d-flex align-items-center">
          <AlertTriangle size={14} className="me-2" /> Risk Factors
        </h6>
        {result.risk_components && result.risk_components.length > 0 ? (
          <div className="list-group list-group-flush border rounded overflow-hidden">
            {result.risk_components.map((component, index) => (
              <div key={index} className="list-group-item d-flex justify-content-between align-items-center p-2 small">
                <span className="fw-medium">{component.label}</span>
                <span className="text-muted" style={{fontSize: '0.8em'}}>{component.key}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-muted small italic p-2 border rounded bg-light">No risk components returned.</div>
        )}
      </div>

      {/* Matched Rules */}
      <div className="matched-rules mb-4">
        <h6 className="fw-bold mb-3 small text-uppercase text-muted d-flex align-items-center">
          <Shield size={14} className="me-2" /> Decision Rules
        </h6>
        {result.matched_rules && result.matched_rules.length > 0 ? (
          <div className="space-y-2">
            {result.matched_rules.map((rule, index) => (
              <div key={index} className="p-2 border rounded small bg-white shadow-sm border-start border-4 border-primary">
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <span className="fw-bold">{rule.rule_id}</span>
                  <span className="badge bg-primary opacity-75">{rule.severity}</span>
                </div>
                <div className="text-muted" style={{fontSize: '0.9em'}}>{rule.reason}</div>
                {rule.explanation && <div className="mt-1 p-1 bg-light rounded x-small italic text-muted">{rule.explanation}</div>}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-muted small italic p-2 border rounded bg-light">No rules matched.</div>
        )}
      </div>

      {/* Debug Section */}
      <div className="pt-3 border-top">
        <button
          className="btn btn-link btn-sm p-0 text-decoration-none text-muted d-flex align-items-center"
          onClick={() => setShowRaw(!showRaw)}
        >
          {showRaw ? <ChevronDown size={14} className="me-1" /> : <ChevronRight size={14} className="me-1" />}
          View Raw API Response
        </button>
        {showRaw && (
          <pre className="mt-2 p-3 bg-dark text-success rounded small overflow-auto" style={{ maxHeight: '300px', fontSize: '0.8em' }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
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
