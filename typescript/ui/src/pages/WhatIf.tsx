import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { backtestApi } from '../api';
import type { BacktestCompareRequest, BacktestCompareResponse, BacktestMetrics } from '../types/api';

export function WhatIf() {
  const [formData, setFormData] = useState<BacktestCompareRequest>({
    baseline_job_id: '',
    candidate_job_id: '',
    tenant_id: 'default', // TODO: useTenant() hook
  });

  const [result, setResult] = useState<BacktestCompareResponse | null>(null);

  // Compare mutation
  const compareMutation = useMutation({
    mutationFn: backtestApi.compare,
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setResult(null);
    compareMutation.mutate(formData);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  return (
    <div className="page">
      <h2>What-If Simulation</h2>
      <p>Compare backtest jobs to understand impact</p>

      <div className="whatif-layout">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Comparison Parameters</h3>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label" htmlFor="baseline_job_id">
                  Baseline Job ID
                </label>
                <input
                  type="text"
                  id="baseline_job_id"
                  name="baseline_job_id"
                  className="form-input"
                  value={formData.baseline_job_id}
                  onChange={handleInputChange}
                  placeholder="e.g., job-123"
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="candidate_job_id">
                  Candidate Job ID
                </label>
                <input
                  type="text"
                  id="candidate_job_id"
                  name="candidate_job_id"
                  className="form-input"
                  value={formData.candidate_job_id}
                  onChange={handleInputChange}
                  placeholder="e.g., job-456"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={compareMutation.isPending}
            >
              {compareMutation.isPending ? 'Running Comparison...' : 'Run Comparison'}
            </button>

            {compareMutation.isPending && (
              <div className="alert alert-info" style={{ marginTop: '1rem' }}>
                Running backtest comparison. This may take a moment...
              </div>
            )}

            {compareMutation.isError && (
              <div className="alert alert-error" style={{ marginTop: '1rem' }}>
                Comparison failed: {compareMutation.error?.message}
              </div>
            )}
          </form>
        </div>

        {result && (
          <div className="comparison-results">
            {/* Delta Summary */}
            {result.delta && (
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Impact Summary</h3>
                </div>
                <div className="metrics-grid">
                  <DeltaMetric
                    label="Match Rate"
                    value={result.delta.match_rate_delta}
                    format="percent"
                  />
                  <DeltaMetric
                    label="Score Mean"
                    value={result.delta.score_mean_delta}
                    format="number"
                  />
                  <DeltaMetric
                    label="Rejected Rate"
                    value={result.delta.rejected_rate_delta}
                    format="percent"
                  />
                </div>
              </div>
            )}

            {/* Side-by-side comparison */}
            <div className="comparison-grid">
              <div className="comparison-section">
                <h4>Base Job ({formData.baseline_job_id})</h4>
                {result.baseline?.metrics && <MetricsDisplay metrics={result.baseline.metrics} />}
              </div>
              <div className="comparison-section">
                <h4>Candidate Job ({formData.candidate_job_id})</h4>
                {result.candidate?.metrics && <MetricsDisplay metrics={result.candidate.metrics} />}
              </div>
            </div>

            {/* No job_id in CompareBacktestsResponse */}
          </div>
        )}
      </div>
    </div>
  );
}

interface DeltaMetricProps {
  label: string;
  value: number;
  format: 'percent' | 'number';
}

function DeltaMetric({ label, value, format }: DeltaMetricProps) {
  const isPositive = value > 0;
  const isNegative = value < 0;
  const isNeutral = value === 0;

  const formattedValue =
    format === 'percent'
      ? `${(value * 100).toFixed(2)}%`
      : value.toFixed(4);

  const displayValue = isPositive ? `+${formattedValue}` : formattedValue;

  return (
    <div className="metric-card">
      <div className="metric-label">{label} Change</div>
      <div
        className={`metric-value ${isPositive
          ? 'delta-positive'
          : isNegative
            ? 'delta-negative'
            : 'delta-neutral'
          }`}
      >
        {displayValue}
      </div>
      {!isNeutral && (
        <div className="metric-indicator">
          {/* Context dependent interpretation of 'Better' */}
          Change
        </div>
      )}
    </div>
  );
}

interface MetricsDisplayProps {
  metrics: BacktestMetrics;
}

function MetricsDisplay({ metrics }: MetricsDisplayProps) {
  return (
    <div className="metrics-list">
      <div className="metric-row">
        <span className="metric-name">Match Rate</span>
        <span className="metric-val">{(metrics.match_rate * 100).toFixed(2)}%</span>
      </div>
      <div className="metric-row">
        <span className="metric-name">Score Mean</span>
        <span className="metric-val">{metrics.score_mean.toFixed(4)}</span>
      </div>
      <div className="metric-row separator">
        <span className="metric-name">Total Transactions</span>
        <span className="metric-val">{Number(metrics.total_records).toLocaleString()}</span>
      </div>
      <div className="metric-row">
        <span className="metric-name">Matches</span>
        <span className="metric-val">
          {Number(metrics.matched_count).toLocaleString()}
        </span>
      </div>
      <div className="metric-row text-danger">
        <span className="metric-name">Rejected</span>
        <span className="metric-val">{Number(metrics.rejected_count).toLocaleString()} ({metrics.rejected_rate.toFixed(2)}%)</span>
      </div>
    </div>
  );
}
