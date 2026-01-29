import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { backtestApi } from '../api';
import type { BacktestCompareRequest, BacktestCompareResponse } from '../types/api';

export function WhatIf() {
  const [formData, setFormData] = useState<BacktestCompareRequest>({
    base_version: '',
    candidate_version: '',
    start_date: getDefaultStartDate(),
    end_date: getDefaultEndDate(),
    rule_id: '',
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

    const request: BacktestCompareRequest = {
      ...formData,
      rule_id: formData.rule_id || undefined,
    };
    compareMutation.mutate(request);
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
      <p>Compare rule versions to understand impact before deployment</p>

      <div className="whatif-layout">
        {/* Input Form */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Comparison Parameters</h3>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label" htmlFor="base_version">
                  Base Version
                </label>
                <input
                  type="text"
                  id="base_version"
                  name="base_version"
                  className="form-input"
                  value={formData.base_version}
                  onChange={handleInputChange}
                  placeholder="e.g., v1.0.0 or current"
                  required
                />
                <small className="form-hint">
                  Current production ruleset version
                </small>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="candidate_version">
                  Candidate Version
                </label>
                <input
                  type="text"
                  id="candidate_version"
                  name="candidate_version"
                  className="form-input"
                  value={formData.candidate_version}
                  onChange={handleInputChange}
                  placeholder="e.g., v1.1.0 or draft"
                  required
                />
                <small className="form-hint">
                  New ruleset version to compare
                </small>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label" htmlFor="start_date">
                  Start Date
                </label>
                <input
                  type="date"
                  id="start_date"
                  name="start_date"
                  className="form-input"
                  value={formData.start_date}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="end_date">
                  End Date
                </label>
                <input
                  type="date"
                  id="end_date"
                  name="end_date"
                  className="form-input"
                  value={formData.end_date}
                  onChange={handleInputChange}
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="rule_id">
                Specific Rule ID (optional)
              </label>
              <input
                type="text"
                id="rule_id"
                name="rule_id"
                className="form-input"
                value={formData.rule_id}
                onChange={handleInputChange}
                placeholder="Leave empty to compare full rulesets"
              />
              <small className="form-hint">
                Compare a single rule instead of full rulesets
              </small>
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

        {/* Results */}
        {result && (
          <div className="comparison-results">
            {/* Delta Summary */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Impact Summary</h3>
              </div>
              <div className="metrics-grid">
                <DeltaMetric
                  label="Precision"
                  value={result.delta.precision}
                  format="percent"
                />
                <DeltaMetric
                  label="Recall"
                  value={result.delta.recall}
                  format="percent"
                />
                <DeltaMetric
                  label="F1 Score"
                  value={result.delta.f1_score}
                  format="percent"
                />
                <DeltaMetric
                  label="Flagged Rate"
                  value={result.delta.flagged_rate_change}
                  format="percent"
                />
              </div>
            </div>

            {/* Side-by-side comparison */}
            <div className="comparison-grid">
              <div className="comparison-section">
                <h4>Base Version ({formData.base_version})</h4>
                <MetricsDisplay metrics={result.base} />
              </div>
              <div className="comparison-section">
                <h4>Candidate Version ({formData.candidate_version})</h4>
                <MetricsDisplay metrics={result.candidate} />
              </div>
            </div>

            {result.job_id && (
              <div className="result-metadata" style={{ marginTop: '1rem' }}>
                <div className="metadata-item">
                  <span className="metadata-label">Job ID:</span>
                  <code>{result.job_id}</code>
                </div>
              </div>
            )}
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
        className={`metric-value ${
          isPositive
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
          {isPositive ? '↑ Better' : '↓ Worse'}
        </div>
      )}
    </div>
  );
}

interface MetricsDisplayProps {
  metrics: {
    precision: number;
    recall: number;
    f1_score: number;
    total_transactions: number;
    flagged_transactions: number;
    true_positives: number;
    false_positives: number;
  };
}

function MetricsDisplay({ metrics }: MetricsDisplayProps) {
  const flaggedRate =
    metrics.total_transactions > 0
      ? (metrics.flagged_transactions / metrics.total_transactions) * 100
      : 0;

  return (
    <div className="metrics-list">
      <div className="metric-row">
        <span className="metric-name">Precision</span>
        <span className="metric-val">{(metrics.precision * 100).toFixed(2)}%</span>
      </div>
      <div className="metric-row">
        <span className="metric-name">Recall</span>
        <span className="metric-val">{(metrics.recall * 100).toFixed(2)}%</span>
      </div>
      <div className="metric-row">
        <span className="metric-name">F1 Score</span>
        <span className="metric-val">{(metrics.f1_score * 100).toFixed(2)}%</span>
      </div>
      <div className="metric-row separator">
        <span className="metric-name">Total Transactions</span>
        <span className="metric-val">{metrics.total_transactions.toLocaleString()}</span>
      </div>
      <div className="metric-row">
        <span className="metric-name">Flagged Transactions</span>
        <span className="metric-val">
          {metrics.flagged_transactions.toLocaleString()} ({flaggedRate.toFixed(2)}%)
        </span>
      </div>
      <div className="metric-row">
        <span className="metric-name">True Positives</span>
        <span className="metric-val">{metrics.true_positives.toLocaleString()}</span>
      </div>
      <div className="metric-row">
        <span className="metric-name">False Positives</span>
        <span className="metric-val">{metrics.false_positives.toLocaleString()}</span>
      </div>
    </div>
  );
}

function getDefaultStartDate(): string {
  const date = new Date();
  date.setDate(date.getDate() - 7);
  return date.toISOString().split('T')[0];
}

function getDefaultEndDate(): string {
  return new Date().toISOString().split('T')[0];
}
