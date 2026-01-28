import { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { rulesApi } from '../api';
import type {
  DraftRule,
  SandboxEvaluateRequest,
  SandboxEvaluateResponse,
} from '../types/api';

const ruleTabs = [
  { path: '/rules', label: 'Management', exact: true },
  { path: '/rules/sandbox', label: 'Sandbox' },
  { path: '/rules/shadow', label: 'Shadow Metrics' },
  { path: '/rules/backtests', label: 'Backtests' },
  { path: '/rules/suggestions', label: 'Suggestions' },
];

export function RuleInspector() {
  const location = useLocation();

  return (
    <div className="page">
      <h2>Rule Inspector</h2>
      <div className="tabs">
        {ruleTabs.map((tab) => {
          const isActive = tab.exact
            ? location.pathname === tab.path
            : location.pathname.startsWith(tab.path);

          return (
            <Link
              key={tab.path}
              to={tab.path}
              className={`tab ${isActive ? 'active' : ''}`}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
      <div className="tab-content">
        <Outlet />
      </div>
    </div>
  );
}

export function RuleManagement() {
  const queryClient = useQueryClient();

  // Fetch draft rules
  const rulesQuery = useQuery({
    queryKey: ['rules', 'draft'],
    queryFn: rulesApi.getDraftRules,
  });

  // Publish mutation
  const publishMutation = useMutation({
    mutationFn: (ruleId: string) => rulesApi.publishRule(ruleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rules', 'draft'] });
    },
  });

  const handlePublish = (ruleId: string) => {
    if (confirm('Are you sure you want to publish this rule to production?')) {
      publishMutation.mutate(ruleId);
    }
  };

  const getStatusBadgeClass = (status: DraftRule['status']) => {
    switch (status) {
      case 'draft':
        return 'status-draft';
      case 'pending_approval':
        return 'status-pending';
      case 'approved':
        return 'status-approved';
      case 'published':
        return 'status-published';
      case 'rejected':
        return 'status-rejected';
      default:
        return 'status-draft';
    }
  };

  return (
    <div>
      <div className="section-header">
        <h3>Draft Rules</h3>
        <p>Manage rule lifecycle and publish approved rules to production</p>
      </div>

      {rulesQuery.isLoading ? (
        <div className="loading">Loading rules...</div>
      ) : rulesQuery.isError ? (
        <div className="alert alert-error">
          Failed to load rules: {rulesQuery.error?.message}
        </div>
      ) : rulesQuery.data?.rules.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">📋</div>
          <div className="empty-state-title">No draft rules</div>
          <p>Create a new rule to get started</p>
        </div>
      ) : (
        <div className="card">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Condition</th>
                <th>Action</th>
                <th>Updated</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {rulesQuery.data?.rules.map((rule) => (
                <tr key={rule.id}>
                  <td>
                    <div className="rule-name">{rule.name}</div>
                    {rule.description && (
                      <div className="rule-description">{rule.description}</div>
                    )}
                  </td>
                  <td>
                    <span
                      className={`status-badge ${getStatusBadgeClass(
                        rule.status
                      )}`}
                    >
                      {rule.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td>
                    <code className="rule-condition">{rule.condition}</code>
                  </td>
                  <td>
                    <span className="rule-action">{rule.action}</span>
                    {rule.score_adjustment !== undefined && (
                      <span className="score-adjustment">
                        {' '}
                        ({rule.score_adjustment > 0 ? '+' : ''}
                        {rule.score_adjustment})
                      </span>
                    )}
                  </td>
                  <td className="date-cell">
                    {new Date(rule.updated_at).toLocaleDateString()}
                  </td>
                  <td>
                    {rule.status === 'approved' && (
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={() => handlePublish(rule.id)}
                        disabled={publishMutation.isPending}
                      >
                        {publishMutation.isPending &&
                        publishMutation.variables === rule.id
                          ? 'Publishing...'
                          : 'Publish'}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {publishMutation.isSuccess && (
        <div className="alert alert-success" style={{ marginTop: '1rem' }}>
          Rule published successfully!
        </div>
      )}

      {publishMutation.isError && (
        <div className="alert alert-error" style={{ marginTop: '1rem' }}>
          Failed to publish rule: {publishMutation.error?.message}
        </div>
      )}
    </div>
  );
}

export function RuleSandbox() {
  const [formData, setFormData] = useState<SandboxEvaluateRequest>({
    base_score: 50,
    features: {},
  });
  const [featuresJson, setFeaturesJson] = useState('{\n  \n}');
  const [result, setResult] = useState<SandboxEvaluateResponse | null>(null);
  const [jsonError, setJsonError] = useState<string | null>(null);

  // Sandbox evaluate mutation
  const evaluateMutation = useMutation({
    mutationFn: rulesApi.sandboxEvaluate,
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setJsonError(null);

    try {
      const features = JSON.parse(featuresJson);
      evaluateMutation.mutate({
        ...formData,
        features,
      });
    } catch {
      setJsonError('Invalid JSON format');
    }
  };

  const loadSampleFeatures = () => {
    const sample = {
      amount: 5000,
      velocity_24h: 15,
      amount_to_avg_ratio_30d: 2.5,
      is_new_device: true,
      hour_of_day: 3,
      days_since_last_transaction: 45,
    };
    setFeaturesJson(JSON.stringify(sample, null, 2));
  };

  return (
    <div>
      <div className="section-header">
        <h3>Sandbox Evaluation</h3>
        <p>Test rules against sample transactions without affecting production</p>
      </div>

      <div className="sandbox-layout">
        {/* Input Form */}
        <div className="card">
          <div className="card-header">
            <h4 className="card-title">Test Input</h4>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label" htmlFor="base_score">
                Base Score (1-99)
              </label>
              <input
                type="number"
                id="base_score"
                name="base_score"
                className="form-input"
                value={formData.base_score}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    base_score: parseInt(e.target.value) || 50,
                  }))
                }
                min="1"
                max="99"
              />
              <small className="form-hint">
                Initial risk score before rule evaluation
              </small>
            </div>

            <div className="form-group">
              <div className="label-with-action">
                <label className="form-label" htmlFor="features">
                  Features (JSON)
                </label>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={loadSampleFeatures}
                >
                  Load Sample
                </button>
              </div>
              <textarea
                id="features"
                name="features"
                className="form-input json-input"
                value={featuresJson}
                onChange={(e) => setFeaturesJson(e.target.value)}
                rows={10}
              />
              {jsonError && (
                <small className="form-error">{jsonError}</small>
              )}
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={evaluateMutation.isPending}
            >
              {evaluateMutation.isPending ? 'Evaluating...' : 'Evaluate Rules'}
            </button>

            {evaluateMutation.isError && (
              <div className="alert alert-error" style={{ marginTop: '1rem' }}>
                Evaluation failed: {evaluateMutation.error?.message}
              </div>
            )}
          </form>
        </div>

        {/* Results */}
        {result && (
          <div className="card">
            <div className="card-header">
              <h4 className="card-title">Evaluation Result</h4>
            </div>

            {/* Final Score */}
            <div className="score-comparison">
              <div className="score-item">
                <div className="score-label">Base Score</div>
                <div className="score-value score-medium">
                  {formData.base_score}
                </div>
              </div>
              <div className="score-arrow">→</div>
              <div className="score-item">
                <div className="score-label">Final Score</div>
                <div
                  className={`score-value ${
                    result.final_score < 30
                      ? 'score-low'
                      : result.final_score < 70
                      ? 'score-medium'
                      : 'score-high'
                  }`}
                >
                  {result.final_score}
                </div>
              </div>
            </div>

            {/* Matched Rules */}
            {result.matched_rules.length > 0 ? (
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
                          <span className="status-badge status-published">
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
            ) : (
              <div className="alert alert-info">No rules matched</div>
            )}

            {/* Shadow Matched Rules */}
            {result.shadow_matched_rules.length > 0 && (
              <div className="shadow-rules">
                <h4>Shadow Rules (Would Match)</h4>
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
        )}
      </div>
    </div>
  );
}

export function RuleShadow() {
  return (
    <div>
      <div className="section-header">
        <h3>Shadow Metrics</h3>
        <p>Compare production vs shadow mode performance</p>
      </div>
      <div className="empty-state">
        <div className="empty-state-icon">📊</div>
        <div className="empty-state-title">Coming Soon</div>
        <p>Shadow metrics comparison will be available in the next release</p>
      </div>
    </div>
  );
}

export function RuleBacktests() {
  return (
    <div>
      <div className="section-header">
        <h3>Backtests</h3>
        <p>Run and view historical backtest results</p>
      </div>
      <div className="empty-state">
        <div className="empty-state-icon">⏱️</div>
        <div className="empty-state-title">Coming Soon</div>
        <p>Backtest management will be available in the next release</p>
      </div>
    </div>
  );
}

export function RuleSuggestions() {
  return (
    <div>
      <div className="section-header">
        <h3>AI Suggestions</h3>
        <p>ML-generated rule recommendations based on fraud patterns</p>
      </div>
      <div className="empty-state">
        <div className="empty-state-icon">🤖</div>
        <div className="empty-state-title">Coming Soon</div>
        <p>AI-powered rule suggestions will be available in the next release</p>
      </div>
    </div>
  );
}
