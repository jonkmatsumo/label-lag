import { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { rulesApi, monitoringApi, backtestApi, suggestionsApi } from '../api';
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
  const [dateRange, setDateRange] = useState(() => {
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - 7);
    return {
      start: start.toISOString().split('T')[0],
      end: end.toISOString().split('T')[0],
    };
  });

  const shadowQuery = useQuery({
    queryKey: ['shadow-comparison', dateRange.start, dateRange.end],
    queryFn: () => monitoringApi.getShadowComparison(dateRange.start, dateRange.end),
  });

  const handleDateChange = (field: 'start' | 'end', value: string) => {
    setDateRange((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div>
      <div className="section-header">
        <h3>Shadow Metrics</h3>
        <p>Compare production vs shadow mode performance</p>
      </div>

      {/* Date Range Filter */}
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="filter-row">
          <div className="form-group inline">
            <label className="form-label">Start Date</label>
            <input
              type="date"
              className="form-input"
              value={dateRange.start}
              onChange={(e) => handleDateChange('start', e.target.value)}
            />
          </div>
          <div className="form-group inline">
            <label className="form-label">End Date</label>
            <input
              type="date"
              className="form-input"
              value={dateRange.end}
              onChange={(e) => handleDateChange('end', e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Results */}
      {shadowQuery.isLoading ? (
        <div className="loading">Loading shadow metrics...</div>
      ) : shadowQuery.isError ? (
        <div className="alert alert-error">
          Failed to load shadow metrics: {shadowQuery.error?.message}
        </div>
      ) : shadowQuery.data && shadowQuery.data.rule_metrics.length > 0 ? (
        <div className="card">
          <div className="card-header">
            <h4 className="card-title">Rule Comparison</h4>
            <span className="text-muted">
              {shadowQuery.data.total_requests.toLocaleString()} total requests
            </span>
          </div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Rule ID</th>
                  <th style={{ textAlign: 'right' }}>Production</th>
                  <th style={{ textAlign: 'right' }}>Shadow</th>
                  <th style={{ textAlign: 'right' }}>Overlap</th>
                  <th style={{ textAlign: 'right' }}>Prod Only</th>
                  <th style={{ textAlign: 'right' }}>Shadow Only</th>
                </tr>
              </thead>
              <tbody>
                {shadowQuery.data.rule_metrics.map((metric) => (
                  <tr key={metric.rule_id}>
                    <td><code>{metric.rule_id}</code></td>
                    <td style={{ textAlign: 'right' }}>{metric.production_matches.toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>{metric.shadow_matches.toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>{metric.overlap_count.toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      <span className={metric.production_only_count > 0 ? 'text-warning' : ''}>
                        {metric.production_only_count.toLocaleString()}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <span className={metric.shadow_only_count > 0 ? 'text-info' : ''}>
                        {metric.shadow_only_count.toLocaleString()}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-state-icon">📊</div>
          <div className="empty-state-title">No shadow data</div>
          <p>No shadow metrics available for the selected date range</p>
        </div>
      )}
    </div>
  );
}

export function RuleBacktests() {
  const [ruleFilter, setRuleFilter] = useState('');

  const backtestsQuery = useQuery({
    queryKey: ['backtest-results', ruleFilter],
    queryFn: () => backtestApi.listResults({ rule_id: ruleFilter || undefined, limit: 50 }),
  });

  const getStatusBadgeClass = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
      case 'success':
        return 'status-published';
      case 'running':
      case 'pending':
        return 'status-pending';
      case 'failed':
      case 'error':
        return 'status-rejected';
      default:
        return 'status-draft';
    }
  };

  return (
    <div>
      <div className="section-header">
        <h3>Backtests</h3>
        <p>View historical backtest results</p>
      </div>

      {/* Filter */}
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="filter-row">
          <div className="form-group inline">
            <label className="form-label">Filter by Rule ID</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g., rule-001"
              value={ruleFilter}
              onChange={(e) => setRuleFilter(e.target.value)}
              style={{ width: '200px' }}
            />
          </div>
        </div>
      </div>

      {/* Results */}
      {backtestsQuery.isLoading ? (
        <div className="loading">Loading backtest results...</div>
      ) : backtestsQuery.isError ? (
        <div className="alert alert-error">
          Failed to load backtests: {backtestsQuery.error?.message}
        </div>
      ) : backtestsQuery.data && backtestsQuery.data.results.length > 0 ? (
        <div className="card">
          <div className="card-header">
            <h4 className="card-title">Results</h4>
            <span className="text-muted">{backtestsQuery.data.total} total</span>
          </div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Rule</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th style={{ textAlign: 'right' }}>Precision</th>
                  <th style={{ textAlign: 'right' }}>Recall</th>
                  <th style={{ textAlign: 'right' }}>F1</th>
                </tr>
              </thead>
              <tbody>
                {backtestsQuery.data.results.map((result) => (
                  <tr key={result.id}>
                    <td><code>{result.id.slice(0, 8)}</code></td>
                    <td><code>{result.rule_id}</code></td>
                    <td>
                      <span className={`status-badge ${getStatusBadgeClass(result.status)}`}>
                        {result.status}
                      </span>
                    </td>
                    <td>{new Date(result.created_at).toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      {result.metrics?.precision?.toFixed(3) ?? '-'}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      {result.metrics?.recall?.toFixed(3) ?? '-'}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      {result.metrics?.f1_score?.toFixed(3) ?? '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-state-icon">⏱️</div>
          <div className="empty-state-title">No backtests found</div>
          <p>
            {ruleFilter
              ? `No backtest results for rule "${ruleFilter}"`
              : 'No backtest results available. Run a backtest from What-If Simulation.'}
          </p>
        </div>
      )}
    </div>
  );
}

export function RuleSuggestions() {
  const [minConfidence, setMinConfidence] = useState(0.7);
  const queryClient = useQueryClient();

  const suggestionsQuery = useQuery({
    queryKey: ['suggestions', minConfidence],
    queryFn: () => suggestionsApi.getHeuristic({ min_confidence: minConfidence }),
  });

  const acceptMutation = useMutation({
    mutationFn: suggestionsApi.accept,
    onSuccess: () => {
      alert('Suggestion accepted! A draft rule has been created.');
      // Refresh draft rules if we were showing them
      queryClient.invalidateQueries({ queryKey: ['rules', 'draft'] });
    },
    onError: (err) => {
      alert(`Failed to accept: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  });

  return (
    <div>
      <div className="section-header">
        <h3>AI Suggestions</h3>
        <p>ML-generated rule recommendations based on fraud patterns</p>
      </div>

      <div className="card mb-3">
        <div className="card-body">
          <label className="form-label me-2">Minimum Confidence:</label>
          <input 
            type="range" className="form-range" 
            min="0.5" max="0.95" step="0.05" 
            value={minConfidence} onChange={e => setMinConfidence(parseFloat(e.target.value))}
            style={{maxWidth: '300px', display: 'inline-block', verticalAlign: 'middle'}}
          />
          <span className="ms-2 fw-bold">{minConfidence.toFixed(2)}</span>
        </div>
      </div>

      {suggestionsQuery.isLoading ? (
        <div className="loading">Analyzing patterns...</div>
      ) : suggestionsQuery.isError ? (
         <div className="alert alert-error">Analysis failed: {suggestionsQuery.error?.message}</div>
      ) : suggestionsQuery.data && suggestionsQuery.data.suggestions.length > 0 ? (
        <div className="row">
           {suggestionsQuery.data.suggestions.map((s: any, idx: number) => (
             <div className="col-md-6 mb-3" key={idx}>
               <div className="card h-100">
                 <div className="card-header d-flex justify-content-between align-items-center">
                   <span className="fw-bold">{s.field} {s.operator} {s.threshold}</span>
                   <span className="badge bg-primary">{(s.confidence * 100).toFixed(0)}% Conf</span>
                 </div>
                 <div className="card-body">
                   <p className="card-text small text-muted">{s.reason}</p>
                   <ul className="small text-muted mb-3">
                     <li>Action: <strong>{s.action}</strong></li>
                     <li>Score: <strong>{s.suggested_score}</strong></li>
                     {s.evidence && (
                       <li>Evidence: Mean {s.evidence.mean?.toFixed(2)}, Count {s.evidence.sample_count}</li>
                     )}
                   </ul>
                   <button 
                     className="btn btn-outline-primary btn-sm w-100"
                     onClick={() => acceptMutation.mutate({ 
                       suggestion: s, 
                       actor: 'user', 
                       custom_id: `suggest_${Date.now()}_${idx}` 
                     })}
                     disabled={acceptMutation.isPending}
                   >
                     Accept & Create Draft
                   </button>
                 </div>
               </div>
             </div>
           ))}
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-state-icon">🤖</div>
          <div className="empty-state-title">No suggestions found</div>
          <p>Try lowering the confidence threshold or generating more fraud data.</p>
        </div>
      )}
    </div>
  );
}
