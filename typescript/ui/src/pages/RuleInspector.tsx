import React, { useState } from 'react';
import { Outlet, Link, useLocation, useSearchParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { rulesApi, monitoringApi, backtestApi, suggestionsApi, analyticsApi } from '../api';
import type {
  DraftRule,
  SandboxEvaluateRequest,
  SandboxEvaluateResponse,
  ApprovalSignalItem,
  RuleSuggestion,
  Rule,
  ReadinessCheck,
  BacktestResult,
} from '../types/api';
import {
  AlertTriangle, CheckCircle, Info, Shield,
  ChevronRight, ChevronDown, User, FileText, Send,
  History, BarChart2, Diff, ArrowRight, Clock
} from 'lucide-react';
import {
  Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ComposedChart, Line, Area, AreaChart
} from 'recharts';
import { ErrorBanner } from '../components/ErrorBanner';
import { DataQualityBadge } from '../components/DataQualityBadge';
import { useTenant } from '../hooks/useTenant';
import {
  canPublishFromReadiness,
  toCheckReadinessStatus,
  toOverallReadinessStatus,
  type ReadinessDisplayStatus,
} from './ruleReadiness';
import { hasBreakingChanges } from './ruleDiff';

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
      <header className="mb-4">
        <h2 className="display-6 fw-bold text-primary">Rule Inspector</h2>
        <div className="alert alert-warning border-0 shadow-sm d-flex align-items-center">
          <Shield size={20} className="me-2" />
          <div>
            <strong>Read-Only Inspection Mode</strong> — Exploration is safe. Production changes require explicit audit trails.
          </div>
        </div>
      </header>

      <div className="tabs mb-4">
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
  const { tenantId } = useTenant();
  const [expandedRule, setExpandedRule] = useState<string | null>(null);

  // Fetch draft rules
  const rulesQuery = useQuery({
    queryKey: ['rules', tenantId, 'draft'],
    queryFn: rulesApi.getDraftRules,
  });

  const getStatusBadgeClass = (status: DraftRule['status']) => {
    switch (status) {
      case 'draft': return 'bg-secondary';
      case 'pending_approval': return 'bg-warning text-dark';
      case 'approved': return 'bg-info text-white';
      case 'published': return 'bg-success';
      case 'rejected': return 'bg-danger';
      default: return 'bg-light text-dark';
    }
  };

  return (
    <div>
      <div className="section-header mb-4">
        <h3>Draft Rules</h3>
        <p className="text-muted">Manage rule lifecycle, review safety signals, and publish approved rules.</p>
      </div>

      {rulesQuery.isLoading ? (
        <div className="text-center p-5"><div className="spinner-border text-primary" /></div>
      ) : rulesQuery.isError ? (
        <ErrorBanner error={rulesQuery.error} title="Failed to load rules" className="alert alert-danger mb-4" />
      ) : rulesQuery.data?.rules.length === 0 ? (
        <div className="empty-state text-center py-5">
          <div className="display-1 text-muted mb-3">📋</div>
          <h4>No draft rules</h4>
          <p>Create a new rule to get started</p>
        </div>
      ) : (
        <div className="card shadow-sm border-0">
          <div className="table-responsive">
            <table className="table table-hover align-middle mb-0">
              <thead className="table-light">
                <tr>
                  <th style={{ width: '40px' }}></th>
                  <th>Rule Name</th>
                  <th>Status</th>
                  <th>Condition</th>
                  <th>Action</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {rulesQuery.data?.rules.map((rule: Rule) => (
                  <React.Fragment key={rule.id}>
                    <tr
                      onClick={() => setExpandedRule(expandedRule === rule.id ? null : rule.id)}
                      style={{ cursor: 'pointer' }}
                      className={expandedRule === rule.id ? 'table-active' : ''}
                    >
                      <td className="text-center">
                        {expandedRule === rule.id ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                      </td>
                      <td>
                        <div className="fw-bold">{rule.id}</div>
                        <div className="small text-muted text-truncate" style={{ maxWidth: '250px' }}>{rule.reason}</div>
                      </td>
                      <td>
                        <span className={`badge rounded-pill ${getStatusBadgeClass(rule.status)}`}>
                          {rule.status.replace('_', ' ')}
                        </span>
                      </td>
                      <td><code className="bg-light px-2 py-1 rounded small">{rule.field} {rule.op} {rule.value_json}</code></td>
                      <td>
                        <span className="small fw-medium">{rule.action}</span>
                        {rule.score !== undefined && (
                          <span className={rule.score > 0 ? 'text-danger ms-1' : 'text-success ms-1'}>
                            ({rule.score > 0 ? '+' : ''}{rule.score})
                          </span>
                        )}
                      </td>
                      <td className="small text-muted">
                        -
                      </td>
                    </tr>
                    {expandedRule === rule.id && (
                      <tr>
                        <td colSpan={6} className="p-0 border-0 bg-light bg-opacity-50">
                          <div className="p-4 border-start border-4 border-primary m-3 bg-white shadow-sm rounded">
                            <RuleDetail rule={rule} onPublished={() => {
                              setExpandedRule(null);
                              rulesQuery.refetch();
                            }} />
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function RuleDetail({ rule, onPublished }: { rule: DraftRule, onPublished: () => void }) {
  const [activeTab, setActiveTab] = useState<'overview' | 'history' | 'impact'>('overview');
  const [showPublishModal, setShowPublishModal] = useState(false);

  return (
    <div>
      <div className="d-flex border-bottom mb-4">
        <button
          className={`btn btn-sm px-3 py-2 rounded-0 border-bottom border-3 ${activeTab === 'overview' ? 'border-primary fw-bold text-primary' : 'border-transparent text-muted'}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={`btn btn-sm px-3 py-2 rounded-0 border-bottom border-3 ${activeTab === 'history' ? 'border-primary fw-bold text-primary' : 'border-transparent text-muted'}`}
          onClick={() => setActiveTab('history')}
        >
          <History size={14} className="me-1" /> Version History
        </button>
        <button
          className={`btn btn-sm px-3 py-2 rounded-0 border-bottom border-3 ${activeTab === 'impact' ? 'border-primary fw-bold text-primary' : 'border-transparent text-muted'}`}
          onClick={() => setActiveTab('impact')}
        >
          <BarChart2 size={14} className="me-1" /> Impact Analysis
        </button>
      </div>

      {activeTab === 'overview' && <RuleOverviewTab rule={rule} onShowPublish={() => setShowPublishModal(true)} />}
      {activeTab === 'history' && <RuleHistoryTab ruleId={rule.id} />}
      {activeTab === 'impact' && <RuleImpactTab ruleId={rule.id} />}

      {showPublishModal && (
        <PublishModal
          rule={rule}
          onClose={() => setShowPublishModal(false)}
          onSuccess={onPublished}
        />
      )}
    </div>
  );
}

function RuleOverviewTab({ rule, onShowPublish }: { rule: DraftRule, onShowPublish: () => void }) {
  const { tenantId } = useTenant();
  const readinessQuery = useQuery({
    queryKey: ['rules', tenantId, rule.id, 'readiness'],
    queryFn: () => rulesApi.getReadiness(rule.id),
    enabled: !!rule.id
  });

  const signalsQuery = useQuery({
    queryKey: ['rules', tenantId, rule.id, 'signals'],
    queryFn: () => rulesApi.getApprovalSignals(rule.id),
    enabled: !!rule.id && (rule.status === 'pending_approval' || rule.status === 'approved')
  });

  const overallReadinessStatus = toOverallReadinessStatus(readinessQuery.data);
  const isReady = canPublishFromReadiness(readinessQuery.data);
  const readinessAlertClass =
    overallReadinessStatus === 'pass'
      ? 'alert-success'
      : overallReadinessStatus === 'warn'
        ? 'alert-warning'
        : overallReadinessStatus === 'fail'
          ? 'alert-danger'
          : 'alert-secondary';
  const readinessLabel = overallReadinessStatus.toUpperCase();

  return (
    <div className="row g-4">
      <div className="col-md-6">
        <h6 className="fw-bold mb-3 d-flex align-items-center small text-uppercase tracking-wider text-muted">
          <Shield size={14} className="me-2" /> Promotion Readiness
        </h6>
        {readinessQuery.isLoading ? (
          <div className="spinner-border spinner-border-sm text-muted" />
        ) : readinessQuery.data ? (
          <div className="space-y-2">
            <div className={`alert ${readinessAlertClass} py-2 small border-0`}>
              <div className="d-flex align-items-center fw-bold text-uppercase">
                {overallReadinessStatus === 'pass' ? <CheckCircle size={14} className="me-2" /> : <AlertTriangle size={14} className="me-2" />}
                {readinessLabel}
              </div>
            </div>
            <ul className="list-group list-group-flush border rounded overflow-hidden">
              {readinessQuery.data.checks.map((check: ReadinessCheck, i: number) => (
                <li key={i} className="list-group-item d-flex justify-content-between align-items-center py-2 px-3 small">
                  <span className="fw-medium">{check.name}</span>
                  <div className="d-flex align-items-center">
                    <span className="text-muted me-2" style={{ fontSize: '0.9em' }}>{check.message}</span>
                    <StatusDot status={toStatusDotValue(toCheckReadinessStatus(check))} />
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : <div className="text-muted small italic">Readiness data unavailable</div>}
      </div>

      <div className="col-md-6 border-start">
        <h6 className="fw-bold mb-3 d-flex align-items-center small text-uppercase tracking-wider text-muted">
          <Info size={14} className="me-2" /> Approval Signals
        </h6>
        {signalsQuery.isLoading ? (
          <div className="spinner-border spinner-border-sm text-muted" />
        ) : signalsQuery.data ? (
          <div className="space-y-2">
            {signalsQuery.data.signals.map((s: ApprovalSignalItem, i: number) => (
              <div key={i} className={`d-flex p-2 rounded small border-start border-3 ${s.severity === 'risk' ? 'bg-danger bg-opacity-10 border-danger' : s.severity === 'warning' ? 'bg-warning bg-opacity-10 border-warning' : 'bg-light border-secondary'}`}>
                <div className="me-2 mt-1">
                  {s.severity === 'risk' ? <AlertTriangle size={14} className="text-danger" /> : s.severity === 'warning' ? <AlertTriangle size={14} className="text-warning" /> : <Info size={14} className="text-info" />}
                </div>
                <div>
                  <div className="fw-bold">{s.label}</div>
                  <div className="text-muted" style={{ fontSize: '0.85em' }}>{s.description}</div>
                </div>
              </div>
            ))}
          </div>
        ) : <div className="text-muted small italic">No signals collected yet.</div>}
      </div>

      <div className="col-12 mt-3 pt-3 border-top d-flex justify-content-end align-items-center gap-3">
        {rule.status === 'approved' && (
          <>
            {!isReady && (
              <div className="text-danger small d-flex align-items-center fw-bold">
                <AlertTriangle size={14} className="me-1" />
                Publication Blocked
              </div>
            )}
            <button
              className="btn btn-primary btn-sm d-flex align-items-center px-3"
              onClick={onShowPublish}
              disabled={!isReady}
            >
              <Send size={14} className="me-2" /> Publish to Production
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function RuleHistoryTab({ ruleId }: { ruleId: string }) {
  const { tenantId } = useTenant();
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);

  const versionsQuery = useQuery({
    queryKey: ['rules', tenantId, ruleId, 'versions'],
    queryFn: () => rulesApi.getVersions(ruleId)
  });

  const diffQuery = useQuery({
    queryKey: ['rules', tenantId, ruleId, 'diff', selectedVersions],
    queryFn: () => rulesApi.getDiff(ruleId, selectedVersions[0], selectedVersions[1]),
    enabled: selectedVersions.length === 2
  });

  const handleToggleVersion = (id: string) => {
    if (selectedVersions.includes(id)) {
      setSelectedVersions(selectedVersions.filter(v => v !== id));
    } else if (selectedVersions.length < 2) {
      setSelectedVersions([...selectedVersions, id]);
    } else {
      setSelectedVersions([selectedVersions[1], id]);
    }
  };

  if (versionsQuery.isLoading) return <div className="spinner-border spinner-border-sm text-primary" />;

  const versions = versionsQuery.data?.versions || [];

  return (
    <div className="row g-4">
      <div className="col-md-5">
        <h6 className="fw-bold mb-3 small text-uppercase tracking-wider text-muted">Version History</h6>
        <div className="list-group list-group-flush border rounded overflow-hidden shadow-sm">
          {versions.map((v: Rule) => (
            <button
              key={v.id}
              className={`list-group-item list-group-item-action py-3 px-3 border-bottom d-flex justify-content-between align-items-start ${selectedVersions.includes(v.id) ? 'bg-primary bg-opacity-10 border-start border-4 border-primary' : ''}`}
              onClick={() => handleToggleVersion(v.id)}
            >
              <div className="flex-grow-1">
                <div className="d-flex justify-content-between mb-1">
                  <span className="font-monospace small fw-bold">v{v.id.substring(0, 8)}</span>
                  <span className="badge bg-light text-dark border small">{v.status}</span>
                </div>
                {/* Timestamp missing in Rule proto */}
                <div className="small text-muted mb-1">-</div>
                <div className="small fw-medium text-truncate" style={{ maxWidth: '200px' }}>{v.reason || 'No reason provided'}</div>
              </div>
              <div className="ms-2">
                {selectedVersions.includes(v.id) && <CheckCircle size={16} className="text-primary" />}
              </div>
            </button>
          ))}
        </div>
        <div className="mt-3 text-muted small italic">Select two versions to compare.</div>
      </div>

      <div className="col-md-7 border-start">
        <h6 className="fw-bold mb-3 small text-uppercase tracking-wider text-muted d-flex justify-content-between">
          <span><Diff size={14} className="me-2" /> Comparison</span>
          {selectedVersions.length === 2 && <span className="badge bg-light text-primary border">Diffing...</span>}
        </h6>

        {selectedVersions.length < 2 ? (
          <div className="text-center py-5 bg-light rounded text-muted">
            <Diff size={48} className="mb-3 opacity-25" />
            <p>Select two versions from the list to see changes.</p>
          </div>
        ) : diffQuery.isLoading ? (
          <div className="text-center py-5"><div className="spinner-border text-primary" /></div>
        ) : diffQuery.data ? (
          <div className="space-y-3">
            {hasBreakingChanges(diffQuery.data) && (
              <div className="alert alert-warning py-2 small d-flex align-items-center border-0">
                <AlertTriangle size={14} className="me-2" />
                Breaking changes detected (behavioral shift).
              </div>
            )}
            <table className="table table-sm small">
              <thead className="table-light">
                <tr>
                  <th>Field</th>
                  <th>Old Value</th>
                  <th></th>
                  <th>New Value</th>
                </tr>
              </thead>
              <tbody>
                {diffQuery.data.changes.map((c, i: number) => (
                  <tr key={i} className={c.change_type === 'modified' ? 'table-warning bg-opacity-10' : ''}>
                    <td className="fw-bold text-muted">{c.field_name}</td>
                    <td className="text-decoration-line-through text-muted">{JSON.stringify(c.before_value)}</td>
                    <td className="text-center"><ArrowRight size={12} className="text-muted" /></td>
                    <td className="fw-bold">{JSON.stringify(c.after_value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : diffQuery.isError ? (
          <ErrorBanner error={diffQuery.error} title="Failed to compute diff" className="alert alert-danger small" />
        ) : null}
      </div>
    </div>
  );
}

function RuleImpactTab({ ruleId }: { ruleId: string }) {
  const { tenantId } = useTenant();
  const attributionQuery = useQuery({
    queryKey: ['analytics', tenantId, 'attribution', ruleId],
    queryFn: () => analyticsApi.getAttribution(ruleId)
  });

  const impactQuery = useQuery({
    queryKey: ['analytics', tenantId, 'impact', ruleId],
    queryFn: ({ signal }) => analyticsApi.getRuleImpact(ruleId, { signal })
  });

  if (attributionQuery.isError || impactQuery.isError) {
    const error = (attributionQuery.error || impactQuery.error) as { status?: number; code?: string; message?: string } | null;
    const isTimeout =
      error?.status === 504 ||
      error?.message?.toLowerCase().includes('timeout') ||
      error?.code === 'GATEWAY_TIMEOUT';

    return (
      <div className="p-4 border rounded bg-light m-4 mt-0">
        {isTimeout ? (
          <div className="text-center py-5 text-muted">
            <Clock size={48} className="mb-3 opacity-25" />
            <h6 className="fw-bold">Request Timed Out</h6>
            <p className="small mb-0">The analytics query took too long to complete. This can happen with very large datasets or complex rules.</p>
          </div>
        ) : (
          <ErrorBanner
            error={error}
            title="Failed to load rule impact attribution"
          />
        )}
      </div>
    );
  }

  if (attributionQuery.isLoading || impactQuery.isLoading) {
    return (
      <div className="text-center py-5 text-muted border rounded bg-light m-4 mt-0" style={{ minHeight: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div className="spinner-border text-primary mb-3" />
        <p className="small mb-0">Loading impact data...</p>
      </div>
    );
  }

  if (!attributionQuery.data) return <div className="alert alert-info">No attribution data available for this rule.</div>;

  const items = attributionQuery.data.items ?? [];
  const buckets = impactQuery.data?.daily_buckets ?? [];
  const totalMatches = items.reduce((sum, item) => sum + Number(item.volume), 0);

  // If there are literally 0 triggers and 0 items and no buckets/triggers
  if (totalMatches === 0 && (Number(impactQuery.data?.total_triggers) === 0 || !impactQuery.data?.total_triggers)) {
    return (
      <div className="text-center py-5 bg-light rounded border text-muted m-4 mt-0" style={{ minHeight: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <BarChart2 size={48} className="mb-3 opacity-25" />
        <h6 className="fw-bold">No Impact Data</h6>
        <p className="small mb-0">This rule has not triggered on any transactions yet.</p>
      </div>
    );
  }
  const netImpact = items.reduce((sum, item) => sum + Number(item.contribution_score), 0);
  const averageImpactPerMatch = totalMatches > 0 ? netImpact / totalMatches : 0;
  const averageDailyImpact = items.length > 0 ? netImpact / items.length : 0;

  // Derived from attribution items.
  const waterfallData = [
    { name: 'Net Impact', value: netImpact, fill: netImpact > 0 ? '#ff7300' : '#82ca9d' },
    { name: 'Avg / Match', value: averageImpactPerMatch, fill: '#8884d8' },
    { name: 'Avg / Day', value: averageDailyImpact, fill: '#413ea0' }
  ];

  // Derived from impact buckets.
  const totalDecisionsChanged = buckets.reduce((sum: number, b) => {
    return sum + (Number(b.decisions_changed_count) || 0);
  }, 0);

  return (
    <div className="row g-4">
      <div className="col-md-6">
        <div className="d-flex align-items-center gap-2 mb-4">
          <h6 className="fw-bold mb-0 small text-uppercase tracking-wider text-muted">Rule Score Attribution (7d Avg)</h6>
          <DataQualityBadge meta={impactQuery.data?.meta} />
        </div>
        <div style={{ height: 300, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={waterfallData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip formatter={(value: number | undefined) => value ? [value.toFixed(1), 'Impact'] : ['0.0', 'Impact']} />
              <Bar dataKey="value" barSize={60}>
                {waterfallData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
              <Line type="monotone" dataKey="value" stroke="#ccc" strokeDasharray="5 5" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="col-md-6">
        <div className="d-flex align-items-center gap-2 mb-4">
          <h6 className="fw-bold mb-0 small text-uppercase tracking-wider text-muted">Daily Impact Trends</h6>
          <DataQualityBadge meta={impactQuery.data?.meta} />
        </div>
        <div style={{ height: 300, width: '100%' }}>
          {buckets.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={buckets} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Area yAxisId="left" type="monotone" dataKey="trigger_count" stroke="#8884d8" fill="#8884d8" name="Triggers" />
                <Area yAxisId="right" type="monotone" dataKey="avg_score_delta" stroke="#82ca9d" fill="#82ca9d" name="Avg Delta" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-100 d-flex align-items-center justify-content-center bg-light border rounded text-muted small">
              No trend data available for this range
            </div>
          )}
        </div>
      </div>

      <div className="col-md-12">
        <div className="mt-2 row text-center g-3">
          <div className="col-md-3">
            <div className="p-3 border rounded bg-light">
              <div className="small text-muted text-uppercase mb-1">Total Triggers</div>
              <div className="h4 mb-0 fw-bold">{impactQuery.data?.total_triggers.toLocaleString() ?? totalMatches.toLocaleString()}</div>
            </div>
          </div>
          <div className="col-md-3">
            <div className="p-3 border rounded bg-light">
              <div className="small text-muted text-uppercase mb-1">Avg Score Delta</div>
              <div className="h4 mb-0 fw-bold">{impactQuery.data?.avg_score_delta.toFixed(1) ?? averageImpactPerMatch.toFixed(1)}</div>
            </div>
          </div>
          <div className="col-md-3">
            <div className="p-3 border rounded bg-light">
              <div className="small text-muted text-uppercase mb-1">Net Impact</div>
              <div className={`h4 mb-0 fw-bold ${netImpact > 0 ? 'text-danger' : 'text-success'}`}>
                {netImpact > 0 ? '+' : ''}{netImpact.toFixed(1)}
              </div>
            </div>
          </div>
          <div className="col-md-3">
            <div className="p-3 border rounded bg-light border-primary bg-primary bg-opacity-10">
              <div className="small text-primary text-uppercase mb-1">Decision Shifts</div>
              <div className="h4 mb-0 fw-bold text-primary">
                {totalDecisionsChanged.toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ... (keep RuleSandbox, RuleShadow, RuleBacktests, RuleSuggestions, PublishModal, StatusDot)

function PublishModal({ rule, onClose, onSuccess }: { rule: DraftRule, onClose: () => void, onSuccess: () => void }) {
  const [actor, setActor] = useState('');
  const [reason, setReason] = useState('');

  const publishMutation = useMutation({
    mutationFn: ({ actor, reason }: { actor: string, reason: string }) =>
      rulesApi.publishRule(rule.id, { actor, reason }),
    onSuccess: () => {
      onSuccess();
      onClose();
    }
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (actor && reason) {
      publishMutation.mutate({ actor, reason });
    }
  };

  return (
    <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}>
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content shadow-lg border-0">
          <div className="modal-header border-0 pb-0">
            <h5 className="modal-title fw-bold">Publish Rule: {rule.id}</h5>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="modal-body p-4">
              <p className="text-muted small mb-4">
                This rule will become effective immediately for live traffic. An audit trail is required.
              </p>

              <div className="mb-3">
                <label className="form-label small fw-bold d-flex align-items-center">
                  <User size={14} className="me-1" /> Authorized Actor
                </label>
                <input
                  type="text"
                  className="form-control"
                  placeholder="name@example.com"
                  value={actor}
                  onChange={e => setActor(e.target.value)}
                  required
                />
              </div>

              <div className="mb-3">
                <label className="form-label small fw-bold d-flex align-items-center">
                  <FileText size={14} className="me-1" /> Business Reason
                </label>
                <textarea
                  className="form-control"
                  rows={3}
                  placeholder="Explain why this rule is being published..."
                  value={reason}
                  onChange={e => setReason(e.target.value)}
                  required
                />
              </div>

              {publishMutation.isError && (
                <ErrorBanner error={publishMutation.error} title="Publishing failed" className="alert alert-danger small py-2" />
              )}
            </div>
            <div className="modal-footer border-0 pt-0">
              <button type="button" className="btn btn-light" onClick={onClose}>Cancel</button>
              <button
                type="submit"
                className="btn btn-primary px-4"
                disabled={publishMutation.isPending || !actor || !reason}
              >
                {publishMutation.isPending ? 'Publishing...' : 'Confirm Publish'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: 'pass' | 'warn' | 'fail' | 'skip' }) {
  const color = status === 'pass' ? 'bg-success' : status === 'warn' ? 'bg-warning' : status === 'fail' ? 'bg-danger' : 'bg-secondary';
  return <span className={`d-inline-block rounded-circle ${color}`} style={{ width: '8px', height: '8px' }} title={status} />;
}

function toStatusDotValue(status: ReadinessDisplayStatus): 'pass' | 'warn' | 'fail' | 'skip' {
  if (status === 'unknown') {
    return 'skip';
  }
  return status;
}

export function RuleSandbox() {
  const [formData, setFormData] = useState<SandboxEvaluateRequest>({
    base_score: 50,
    features: {},
  });
  const [inputMode, setInputMode] = useState<'sliders' | 'json'>('sliders');
  const [featuresJson, setFeaturesJson] = useState('{}');

  // Default common features for sliders
  const [sliderFeatures, setSliderFeatures] = useState({
    amount: 100,
    velocity_24h: 1,
    amount_to_avg_ratio_30d: 1.0,
    merchant_risk_score: 20,
    bank_connections_24h: 1,
    balance_volatility_z_score: 0.0,
    has_history: true
  });

  const [result, setResult] = useState<SandboxEvaluateResponse | null>(null);
  const [jsonError, setJsonError] = useState<string | null>(null);

  // Sync sliders to JSON
  /*
  useEffect(() => {
    if (inputMode === 'sliders') {
      setFeaturesJson(JSON.stringify(sliderFeatures, null, 2));
    }
  }, [sliderFeatures, inputMode]);
  */
  // Derived state for featuresJson instead of effect
  const activeFeaturesJson = inputMode === 'sliders' ? JSON.stringify(sliderFeatures, null, 2) : featuresJson;

  const evaluateMutation = useMutation({
    mutationFn: rulesApi.sandboxEvaluate,
    onSuccess: (data) => setResult(data),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setJsonError(null);
    setJsonError(null);
    try {
      const features = JSON.parse(activeFeaturesJson);
      evaluateMutation.mutate({ ...formData, features });
    } catch {
      setJsonError('Invalid JSON format');
    }
  };

  const updateSlider = (key: string, val: number | boolean) => {
    setSliderFeatures(prev => ({ ...prev, [key]: val }));
  };

  return (
    <div>
      <div className="section-header mb-4">
        <h3>Rule Sandbox</h3>
        <p className="text-muted">Experiment with rule logic by simulating feature inputs. No production impact.</p>
      </div>

      <div className="row g-4">
        <div className="col-lg-5">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header bg-white py-3 d-flex justify-content-between align-items-center">
              <h5 className="mb-0 small fw-bold text-uppercase tracking-wider">Test Input</h5>
              <div className="btn-group btn-group-sm">
                <button className={`btn ${inputMode === 'sliders' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setInputMode('sliders')}>Sliders</button>
                <button className={`btn ${inputMode === 'json' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setInputMode('json')}>JSON</button>
              </div>
            </div>
            <div className="card-body p-4">
              <form onSubmit={handleSubmit}>
                <div className="mb-4">
                  <label className="form-label small fw-bold">Base Score (Model Baseline)</label>
                  <input type="range" className="form-range" min="1" max="99" value={formData.base_score} onChange={e => setFormData({ ...formData, base_score: parseInt(e.target.value) })} />
                  <div className="text-center fw-bold h4 text-primary">{formData.base_score}</div>
                </div>

                {inputMode === 'sliders' ? (
                  <div className="space-y-3">
                    <SliderInput label="Transaction Amount" value={sliderFeatures.amount} min={0} max={10000} step={10} onChange={v => updateSlider('amount', v)} />
                    <SliderInput label="24h Velocity" value={sliderFeatures.velocity_24h} min={0} max={50} step={1} onChange={v => updateSlider('velocity_24h', v)} />
                    <SliderInput label="Amount/Avg Ratio" value={sliderFeatures.amount_to_avg_ratio_30d} min={0} max={10} step={0.1} onChange={v => updateSlider('amount_to_avg_ratio_30d', v)} />
                    <SliderInput label="Merchant Risk" value={sliderFeatures.merchant_risk_score} min={0} max={100} step={1} onChange={v => updateSlider('merchant_risk_score', v)} />
                    <div className="form-check form-switch mt-3">
                      <input className="form-check-input" type="checkbox" checked={sliderFeatures.has_history} onChange={e => updateSlider('has_history', e.target.checked)} id="hasHistory" />
                      <label className="form-check-label small fw-bold" htmlFor="hasHistory">User Has History</label>
                    </div>
                  </div>
                ) : (
                  <div className="mb-3">
                    <textarea
                      className={`form-control font-monospace small ${jsonError ? 'is-invalid' : ''}`}
                      rows={12} value={activeFeaturesJson} onChange={e => {
                        if (inputMode === 'json') setFeaturesJson(e.target.value);
                      }}
                    />
                    {jsonError && <div className="invalid-feedback">{jsonError}</div>}
                  </div>
                )}

                <button type="submit" className="btn btn-primary w-100 mt-4 py-2 fw-bold" disabled={evaluateMutation.isPending}>
                  {evaluateMutation.isPending ? 'Evaluating...' : 'Evaluate Rules'}
                </button>
              </form>
            </div>
          </div>
        </div>

        <div className="col-lg-7">
          {result ? (
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header bg-white py-3">
                <h5 className="mb-0 small fw-bold text-uppercase tracking-wider">Evaluation Results</h5>
              </div>
              <div className="card-body p-4">
                <div className="d-flex justify-content-around align-items-center mb-5 p-4 bg-light rounded-3">
                  <div className="text-center">
                    <div className="text-muted small text-uppercase mb-1">Baseline</div>
                    <div className="h3 mb-0 fw-bold">{formData.base_score}</div>
                  </div>
                  <ArrowRight className="text-muted" />
                  <div className="text-center">
                    <div className="text-muted small text-uppercase mb-1">Final Score</div>
                    <div className={`h1 mb-0 fw-bold ${result.final_score >= 80 ? 'text-danger' : result.final_score >= 30 ? 'text-warning' : 'text-success'}`}>
                      {result.final_score}
                    </div>
                    <span className="badge rounded-pill bg-light text-dark border mt-1">{result.risk_label}</span>
                  </div>
                </div>

                <h6 className="fw-bold mb-3 small text-uppercase text-muted">Matched Rules ({result.matched_rules.length})</h6>
                {result.matched_rules.length > 0 ? (
                  <div className="list-group list-group-flush border rounded overflow-hidden">
                    {result.matched_rules.map((r, i) => (
                      <div key={i} className="list-group-item d-flex justify-content-between align-items-center p-3">
                        <div>
                          <div className="fw-bold">{r.name}</div>
                          <div className="small text-muted">{r.reason}</div>
                        </div>
                        <div className="text-end">
                          <span className="badge bg-primary mb-1 d-block">{r.action}</span>
                          {r.score_adjustment && <div className="small fw-bold text-danger">+{r.score_adjustment}</div>}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : <div className="alert alert-info py-2 small">No active rules matched these inputs.</div>}

                {result.shadow_matched_rules.length > 0 && (
                  <div className="mt-4">
                    <h6 className="fw-bold mb-3 small text-uppercase text-muted">Shadow Rules ({result.shadow_matched_rules.length})</h6>
                    <div className="list-group list-group-flush border rounded overflow-hidden opacity-75">
                      {result.shadow_matched_rules.map((r, i) => (
                        <div key={i} className="list-group-item d-flex justify-content-between align-items-center p-2 bg-light">
                          <span className="small fw-medium">{r.name}</span>
                          <span className="badge bg-secondary small">shadow</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card shadow-sm border-0 border-dashed h-100 d-flex align-items-center justify-content-center text-muted py-5">
              <div className="text-center">
                <Send size={48} className="mb-3 opacity-25" />
                <p>Submit inputs to see rule evaluation results.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SliderInput({ label, value, min, max, step, onChange }: { label: string, value: number, min: number, max: number, step: number, onChange: (v: number) => void }) {
  return (
    <div className="mb-3">
      <div className="d-flex justify-content-between mb-1">
        <label className="form-label small fw-bold mb-0">{label}</label>
        <span className="small font-monospace fw-bold text-primary">{value}</span>
      </div>
      <input type="range" className="form-range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  );
}

export function RuleShadow() {
  const { tenantId } = useTenant();
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
    queryKey: ['shadow-comparison', tenantId, dateRange.start, dateRange.end],
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
        <ErrorBanner error={shadowQuery.error} title="Failed to load shadow metrics" className="alert alert-danger" />
      ) : shadowQuery.data && shadowQuery.data.metrics ? (
        <div className="card">
          <div className="card-header">
            <h4 className="card-title">Rule Comparison</h4>
            <span className="text-muted">
              {(shadowQuery.data.metrics?.total_evaluations || 0).toLocaleString()} total evaluations
            </span>
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
  const { tenantId } = useTenant();
  const [searchParams, setSearchParams] = useSearchParams();
  const ruleFilter = searchParams.get('rule_id') || '';
  const [page, setPage] = useState(1);
  const pageSize = 10;

  const backtestsQuery = useQuery({
    queryKey: ['backtest-results', tenantId, ruleFilter],
    queryFn: () => backtestApi.listResults({ rule_id: ruleFilter || undefined, limit: 100 }),
  });

  // Effect to reset page removed to avoid "state in effect" warning.
  // Ideally page should be in URL params.
  // For now, we will reset it ONLY when we change the filter handler.
  /*
  useEffect(() => {
    setPage(1);
  }, [ruleFilter]);
  */

  const handleFilterChange = (val: string) => {
    const newParams = new URLSearchParams(searchParams);
    if (val) newParams.set('rule_id', val);
    else newParams.delete('rule_id');
    setSearchParams(newParams);
    setPage(1); // Reset page manually when filter changes
  };

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

  const results = backtestsQuery.data?.results || [];
  const totalPages = Math.ceil(results.length / pageSize);
  const paginatedResults = results.slice((page - 1) * pageSize, page * pageSize);

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
              onChange={(e) => handleFilterChange(e.target.value)}
              style={{ width: '200px' }}
            />
          </div>
        </div>
      </div>

      {/* Results */}
      {backtestsQuery.isLoading ? (
        <div className="loading">Loading backtest results...</div>
      ) : backtestsQuery.isError ? (
        <ErrorBanner error={backtestsQuery.error} title="Failed to load backtests" className="alert alert-danger" />
      ) : results.length > 0 ? (
        <div className="card">
          <div className="card-header">
            <h4 className="card-title">Results</h4>
            <span className="text-muted">{results.length} recent results</span>
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
                {paginatedResults.map((result: BacktestResult) => (
                  <tr key={result.job_id}>
                    <td><code>{result.job_id.slice(0, 8)}</code></td>
                    <td><code>{result.rule_id}</code></td>
                    <td>
                      <span className={`status-badge ${getStatusBadgeClass('completed')}`}>
                        Completed
                      </span>
                    </td>
                    <td>{new Date(result.completed_at || 0).toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      {(result.metrics?.match_rate || 0).toFixed(3)}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      -
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      -
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="pagination-controls" style={{ padding: '1rem', display: 'flex', justifyContent: 'center', gap: '1rem', alignItems: 'center' }}>
              <button
                className="btn btn-secondary btn-sm"
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
              >
                &lt; Prev
              </button>
              <span className="text-muted small">Page {page} of {totalPages}</span>
              <button
                className="btn btn-secondary btn-sm"
                disabled={page === totalPages}
                onClick={() => setPage(p => p + 1)}
              >
                Next &gt;
              </button>
            </div>
          )}
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
  const { tenantId } = useTenant();
  const [minConfidence, setMinConfidence] = useState(0.7);
  const queryClient = useQueryClient();

  const suggestionsQuery = useQuery({
    queryKey: ['suggestions', tenantId, minConfidence],
    queryFn: () => suggestionsApi.getHeuristic({ min_confidence: minConfidence }),
  });

  const acceptMutation = useMutation({
    mutationFn: suggestionsApi.accept,
    onSuccess: () => {
      alert('Suggestion accepted! A draft rule has been created.');
      // Refresh draft rules if we were showing them
      queryClient.invalidateQueries({ queryKey: ['rules', tenantId, 'draft'] });
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
            style={{ maxWidth: '300px', display: 'inline-block', verticalAlign: 'middle' }}
          />
          <span className="ms-2 fw-bold">{minConfidence.toFixed(2)}</span>
        </div>
      </div>

      {suggestionsQuery.isLoading ? (
        <div className="loading">Analyzing patterns...</div>
      ) : suggestionsQuery.isError ? (
        <ErrorBanner error={suggestionsQuery.error} title="Analysis failed" className="alert alert-danger" />
      ) : suggestionsQuery.data && suggestionsQuery.data.suggestions.length > 0 ? (
        <div className="row">
          {suggestionsQuery.data.suggestions.map((s: RuleSuggestion, idx: number) => (
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
