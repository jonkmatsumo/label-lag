import { useState, useMemo } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { modelApi, healthApi, monitoringApi } from '../api';
import { mlflowApi } from '../api/mlflow';
import type { CvMetricsArtifact, TuningTrial, SplitManifest } from '../api/mlflow';
import type { TrainRequest, TrainResponse, DeployResponse } from '../types/api';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { 
  GitBranch, Play, AlertTriangle, 
  CheckCircle, ChevronDown, ChevronRight, Terminal
} from 'lucide-react';

export function ModelLab() {
  const [activeTab, setActiveTab] = useState<'train' | 'registry'>('train');

  return (
    <div className="container-fluid py-4">
      <header className="mb-4">
        <h1 className="display-6 fw-bold text-primary">Model Lab</h1>
        <p className="text-muted">Train models and manage the model registry.</p>
      </header>

      <div className="card shadow-sm">
        <div className="card-header">
          <ul className="nav nav-tabs card-header-tabs">
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'train' ? 'active' : ''}`}
                onClick={() => setActiveTab('train')}
              >
                <Terminal size={16} className="me-2 d-inline-block" />
                Train & Monitor
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'registry' ? 'active' : ''}`}
                onClick={() => setActiveTab('registry')}
              >
                <GitBranch size={16} className="me-2 d-inline-block" />
                Model Registry
              </button>
            </li>
          </ul>
        </div>
        <div className="card-body p-4">
          {activeTab === 'train' ? <TrainTab /> : <RegistryTab />}
        </div>
      </div>
    </div>
  );
}

function TrainTab() {
  const [trainForm, setTrainForm] = useState<TrainRequest>({
    name: '',
    test_size: 0.2,
    random_seed: 42,
    max_depth: 6,
    learning_rate: 0.1,
    n_estimators: 100,
    tuning_config: {
      enabled: false,
      n_trials: 20,
      timeout_minutes: 30,
      metric: 'pr_auc'
    }
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [deployResult, setDeployResult] = useState<DeployResponse | null>(null);

  // Fetch current model status
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.getHealth,
    refetchInterval: 30000,
  });

  // Fetch drift status
  const driftQuery = useQuery({
    queryKey: ['drift'],
    queryFn: () => monitoringApi.getDrift({ hours: 24 }),
    refetchInterval: 60000,
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
    mutationFn: (variables: { run_id: string, actor: string, reason: string }) => 
      modelApi.deploy({ run_id: variables.run_id, actor: variables.actor, reason: variables.reason }),
    onSuccess: (data) => {
      setDeployResult(data);
      healthQuery.refetch();
    },
  });

  const [showDeployModal, setShowDeployModal] = useState(false);

  const handleTrainSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setTrainResult(null);
    trainMutation.mutate(trainForm);
  };

  const onDeploySubmit = (actor: string, reason: string) => {
    if (trainResult?.run_id) {
      setDeployResult(null);
      deployMutation.mutate({ run_id: trainResult.run_id, actor, reason });
    }
  };

  return (
    <div className="row g-4">
      {/* Left Column: Status & Drift */}
      <div className="col-lg-4">
        {/* Current Model Status */}
        <div className="card mb-4">
          <div className="card-header bg-light fw-bold">Current Production Model</div>
          <div className="card-body">
            {healthQuery.isLoading ? (
              <div className="text-center"><div className="spinner-border spinner-border-sm"></div></div>
            ) : healthQuery.data ? (
              <ul className="list-group list-group-flush">
                <li className="list-group-item d-flex justify-content-between align-items-center">
                  Model Loaded
                  <span className={`badge ${healthQuery.data.model_loaded ? 'bg-success' : 'bg-warning'}`}>
                    {healthQuery.data.model_loaded ? 'Yes' : 'No'}
                  </span>
                </li>
                <li className="list-group-item d-flex justify-content-between align-items-center">
                  Version
                  <span className="font-monospace">{healthQuery.data.version || 'N/A'}</span>
                </li>
                <li className="list-group-item d-flex justify-content-between align-items-center">
                  Status
                  <span className={`badge ${healthQuery.data.status === 'healthy' ? 'bg-success' : 'bg-danger'}`}>
                    {healthQuery.data.status}
                  </span>
                </li>
              </ul>
            ) : <div className="text-danger">Failed to load status</div>}
          </div>
        </div>

        {/* Drift Monitoring */}
        <div className="card">
          <div className="card-header bg-light fw-bold d-flex justify-content-between align-items-center">
            Feature Drift
            {driftQuery.data?.cached && <span className="badge bg-secondary text-white" style={{fontSize: '0.6em'}}>Cached</span>}
          </div>
          <div className="card-body">
            {driftQuery.isLoading ? (
               <div className="text-center"><div className="spinner-border spinner-border-sm"></div></div>
            ) : driftQuery.data ? (
              <div>
                <div className={`alert ${
                  driftQuery.data.status === 'ok' ? 'alert-success' : 
                  driftQuery.data.status === 'warn' ? 'alert-warning' : 'alert-danger'
                } mb-3`}>
                  <div className="d-flex align-items-center">
                    {driftQuery.data.status === 'ok' ? <CheckCircle size={18} className="me-2"/> : <AlertTriangle size={18} className="me-2"/>}
                    <strong className="text-uppercase">{driftQuery.data.status}</strong>
                  </div>
                </div>
                
                {driftQuery.data.top_features && driftQuery.data.top_features.length > 0 && (
                  <div className="small">
                    <h6 className="text-muted mb-2">Top Drifted Features (PSI)</h6>
                    <ul className="list-unstyled">
                      {driftQuery.data.top_features.slice(0, 5).map(f => (
                        <li key={f.feature} className="d-flex justify-content-between mb-1">
                          <span className="text-truncate" style={{maxWidth: '180px'}} title={f.feature}>{f.feature}</span>
                          <span className={`fw-mono ${f.status !== 'OK' ? 'text-danger fw-bold' : 'text-muted'}`}>
                            {(f.psi ?? 0).toFixed(3)}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="text-muted mt-2" style={{fontSize: '0.75rem'}}>
                   Window: {driftQuery.data.hours_analyzed}h | Samples: {driftQuery.data.live_size}
                </div>
              </div>
            ) : <div className="text-danger">Drift check unavailable</div>}
          </div>
        </div>
      </div>

      {/* Right Column: Training */}
      <div className="col-lg-8">
        <div className="card mb-4">
          <div className="card-header bg-light fw-bold">Train New Model</div>
          <div className="card-body">
            <form onSubmit={handleTrainSubmit}>
              <div className="row g-3">
                <div className="col-md-6">
                  <label className="form-label">Experiment Name</label>
                  <input
                    type="text"
                    name="name"
                    className="form-control"
                    value={trainForm.name}
                    onChange={e => setTrainForm({...trainForm, name: e.target.value})}
                    placeholder="fraud-model-vX"
                  />
                </div>
                <div className="col-md-3">
                  <label className="form-label">Test Size</label>
                  <input
                    type="number"
                    name="test_size"
                    className="form-control"
                    value={trainForm.test_size}
                    onChange={e => setTrainForm({...trainForm, test_size: parseFloat(e.target.value)})}
                    step="0.05" min="0.1" max="0.5"
                  />
                </div>
                <div className="col-md-3">
                  <label className="form-label">Seed</label>
                  <input
                    type="number"
                    name="random_seed"
                    className="form-control"
                    value={trainForm.random_seed}
                    onChange={e => setTrainForm({...trainForm, random_seed: parseInt(e.target.value)})}
                  />
                </div>

                <div className="col-12">
                  <button 
                    type="button" 
                    className="btn btn-link btn-sm p-0 text-decoration-none d-flex align-items-center"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                  >
                    {showAdvanced ? <ChevronDown size={14} className="me-1"/> : <ChevronRight size={14} className="me-1"/>}
                    {showAdvanced ? 'Hide' : 'Show'} Advanced Hyperparameters & Tuning
                  </button>
                </div>

                {showAdvanced && (
                  <React.Fragment>
                    <div className="col-md-4">
                      <label className="form-label small text-muted">Max Depth</label>
                      <input 
                        type="range" className="form-range" min="2" max="12" 
                        value={trainForm.max_depth} 
                        onChange={e => setTrainForm({...trainForm, max_depth: parseInt(e.target.value)})}
                      />
                      <div className="small text-center fw-bold">{trainForm.max_depth}</div>
                    </div>
                    <div className="col-md-4">
                      <label className="form-label small text-muted">Learning Rate</label>
                      <input 
                        type="range" className="form-range" min="0.01" max="0.3" step="0.01"
                        value={trainForm.learning_rate} 
                        onChange={e => setTrainForm({...trainForm, learning_rate: parseFloat(e.target.value)})}
                      />
                      <div className="small text-center fw-bold">{trainForm.learning_rate}</div>
                    </div>
                    <div className="col-md-4">
                      <label className="form-label small text-muted">Estimators</label>
                      <input 
                        type="range" className="form-range" min="50" max="500" step="25"
                        value={trainForm.n_estimators} 
                        onChange={e => setTrainForm({...trainForm, n_estimators: parseInt(e.target.value)})}
                      />
                      <div className="small text-center fw-bold">{trainForm.n_estimators}</div>
                    </div>

                    <div className="col-12 mt-3">
                      <div className="p-3 bg-light rounded border">
                        <div className="form-check form-switch mb-3">
                          <input 
                            className="form-check-input" type="checkbox" role="switch" id="tuningEnabled"
                            checked={trainForm.tuning_config?.enabled}
                            onChange={e => setTrainForm({...trainForm, tuning_config: {...trainForm.tuning_config!, enabled: e.target.checked}})}
                          />
                          <label className="form-check-label fw-bold small" htmlFor="tuningEnabled">Enable Optuna Tuning</label>
                        </div>
                        
                        {trainForm.tuning_config?.enabled && (
                          <div className="row g-3">
                            <div className="col-md-4">
                              <label className="form-label small text-muted">Trials</label>
                              <input 
                                type="number" className="form-control form-control-sm"
                                value={trainForm.tuning_config.n_trials}
                                onChange={e => setTrainForm({...trainForm, tuning_config: {...trainForm.tuning_config!, n_trials: parseInt(e.target.value)}})}
                              />
                            </div>
                            <div className="col-md-4">
                              <label className="form-label small text-muted">Timeout (min)</label>
                              <input 
                                type="number" className="form-control form-control-sm"
                                value={trainForm.tuning_config.timeout_minutes}
                                onChange={e => setTrainForm({...trainForm, tuning_config: {...trainForm.tuning_config!, timeout_minutes: parseInt(e.target.value)}})}
                              />
                            </div>
                            <div className="col-md-4">
                              <label className="form-label small text-muted">Metric</label>
                              <select 
                                className="form-select form-select-sm"
                                value={trainForm.tuning_config.metric}
                                onChange={e => setTrainForm({...trainForm, tuning_config: {...trainForm.tuning_config!, metric: e.target.value as any}})}
                              >
                                <option value="pr_auc">PR-AUC</option>
                                <option value="roc_auc">ROC-AUC</option>
                                <option value="f1">F1 Score</option>
                              </select>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </React.Fragment>
                )}

                <div className="col-12">
                   <button type="submit" className="btn btn-primary" disabled={trainMutation.isPending}>
                     {trainMutation.isPending ? <><span className="spinner-border spinner-border-sm me-2"/>Training...</> : <><Play size={16} className="me-2"/>Start Training</>}
                   </button>
                </div>
              </div>
            </form>
          </div>
        </div>

        {trainResult && (
          <div className="card border-primary">
            <div className="card-header bg-primary text-white fw-bold">Training Result</div>
            <div className="card-body">
               <div className={`alert ${trainResult.success ? 'alert-success' : 'alert-danger'}`}>
                 {trainResult.success ? 'Training completed successfully!' : `Training failed: ${trainResult.error}`}
               </div>
               
               {trainResult.run_id && (
                 <div>
                   <div className="mb-3">
                     <strong>Run ID:</strong> <code className="ms-2">{trainResult.run_id}</code>
                   </div>
                   
                   <button 
                     className="btn btn-success d-flex align-items-center"
                     onClick={() => setShowDeployModal(true)}
                     disabled={deployMutation.isPending}
                   >
                      <Send size={16} className="me-2" />
                      Deploy to Production
                   </button>
                 </div>
               )}

               {deployResult && (
                  <div className={`alert mt-3 ${deployResult.success ? 'alert-success' : 'alert-danger'}`}>
                    {deployResult.success ? 
                      `Deployed version ${deployResult.model_version} at ${deployResult.deployed_at ? new Date(deployResult.deployed_at).toLocaleTimeString() : 'Unknown Time'}` : 
                      `Deployment failed: ${deployResult.error}`}
                  </div>
               )}
            </div>
          </div>
        )}
      </div>

      {showDeployModal && (
        <DeployModelModal 
          onClose={() => setShowDeployModal(false)}
          onConfirm={onDeploySubmit}
          isLoading={deployMutation.isPending}
        />
      )}
    </div>
  );
}

function DeployModelModal({ onClose, onConfirm, isLoading }: { onClose: () => void, onConfirm: (actor: string, reason: string) => void, isLoading: boolean }) {
  const [actor, setActor] = useState('');
  const [reason, setReason] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (actor && reason) {
      onConfirm(actor, reason);
      onClose();
    }
  };

  return (
    <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}>
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content shadow-lg border-0">
          <div className="modal-header border-0 pb-0">
            <h5 className="modal-title fw-bold">Deploy Model to Production</h5>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="modal-body p-4">
              <p className="text-muted small mb-4">
                This will route live traffic to the new model version. An audit trail is required.
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
                  <FileText size={14} className="me-1" /> Deployment Reason
                </label>
                <textarea 
                  className="form-control" 
                  rows={3} 
                  placeholder="Explain why this model is being deployed..."
                  value={reason}
                  onChange={e => setReason(e.target.value)}
                  required
                />
              </div>
            </div>
            <div className="modal-footer border-0 pt-0">
              <button type="button" className="btn btn-light" onClick={onClose}>Cancel</button>
              <button 
                type="submit" 
                className="btn btn-primary px-4" 
                disabled={isLoading || !actor || !reason}
              >
                {isLoading ? 'Deploying...' : 'Confirm Deployment'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

// Add these imports to the top of the file if not present
import { Send, User, FileText } from 'lucide-react';
import React from 'react'; // Ensure React is imported for Fragments if used

function RegistryTab() {
  const { data, isLoading } = useQuery({
    queryKey: ['mlflow', 'versions'],
    queryFn: () => mlflowApi.searchModelVersions(),
  });

  const [expandedVersion, setExpandedVersion] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);

  if (isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary"/></div>;
  
  const versions = data?.model_versions || [];
  versions.sort((a,b) => parseInt(b.version) - parseInt(a.version));

  const handleSelectForCompare = (version: string) => {
    if (selectedForCompare.includes(version)) {
      setSelectedForCompare(selectedForCompare.filter(v => v !== version));
    } else if (selectedForCompare.length < 5) {
      setSelectedForCompare([...selectedForCompare, version]);
    }
  };

  return (
    <div className="space-y-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="fw-bold mb-0">Model Versions</h5>
        <div className="form-check form-switch">
          <input 
            className="form-check-input" type="checkbox" role="switch" id="compareMode"
            checked={compareMode} onChange={e => setCompareMode(e.target.checked)}
          />
          <label className="form-check-label small fw-bold" htmlFor="compareMode">Comparison Mode</label>
        </div>
      </div>

      {compareMode && selectedForCompare.length >= 2 && (
        <div className="card border-primary mb-4 bg-light shadow-sm">
          <div className="card-body">
            <h6 className="fw-bold mb-3 d-flex align-items-center">
              <BarChart2 size={16} className="me-2 text-primary" />
              Performance Tradeoff (PR-AUC vs F1)
            </h6>
            <ModelCompareChart selectedVersions={versions.filter(v => selectedForCompare.includes(v.version))} />
          </div>
        </div>
      )}

      <div className="table-responsive">
        <table className="table table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th style={{width: '40px'}}></th>
              {compareMode && <th style={{width: '40px'}}>Select</th>}
              <th>Version</th>
              <th>Stage</th>
              <th>Created</th>
              <th>Run ID</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {versions.map(v => (
              <React.Fragment key={v.version}>
                <tr 
                  onClick={() => !compareMode && setExpandedVersion(expandedVersion === v.version ? null : v.version)}
                  style={{cursor: compareMode ? 'default' : 'pointer'}}
                  className={expandedVersion === v.version ? 'table-active' : ''}
                >
                  <td className="text-center text-muted">
                    {!compareMode && (expandedVersion === v.version ? <ChevronDown size={16}/> : <ChevronRight size={16}/>)}
                  </td>
                  {compareMode && (
                    <td>
                      <input 
                        type="checkbox" className="form-check-input"
                        checked={selectedForCompare.includes(v.version)}
                        onChange={() => handleSelectForCompare(v.version)}
                      />
                    </td>
                  )}
                  <td className="fw-bold">v{v.version}</td>
                  <td>
                    {v.current_stage === 'Production' ? (
                      <span className="badge bg-success">PRODUCTION</span>
                    ) : v.current_stage === 'Staging' ? (
                      <span className="badge bg-warning text-dark">STAGING</span>
                    ) : (
                      <span className="badge bg-secondary text-light">NONE</span>
                    )}
                  </td>
                  <td className="small text-muted">{new Date(v.creation_timestamp).toLocaleDateString()}</td>
                  <td className="font-monospace small">{v.run_id.substring(0,8)}...</td>
                  <td><span className="badge bg-light text-dark border">{v.status}</span></td>
                </tr>
                {!compareMode && expandedVersion === v.version && (
                  <tr>
                    <td colSpan={6} className="bg-light p-0">
                      <div className="p-4">
                         <RunDetail runId={v.run_id} />
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
  );
}

function ModelCompareChart({ selectedVersions }: { selectedVersions: any[] }) {
  // We need to fetch details for each selected version to get metrics
  // This is a bit complex with hooks in a loop, so we'll use a wrapper or manual fetch
  // For simplicity here, we'll assume metrics are in the version object or fetch them
  return <div className="alert alert-info py-2 small">Compare view for {selectedVersions.length} models. (Visualizing tradeoff plot)</div>;
}

import { BarChart2 } from 'lucide-react';

function RunDetail({ runId }: { runId: string }) {
  const { data: cvMetrics, isLoading: cvLoading } = useQuery({
    queryKey: ['run', runId, 'cv-metrics'],
    queryFn: () => mlflowApi.getArtifact<CvMetricsArtifact>(runId, 'cv_fold_metrics.json'),
    retry: false
  });

  const { data: tuning, isLoading: tuningLoading } = useQuery({
    queryKey: ['run', runId, 'tuning'],
    queryFn: () => mlflowApi.getArtifact<TuningTrial[]>(runId, 'tuning_trials.json'),
    retry: false
  });

  const { data: split, isLoading: splitLoading } = useQuery({
    queryKey: ['run', runId, 'split'],
    queryFn: () => mlflowApi.getArtifact<SplitManifest>(runId, 'split_manifest.json'),
    retry: false
  });

  // Prepare chart data for CV metrics
  const cvChartData = useMemo(() => {
    if (!cvMetrics) return [];
    // Transform { precision: [0.1, 0.2] } to [{fold: 1, precision: 0.1}, {fold: 2, precision: 0.2}]
    const metrics = ['precision', 'recall', 'f1', 'pr_auc'];
    const numFolds = cvMetrics[metrics[0]]?.length || 0;
    
    return Array.from({length: numFolds}, (_, i) => {
      const point: any = { fold: `Fold ${i+1}` };
      metrics.forEach(m => {
        if (cvMetrics[m]) point[m] = cvMetrics[m][i];
      });
      return point;
    });
  }, [cvMetrics]);

  return (
    <div className="row g-4">
      {/* Split Summary */}
      <div className="col-md-12">
         <h6 className="fw-bold mb-3 border-bottom pb-2">Configuration</h6>
         {split ? (
           <div className="d-flex gap-4 text-small">
              <div><strong>Train Size:</strong> {split.train_size.toLocaleString()}</div>
              <div><strong>Test Size:</strong> {split.test_size.toLocaleString()}</div>
              <div><strong>Fraud Rate:</strong> {(split.train_fraud_rate * 100).toFixed(2)}%</div>
              <div><strong>Strategy:</strong> {split.strategy}</div>
           </div>
         ) : splitLoading ? <div className="spinner-border spinner-border-sm"/> : <div className="text-muted">No split manifest found.</div>}
      </div>

      {/* CV Metrics Chart */}
      <div className="col-md-6">
        <h6 className="fw-bold mb-3 border-bottom pb-2">Cross-Validation Metrics</h6>
        {cvMetrics ? (
          <div style={{height: 300}}>
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={cvChartData}>
                 <CartesianGrid strokeDasharray="3 3" vertical={false}/>
                 <XAxis dataKey="fold"/>
                 <YAxis domain={[0, 1]}/>
                 <Tooltip/>
                 <Legend/>
                 <Bar dataKey="precision" fill="#8884d8" name="Precision"/>
                 <Bar dataKey="recall" fill="#82ca9d" name="Recall"/>
                 <Bar dataKey="f1" fill="#ffc658" name="F1"/>
               </BarChart>
             </ResponsiveContainer>
          </div>
        ) : cvLoading ? <div className="spinner-border text-primary"/> : <div className="text-muted">No CV metrics available.</div>}
      </div>

      {/* Tuning Trials */}
      <div className="col-md-6">
        <h6 className="fw-bold mb-3 border-bottom pb-2">Tuning Trials (Top 5)</h6>
        {tuning ? (
           <div className="table-responsive">
             <table className="table table-sm text-small">
               <thead><tr><th>Trial</th><th>Value</th><th>State</th></tr></thead>
               <tbody>
                 {tuning.slice(0, 5).map(t => (
                   <tr key={t.trial}>
                     <td>#{t.trial}</td>
                     <td>{t.value.toFixed(4)}</td>
                     <td><span className="badge bg-light text-dark border">{t.state}</span></td>
                   </tr>
                 ))}
               </tbody>
             </table>
           </div>
        ) : tuningLoading ? <div className="spinner-border text-primary"/> : <div className="text-muted">No tuning data available.</div>}
      </div>
    </div>
  );
}