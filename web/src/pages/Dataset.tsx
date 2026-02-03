import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { datasetApi } from '../api/dataset';
import { monitoringApi } from '../api';
import type { RelationshipMetric, CorrelationPair } from '../types/api';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { AlertCircle, CheckCircle, RefreshCw, Trash2, Database, Activity } from 'lucide-react';

export function Dataset() {
  const [activeTab, setActiveTab] = useState<'overview' | 'generate' | 'diagnostics' | 'drift'>('overview');
  
  return (
    <div className="container-fluid py-4">
      <header className="mb-4">
        <h1 className="display-6 fw-bold text-primary">Synthetic Dataset</h1>
        <p className="text-muted">Generate and manage synthetic training data for model development.</p>
      </header>

      <div className="card shadow-sm border-0">
        <div className="card-header bg-white py-3">
          <ul className="nav nav-pills card-header-pills">
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                <Database size={16} className="me-2" />
                Overview
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'generate' ? 'active' : ''}`}
                onClick={() => setActiveTab('generate')}
              >
                <RefreshCw size={16} className="me-2" />
                Generate
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'diagnostics' ? 'active' : ''}`}
                onClick={() => setActiveTab('diagnostics')}
              >
                <Activity size={16} className="me-2" />
                Diagnostics
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'drift' ? 'active' : ''}`}
                onClick={() => setActiveTab('drift')}
              >
                <AlertCircle size={16} className="me-2" />
                Drift Analysis
              </button>
            </li>
          </ul>
        </div>
        <div className="card-body p-4">
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'generate' && <GenerateTab />}
          {activeTab === 'diagnostics' && <DiagnosticsTab />}
          {activeTab === 'drift' && <DriftTab />}
        </div>
      </div>
    </div>
  );
}

function OverviewTab() {
  const { data: overview, isLoading, error } = useQuery({
    queryKey: ['dataset', 'overview'],
    queryFn: datasetApi.getOverview,
  });

  const { data: schemaData } = useQuery({
    queryKey: ['dataset', 'schema'],
    queryFn: datasetApi.getSchema,
  });

  if (isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary" role="status"></div></div>;
  if (error) return <div className="alert alert-danger">Error loading overview: {(error as Error).message}</div>;

  return (
    <div className="space-y-4">
      <h4 className="mb-3">Dataset Metrics</h4>
      <div className="row g-3 mb-4">
        <div className="col-md-4">
          <div className="card h-100 bg-light border-0">
            <div className="card-body text-center">
              <h6 className="text-muted text-uppercase small">Total Records</h6>
              <h2 className="display-6 fw-bold mb-0">{overview?.total_records?.toLocaleString() ?? '—'}</h2>
              <div className="text-danger small mt-2">
                {overview?.fraud_records?.toLocaleString()} fraud records
              </div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card h-100 bg-light border-0">
            <div className="card-body text-center">
              <h6 className="text-muted text-uppercase small">Fraud Rate</h6>
              <h2 className="display-6 fw-bold mb-0">{(overview?.fraud_rate ?? 0).toFixed(2)}%</h2>
              <div className="text-muted small mt-2">
                {overview?.unique_users?.toLocaleString()} unique users
              </div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card h-100 bg-light border-0">
            <div className="card-body text-center">
              <h6 className="text-muted text-uppercase small">Time Range</h6>
              <p className="mb-0 mt-2 fw-medium">
                {overview?.min_transaction_timestamp ? new Date(overview.min_transaction_timestamp).toLocaleDateString() : '—'}
              </p>
              <div className="text-muted small">to</div>
              <p className="mb-0 fw-medium">
                {overview?.max_transaction_timestamp ? new Date(overview.max_transaction_timestamp).toLocaleDateString() : '—'}
              </p>
            </div>
          </div>
        </div>
      </div>

      <hr className="my-4" />

      <h4 className="mb-3">Schema Summary</h4>
      {schemaData?.columns ? (
        <div className="table-responsive">
          <table className="table table-hover table-sm">
            <thead>
              <tr>
                <th>Table</th>
                <th>Column</th>
                <th>Type</th>
                <th>Nullable</th>
              </tr>
            </thead>
            <tbody>
              {schemaData.columns.map((col, idx) => (
                <tr key={`${col.table_name}-${col.column_name}-${idx}`}>
                  <td><span className="badge bg-secondary">{col.table_name}</span></td>
                  <td className="font-monospace">{col.column_name}</td>
                  <td><code>{col.data_type}</code></td>
                  <td>{col.is_nullable === 'YES' ? '✅' : '❌'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-muted fst-italic">No schema information available.</div>
      )}
    </div>
  );
}

function GenerateTab() {
  const [numUsers, setNumUsers] = useState(500);
  const [fraudRate, setFraudRate] = useState(0.05);
  const [dropExisting, setDropExisting] = useState(true);
  
  const queryClient = useQueryClient();

  const generateMutation = useMutation({
    mutationFn: datasetApi.generateData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dataset'] });
    },
  });

  const clearMutation = useMutation({
    mutationFn: datasetApi.clearData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dataset'] });
    },
  });

  return (
    <div className="row">
      <div className="col-lg-6">
        <div className="card mb-4">
          <div className="card-header bg-transparent fw-bold">Generation Parameters</div>
          <div className="card-body">
            <div className="mb-3">
              <label className="form-label">Number of Users</label>
              <input 
                type="range" 
                className="form-range" 
                min="100" max="5000" step="100" 
                value={numUsers} 
                onChange={e => setNumUsers(parseInt(e.target.value))}
              />
              <div className="d-flex justify-content-between text-muted small">
                <span>100</span>
                <span className="fw-bold text-primary">{numUsers}</span>
                <span>5000</span>
              </div>
            </div>

            <div className="mb-3">
              <label className="form-label">Fraud Rate</label>
              <input 
                type="range" 
                className="form-range" 
                min="0.01" max="0.20" step="0.01" 
                value={fraudRate} 
                onChange={e => setFraudRate(parseFloat(e.target.value))}
              />
              <div className="d-flex justify-content-between text-muted small">
                <span>1%</span>
                <span className="fw-bold text-danger">{(fraudRate * 100).toFixed(0)}%</span>
                <span>20%</span>
              </div>
            </div>

            <div className="mb-4 form-check">
              <input 
                type="checkbox" 
                className="form-check-input" 
                id="dropExisting" 
                checked={dropExisting} 
                onChange={e => setDropExisting(e.target.checked)}
              />
              <label className="form-check-label" htmlFor="dropExisting">
                Drop existing data before generating
              </label>
            </div>

            <button 
              className="btn btn-primary w-100" 
              onClick={() => generateMutation.mutate({ num_users: numUsers, fraud_rate: fraudRate, drop_existing: dropExisting })}
              disabled={generateMutation.isPending}
            >
              {generateMutation.isPending ? (
                <><span className="spinner-border spinner-border-sm me-2"/>Generating...</>
              ) : (
                'Generate Data'
              )}
            </button>
          </div>
        </div>

        {generateMutation.isSuccess && (
          <div className="alert alert-success d-flex align-items-center">
            <CheckCircle className="me-2" size={20} />
            <div>
              <strong>Generation Complete!</strong>
              <div className="small">
                Created {generateMutation.data.total_records} records 
                ({generateMutation.data.fraud_records} fraud). 
                Materialized {generateMutation.data.features_materialized} feature snapshots.
              </div>
            </div>
          </div>
        )}

        {generateMutation.isError && (
          <div className="alert alert-danger d-flex align-items-center">
            <AlertCircle className="me-2" size={20} />
            <div>
              <strong>Generation Failed</strong>
              <div className="small">{(generateMutation.error as Error).message}</div>
            </div>
          </div>
        )}
      </div>

      <div className="col-lg-6">
        <div className="card border-danger mb-4">
          <div className="card-header bg-danger text-white fw-bold">Danger Zone</div>
          <div className="card-body">
            <p className="card-text text-muted">
              Permanently delete all generated records, metadata, and feature snapshots. 
              This action cannot be undone.
            </p>
            <button 
              className="btn btn-outline-danger" 
              onClick={() => {
                if(confirm('Are you sure you want to delete all data?')) {
                  clearMutation.mutate();
                }
              }}
              disabled={clearMutation.isPending}
            >
              <Trash2 className="me-2 d-inline-block" size={16} />
              Clear All Data
            </button>
          </div>
        </div>

        {clearMutation.isSuccess && (
          <div className="alert alert-success">
            Cleared tables: {clearMutation.data.tables_cleared.join(', ')}
          </div>
        )}
      </div>
    </div>
  );
}

function DiagnosticsTab() {
  const [diagMode, setDiagMode] = useState<'distributions' | 'missingness' | 'outliers' | 'relationships'>('distributions');
  const [selectedFeature, setSelectedFeature] = useState<string>('velocity_24h');
  const [stratify] = useState(true);

  const { data, isLoading } = useQuery({
    queryKey: ['dataset', 'sample', stratify],
    queryFn: () => datasetApi.getFeatureSample(2000, stratify),
  });

  const samples = data?.samples || [];
  
  const numericKeys = useMemo(() => {
    if (!samples.length) return [];
    const first = samples[0];
    return Object.keys(first).filter(k => 
      typeof first[k] === 'number' && 
      !['record_id', 'user_id', 'snapshot_id', 'is_fraudulent'].includes(k)
    );
  }, [samples]);

  // Histogram calculation
  const histogramData = useMemo(() => {
    if (!samples.length || !selectedFeature) return [];
    const values = samples.map(s => Number(s[selectedFeature])).filter(v => !isNaN(v));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const bins = 20;
    const binSize = (max - min) / bins;
    const binData = Array.from({ length: bins }, (_, i) => ({
      binStart: min + i * binSize,
      label: (min + i * binSize).toFixed(1),
      fraud: 0, legit: 0
    }));
    samples.forEach(s => {
      const val = Number(s[selectedFeature]);
      if (isNaN(val)) return;
      const binIdx = Math.min(Math.floor((val - min) / binSize), bins - 1);
      if (binIdx >= 0) {
        if (s.is_fraudulent) binData[binIdx].fraud++;
        else binData[binIdx].legit++;
      }
    });
    return binData;
  }, [samples, selectedFeature]);

  // Missingness calculation
  const missingnessData = useMemo(() => {
    if (!samples.length) return [];
    const keys = Object.keys(samples[0]);
    return keys.map(k => ({
      column: k,
      pct: (samples.filter(s => s[k] === null || s[k] === undefined).length / samples.length) * 100
    })).sort((a,b) => b.pct - a.pct);
  }, [samples]);

  // Outlier calculation (IQR)
  const outlierStats = useMemo(() => {
    if (!samples.length || !selectedFeature) return null;
    const values = samples.map(s => Number(s[selectedFeature])).filter(v => !isNaN(v)).sort((a,b) => a - b);
    if (values.length < 4) return null;
    
    const q1Idx = Math.floor(values.length * 0.25);
    const q2Idx = Math.floor(values.length * 0.5);
    const q3Idx = Math.floor(values.length * 0.75);
    
    const q1 = values[q1Idx];
    const q2 = values[q2Idx];
    const q3 = values[q3Idx];
    const iqr = q3 - q1;
    
    const lowWhisker = Math.max(values[0], q1 - 1.5 * iqr);
    const highWhisker = Math.min(values[values.length - 1], q3 + 1.5 * iqr);
    
    const outliers = values.filter(v => v < lowWhisker || v > highWhisker);
    const outlierPct = (outliers.length / values.length) * 100;
    
    return { min: values[0], max: values[values.length-1], q1, q2, q3, lowWhisker, highWhisker, outlierCount: outliers.length, outlierPct };
  }, [samples, selectedFeature]);

  if (isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;

  return (
    <div>
      <div className="btn-group btn-group-sm mb-4 shadow-sm">
        <button className={`btn ${diagMode === 'distributions' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setDiagMode('distributions')}>Distributions</button>
        <button className={`btn ${diagMode === 'missingness' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setDiagMode('missingness')}>Missingness</button>
        <button className={`btn ${diagMode === 'outliers' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setDiagMode('outliers')}>Outliers</button>
        <button className={`btn ${diagMode === 'relationships' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setDiagMode('relationships')}>Relationships</button>
      </div>

      <div className="row">
        {diagMode !== 'relationships' && (
          <div className="col-md-3">
            <div className="card border-0 bg-light shadow-sm">
              <div className="card-body">
                <label className="form-label small fw-bold">Target Feature</label>
                <select className="form-select form-select-sm mb-3 shadow-none border" value={selectedFeature} onChange={e => setSelectedFeature(e.target.value)}>
                  {numericKeys.map(k => <option key={k} value={k}>{k}</option>)}
                </select>
                <div className="small text-muted mb-2">Sample size: {samples.length}</div>
              </div>
            </div>
          </div>
        )}
        <div className={diagMode === 'relationships' ? 'col-12' : 'col-md-9'}>
          {diagMode === 'distributions' && (
            <div style={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={histogramData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="label" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="legit" name="Legitimate" stackId="a" fill="#4caf50" />
                  <Bar dataKey="fraud" name="Fraudulent" stackId="a" fill="#f44336" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {diagMode === 'missingness' && (
            <div style={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={missingnessData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis dataKey="column" type="category" width={150} />
                  <Tooltip formatter={(v: number | undefined) => v ? `${v.toFixed(1)}%` : '0%'} />
                  <Bar dataKey="pct" name="Missing %" fill="#ffc658" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {diagMode === 'outliers' && (
             <div className="card border-0 shadow-sm">
              <div className="card-body p-4">
                <h6 className="fw-bold mb-4 d-flex align-items-center">
                   <Activity size={16} className="me-2 text-primary" />
                   Outlier Analysis for {selectedFeature}
                </h6>
                {outlierStats ? (
                  <div className="row align-items-center">
                    <div className="col-lg-8">
                      <div className="mb-4" style={{ height: '140px', position: 'relative', padding: '0 40px' }}>
                         {/* Scale background */}
                         <div style={{ position: 'absolute', bottom: '0', left: '40px', right: '40px', height: '1px', backgroundColor: '#dee2e6' }}></div>
                         
                         {/* Whisker Line */}
                         <div style={{ position: 'absolute', top: '50%', left: `${((outlierStats.lowWhisker - outlierStats.min) / (outlierStats.max - outlierStats.min)) * 80 + 10}%`, right: `${100 - (((outlierStats.highWhisker - outlierStats.min) / (outlierStats.max - outlierStats.min)) * 80 + 10)}%`, height: '2px', backgroundColor: '#6c757d' }}></div>
                         
                         {/* Low Whisker */}
                         <div style={{ position: 'absolute', top: '40%', bottom: '40%', left: `${((outlierStats.lowWhisker - outlierStats.min) / (outlierStats.max - outlierStats.min)) * 80 + 10}%`, width: '2px', backgroundColor: '#6c757d' }}></div>
                         
                         {/* High Whisker */}
                         <div style={{ position: 'absolute', top: '40%', bottom: '40%', left: `${((outlierStats.highWhisker - outlierStats.min) / (outlierStats.max - outlierStats.min)) * 80 + 10}%`, width: '2px', backgroundColor: '#6c757d' }}></div>
                         
                         {/* Box (Q1 to Q3) */}
                         <div style={{ 
                           position: 'absolute', top: '30%', bottom: '30%', 
                           left: `${((outlierStats.q1 - outlierStats.min) / (outlierStats.max - outlierStats.min)) * 80 + 10}%`, 
                           width: `${((outlierStats.q3 - outlierStats.q1) / (outlierStats.max - outlierStats.min)) * 80}%`, 
                           backgroundColor: 'rgba(13, 110, 253, 0.15)',
                           border: '2px solid #0d6efd',
                           borderRadius: '2px'
                         }}></div>
                         
                         {/* Median Line */}
                         <div style={{ position: 'absolute', top: '30%', bottom: '30%', left: `${((outlierStats.q2 - outlierStats.min) / (outlierStats.max - outlierStats.min)) * 80 + 10}%`, width: '3px', backgroundColor: '#0d6efd' }}></div>
                      </div>
                      <div className="d-flex justify-content-between text-muted small px-2">
                        <span>Min: {outlierStats.min.toFixed(2)}</span>
                        <span>Max: {outlierStats.max.toFixed(2)}</span>
                      </div>
                    </div>
                    <div className="col-lg-4">
                      <div className="p-3 bg-light rounded border-start border-4 border-primary">
                        <div className="mb-2 d-flex justify-content-between">
                          <span className="text-muted">Median</span>
                          <span className="fw-bold">{outlierStats.q2.toFixed(4)}</span>
                        </div>
                        <div className="mb-2 d-flex justify-content-between">
                          <span className="text-muted">IQR</span>
                          <span className="fw-bold">{(outlierStats.q3 - outlierStats.q1).toFixed(4)}</span>
                        </div>
                        <div className="mb-2 d-flex justify-content-between">
                          <span className="text-muted">Outliers</span>
                          <span className="text-danger fw-bold">{outlierStats.outlierCount} ({outlierStats.outlierPct.toFixed(1)}%)</span>
                        </div>
                        <div className="small text-muted mt-2 pt-2 border-top">
                          Normal range: [{outlierStats.lowWhisker.toFixed(2)}, {outlierStats.highWhisker.toFixed(2)}]
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-5 text-muted fst-italic">Insufficient numeric samples for analysis.</div>
                )}
              </div>
            </div>
          )}
          {diagMode === 'relationships' && <RelationshipsTab featureColumns={numericKeys} />}
        </div>
      </div>
    </div>
  );
}

function RelationshipsTab({ featureColumns }: { featureColumns: string[] }) {
  const [activeSubTab, setActiveSubTab] = useState<'target' | 'numeric' | 'categorical'>('target');
  const [targetColumn, setTargetColumn] = useState('is_fraudulent');

  const targetQuery = useQuery({
    queryKey: ['dataset', 'relationships', targetColumn],
    queryFn: () => datasetApi.getRelationships(1000, targetColumn),
    enabled: activeSubTab === 'target'
  });

  const matrixQuery = useQuery({
    queryKey: ['dataset', 'correlations'],
    queryFn: () => datasetApi.getCorrelations(1000),
    enabled: activeSubTab !== 'target',
    staleTime: 300000 // 5 minutes cache
  });

  return (
    <div className="card border-0 shadow-sm">
      <div className="card-header bg-white py-3 border-bottom">
        <ul className="nav nav-tabs card-header-tabs">
          <li className="nav-item">
            <button 
              className={`nav-link ${activeSubTab === 'target' ? 'active' : ''}`} 
              onClick={() => setActiveSubTab('target')}
            >
              Feature vs Target
            </button>
          </li>
          <li className="nav-item">
            <button 
              className={`nav-link ${activeSubTab === 'numeric' ? 'active' : ''}`} 
              onClick={() => setActiveSubTab('numeric')}
            >
              Numeric Heatmap (Pearson)
            </button>
          </li>
          <li className="nav-item">
            <button 
              className={`nav-link ${activeSubTab === 'categorical' ? 'active' : ''}`} 
              onClick={() => setActiveSubTab('categorical')}
            >
              Categorical Heatmap (Cramér's V)
            </button>
          </li>
        </ul>
      </div>
      
      <div className="card-body p-4">
        {activeSubTab === 'target' && (
           <TargetRelationshipsTable 
             data={targetQuery.data} 
             isLoading={targetQuery.isLoading} 
             targetColumn={targetColumn} 
             setTargetColumn={setTargetColumn} 
             featureColumns={featureColumns} 
           />
        )}
        
        {activeSubTab === 'numeric' && (
           <CorrelationHeatmap 
             title="Pearson Correlation Matrix" 
             data={matrixQuery.data?.pearson} 
             columns={matrixQuery.data?.numeric_columns} 
             isLoading={matrixQuery.isLoading}
             baseColor="25, 135, 84" // Green-ish
           />
        )}

        {activeSubTab === 'categorical' && (
           <CorrelationHeatmap 
             title="Cramér's V Association Matrix" 
             data={matrixQuery.data?.cramers_v} 
             columns={matrixQuery.data?.categorical_columns} 
             isLoading={matrixQuery.isLoading}
             baseColor="255, 193, 7" // Orange-ish
           />
        )}
      </div>
    </div>
  );
}

function TargetRelationshipsTable({ data, isLoading, targetColumn, setTargetColumn, featureColumns }: { data?: { relationships: RelationshipMetric[] }, isLoading: boolean, targetColumn: string, setTargetColumn: (v: string) => void, featureColumns: string[] }) {
  if (isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h6 className="fw-bold mb-0">Correlations with {targetColumn}</h6>
        <div className="d-flex align-items-center gap-2">
          <span className="small text-muted">Target:</span>
          <select className="form-select form-select-sm shadow-none border" style={{width: '180px'}} value={targetColumn} onChange={e => setTargetColumn(e.target.value)}>
             <option value="is_fraudulent">is_fraudulent</option>
             {featureColumns.map((c: string) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>
      <div className="table-responsive">
        <table className="table table-hover mb-0 small align-middle">
          <thead className="table-light">
            <tr>
              <th>Feature</th>
              <th>Metric Type</th>
              <th>Coefficient</th>
              <th style={{width: '200px'}}>Strength</th>
            </tr>
          </thead>
          <tbody>
            {data?.relationships && data.relationships.length > 0 ? (
              data.relationships.map((rel: RelationshipMetric, idx: number) => (
                <tr key={idx}>
                  <td className="fw-medium">{rel.feature_a}</td>
                  <td>
                    <span className="badge bg-light text-dark border font-monospace text-uppercase" style={{fontSize: '0.7rem'}}>
                      {rel.metric_type}
                    </span>
                  </td>
                  <td className="font-monospace fw-bold">{rel.value.toFixed(4)}</td>
                  <td>
                    <div className="d-flex align-items-center gap-2">
                      <div className="progress flex-grow-1" style={{height: '6px'}}>
                        <div 
                          className={`progress-bar ${Math.abs(rel.value) > 0.5 ? 'bg-danger' : Math.abs(rel.value) > 0.2 ? 'bg-warning' : 'bg-success'}`}
                          style={{width: `${Math.min(Math.abs(rel.value) * 100, 100)}%`}}
                        ></div>
                      </div>
                      <span className="text-muted" style={{fontSize: '0.7rem'}}>
                        {Math.abs(rel.value) > 0.5 ? 'Strong' : Math.abs(rel.value) > 0.2 ? 'Moderate' : 'Weak'}
                      </span>
                    </div>
                  </td>
                </tr>
              ))
            ) : (
              <tr><td colSpan={4} className="text-center py-4 text-muted">No relationships found.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CorrelationHeatmap({ title, data, columns, isLoading, baseColor = "13, 110, 253" }: { title: string, data?: CorrelationPair[], columns?: string[], isLoading: boolean, baseColor?: string }) {
  if (isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;
  if (!data || !columns || columns.length === 0) return <div className="text-muted text-center p-5 bg-light rounded border border-dashed">No numeric data available for correlation analysis.</div>;

  // Limit columns to avoid clutter (e.g. top 20)
  const displayCols = columns.slice(0, 20);
  
  // Create matrix map
  const matrix: Record<string, number> = {};
  data.forEach((d: CorrelationPair) => {
    matrix[`${d.feature_a}:${d.feature_b}`] = d.value;
  });

  return (
    <div className="overflow-auto">
      <h6 className="fw-bold mb-3">{title}</h6>
      <div className="table-responsive">
        <table className="table table-bordered table-sm small text-center" style={{tableLayout: 'fixed', width: 'max-content'}}>
          <thead>
            <tr>
              <th style={{width: '140px', backgroundColor: '#f8f9fa'}} className="text-start ps-3">Feature</th>
              {displayCols.map((c: string) => (
                <th key={c} className="text-truncate" style={{width: '80px', maxWidth: '80px', fontSize: '0.75rem', transform: 'rotate(-45deg)', height: '80px', verticalAlign: 'bottom'}} title={c}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayCols.map((row: string) => (
              <tr key={row}>
                <th className="text-end pe-3 text-truncate" style={{maxWidth: '140px', backgroundColor: '#f8f9fa'}} title={row}>{row}</th>
                {displayCols.map((col: string) => {
                  const val = matrix[`${row}:${col}`] || 0;
                  // Use absolute value for opacity
                  const opacity = Math.max(0.05, Math.abs(val));
                  const bg = `rgba(${baseColor}, ${opacity})`;
                  const textColor = opacity > 0.6 ? '#fff' : '#000';
                  
                  return (
                    <td key={col} style={{backgroundColor: bg, color: textColor, transition: 'all 0.2s'}}>
                      {val.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {columns.length > 20 && <div className="text-muted small italic mt-2">* Showing top 20 columns only.</div>}
    </div>
  );
}

function DriftTab() {
  const driftQuery = useQuery({
    queryKey: ['drift', 'detailed'],
    queryFn: () => monitoringApi.getDrift({ hours: 24, force_refresh: true })
  });

  if (driftQuery.isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;

  return (
    <div className="row g-4">
      <div className="col-md-4">
        <div className={`card h-100 border-0 shadow-sm ${driftQuery.data?.drift_detected ? 'bg-danger bg-opacity-10' : 'bg-success bg-opacity-10'}`}>
          <div className="card-body text-center d-flex flex-column justify-content-center">
            <div className="display-4 mb-2">
              {driftQuery.data?.drift_detected ? '🚨' : '✅'}
            </div>
            <h4 className="fw-bold">{driftQuery.data?.drift_detected ? 'Drift Detected' : 'No Drift'}</h4>
            <p className="text-muted small mb-0">Status: {driftQuery.data?.status}</p>
          </div>
        </div>
      </div>
      <div className="col-md-8">
        <div className="card border-0 shadow-sm">
          <div className="card-header bg-white py-3 fw-bold">Top Feature Shifts (PSI)</div>
          <div className="table-responsive">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr><th>Feature</th><th>PSI</th><th>Status</th></tr>
              </thead>
              <tbody>
                {driftQuery.data?.top_features?.map(f => (
                  <tr key={f.feature}>
                    <td className="fw-medium">{f.feature}</td>
                    <td className="font-monospace">{(f.psi || 0).toFixed(4)}</td>
                    <td>
                      <span className={`badge rounded-pill ${f.status === 'OK' ? 'bg-success' : f.status === 'WARN' ? 'bg-warning text-dark' : 'bg-danger'}`}>
                        {f.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
