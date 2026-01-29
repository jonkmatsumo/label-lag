import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { datasetApi } from '../api/dataset';
import type { FeatureSample } from '../api/dataset';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis
} from 'recharts';
import { AlertCircle, CheckCircle, RefreshCw, Trash2, Database, Activity, Table as TableIcon } from 'lucide-react';

export function Dataset() {
  const [activeTab, setActiveTab] = useState<'overview' | 'generate' | 'diagnostics'>('overview');
  
  return (
    <div className="container-fluid py-4">
      <header className="mb-4">
        <h1 className="display-6 fw-bold text-primary">Synthetic Dataset</h1>
        <p className="text-muted">Generate and manage synthetic training data for model development.</p>
      </header>

      <div className="card shadow-sm">
        <div className="card-header">
          <ul className="nav nav-tabs card-header-tabs">
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                <Database size={16} className="me-2 d-inline-block" />
                Overview
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'generate' ? 'active' : ''}`}
                onClick={() => setActiveTab('generate')}
              >
                <RefreshCw size={16} className="me-2 d-inline-block" />
                Generate
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'diagnostics' ? 'active' : ''}`}
                onClick={() => setActiveTab('diagnostics')}
              >
                <Activity size={16} className="me-2 d-inline-block" />
                Diagnostics
              </button>
            </li>
          </ul>
        </div>
        <div className="card-body p-4">
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'generate' && <GenerateTab />}
          {activeTab === 'diagnostics' && <DiagnosticsTab />}
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
  const [selectedFeature, setSelectedFeature] = useState<string>('velocity_24h');
  const [stratify, setStratify] = useState(true);

  const { data, isLoading, error } = useQuery({
    queryKey: ['dataset', 'sample', stratify],
    queryFn: () => datasetApi.getFeatureSample(2000, stratify),
  });

  const samples = data?.samples || [];
  
  // Extract numeric keys for dropdown
  const numericKeys = useMemo(() => {
    if (!samples.length) return [];
    const first = samples[0];
    return Object.keys(first).filter(k => 
      typeof first[k] === 'number' && 
      !['record_id', 'user_id', 'snapshot_id', 'is_fraudulent'].includes(k)
    );
  }, [samples]);

  // Compute histogram data client-side
  const histogramData = useMemo(() => {
    if (!samples.length || !selectedFeature) return [];
    
    const values = samples.map(s => Number(s[selectedFeature])).filter(v => !isNaN(v));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const bins = 20;
    const binSize = (max - min) / bins;

    const binData = Array.from({ length: bins }, (_, i) => ({
      binStart: min + i * binSize,
      binEnd: min + (i + 1) * binSize,
      fraud: 0,
      legit: 0,
      label: `${(min + i * binSize).toFixed(1)}`
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

  // Compute basic correlation with fraud (simple point-biserial approximation)
  const correlations = useMemo(() => {
    if (!samples.length || !numericKeys.length) return [];
    
    const fraudVals = samples.map(s => s.is_fraudulent ? 1 : 0);
    const fraudMean = fraudVals.reduce((a,b) => a+b, 0) / fraudVals.length;

    return numericKeys.map(key => {
      const vals = samples.map(s => Number(s[key]) || 0);
      const mean = vals.reduce((a,b) => a+b, 0) / vals.length;
      
      let num = 0, den1 = 0, den2 = 0;
      for(let i=0; i<samples.length; i++) {
        const dx = vals[i] - mean;
        const dy = fraudVals[i] - fraudMean;
        num += dx * dy;
        den1 += dx * dx;
        den2 += dy * dy;
      }
      const corr = num / Math.sqrt(den1 * den2);
      return { key, corr };
    }).sort((a,b) => Math.abs(b.corr) - Math.abs(a.corr));
  }, [samples, numericKeys]);

  if (isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary"></div></div>;
  if (error) return <div className="alert alert-danger">Error: {(error as Error).message}</div>;

  return (
    <div className="row">
      <div className="col-md-3">
        <div className="card mb-3">
          <div className="card-body">
            <h5 className="card-title mb-3">Feature Analysis</h5>
            <div className="mb-3">
              <label className="form-label">Select Feature</label>
              <select 
                className="form-select" 
                value={selectedFeature} 
                onChange={e => setSelectedFeature(e.target.value)}
              >
                {numericKeys.map(k => <option key={k} value={k}>{k}</option>)}
              </select>
            </div>
            
            <h6 className="mt-4 mb-2">Top Correlations (Target)</h6>
            <div className="list-group list-group-flush small">
              {correlations.slice(0, 5).map(c => (
                <div key={c.key} className="list-group-item d-flex justify-content-between px-0">
                  <span className="text-truncate" title={c.key}>{c.key}</span>
                  <span className={`fw-bold ${Math.abs(c.corr) > 0.5 ? 'text-primary' : ''}`}>
                    {c.corr.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      <div className="col-md-9">
        <div className="card">
          <div className="card-header bg-white">
            Distribution: <strong>{selectedFeature}</strong>
          </div>
          <div className="card-body" style={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogramData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
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
        </div>
      </div>
    </div>
  );
}