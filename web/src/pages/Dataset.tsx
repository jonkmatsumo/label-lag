import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { datasetApi } from '../api/dataset';
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

// ... (keep OverviewTab, GenerateTab)

function DiagnosticsTab() {
  const [diagMode, setDiagMode] = useState<'distributions' | 'missingness' | 'outliers'>('distributions');
  const [selectedFeature, setSelectedFeature] = useState<string>('velocity_24h');
  const [stratify] = useState(true);

  const { data, isLoading, error } = useQuery({
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

  if (isLoading) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;

  return (
    <div>
      <div className="btn-group btn-group-sm mb-4">
        <button className={`btn ${diagMode === 'distributions' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setDiagMode('distributions')}>Distributions</button>
        <button className={`btn ${diagMode === 'missingness' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setDiagMode('missingness')}>Missingness</button>
        <button className={`btn ${diagMode === 'outliers' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setDiagMode('outliers')}>Outliers</button>
      </div>

      <div className="row">
        <div className="col-md-3">
          <div className="card border-0 bg-light">
            <div className="card-body">
              <label className="form-label small fw-bold">Target Feature</label>
              <select className="form-select form-select-sm mb-3" value={selectedFeature} onChange={e => setSelectedFeature(e.target.value)}>
                {numericKeys.map(k => <option key={k} value={k}>{k}</option>)}
              </select>
              <div className="small text-muted mb-2">Sample size: {samples.length}</div>
            </div>
          </div>
        </div>
        <div className="col-md-9">
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
                  <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
                  <Bar dataKey="pct" name="Missing %" fill="#ffc658" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {diagMode === 'outliers' && (
            <div className="text-center py-5 text-muted">
              <Activity size={48} className="mb-3 opacity-25" />
              <p>Outlier boxplot visualization for <strong>{selectedFeature}</strong> (Restoring MVP parity)</p>
            </div>
          )}
        </div>
      </div>
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
    const initialVal: number = 0;
    const fraudMean = fraudVals.reduce((a, b) => a + b, initialVal) / fraudVals.length;

    return numericKeys.map(key => {
      const vals = samples.map(s => Number(s[key]) || 0);
      const mean = vals.reduce((a, b) => a + b, initialVal) / vals.length;
      
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