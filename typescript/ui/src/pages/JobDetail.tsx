import { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { jobsApi } from '../api';
import { useTenant } from '../hooks/useTenant';
import { ErrorBanner } from '../components/ErrorBanner';
import { JobStatusBadge } from './Jobs';
import { ArrowLeft } from 'lucide-react';

const TERMINAL_STATUSES = ['COMPLETED', 'FAILED', 'CANCELLED'];

export function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const { tenantId } = useTenant();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showConfirm, setShowConfirm] = useState<'cancel' | 'retry' | null>(null);

  const jobQuery = useQuery({
    queryKey: ['jobs', tenantId, id],
    queryFn: () => jobsApi.get(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.job?.status;
      if (status && TERMINAL_STATUSES.includes(status)) return false;
      return 5000;
    },
  });

  const eventsQuery = useQuery({
    queryKey: ['jobs', tenantId, id, 'events'],
    queryFn: () => jobsApi.getEvents(id!),
    enabled: !!id,
    refetchInterval: () => {
      const status = jobQuery.data?.job?.status;
      if (status && TERMINAL_STATUSES.includes(status)) return false;
      return 5000;
    },
  });

  const cancelMutation = useMutation({
    mutationFn: () => jobsApi.cancel(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs', tenantId, id] });
      setShowConfirm(null);
    },
  });

  const retryMutation = useMutation({
    mutationFn: () => jobsApi.retry(id!),
    onSuccess: (data) => {
      setShowConfirm(null);
      navigate(`/jobs/${data.new_job_id}`);
    },
  });

  const job = jobQuery.data?.job;
  const isTerminal = job?.status ? TERMINAL_STATUSES.includes(job.status) : false;
  const canCancel = job?.status === 'QUEUED' || job?.status === 'RUNNING';
  const canRetry = job?.status === 'FAILED';

  if (jobQuery.isLoading) {
    return (
      <div className="container-fluid py-4 text-center">
        <div className="spinner-border text-primary" />
      </div>
    );
  }

  if (jobQuery.isError) {
    return (
      <div className="container-fluid py-4">
        <ErrorBanner error={jobQuery.error} title="Failed to load job" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="container-fluid py-4">
        <div className="alert alert-warning">Job not found</div>
      </div>
    );
  }

  return (
    <div className="container-fluid py-4">
      <Link to="/jobs" className="btn btn-sm btn-outline-secondary mb-3">
        <ArrowLeft size={14} className="me-1" /> Back to Jobs
      </Link>

      <div className="card shadow-sm border-0 mb-4">
        <div className="card-header bg-white py-3 d-flex justify-content-between align-items-center">
          <div>
            <h2 className="h5 fw-bold mb-1">
              Job <span className="font-monospace">{job.job_id.slice(0, 12)}...</span>
            </h2>
            <div className="d-flex gap-2 align-items-center">
              <JobStatusBadge status={job.status} />
              <span className="badge bg-light text-dark border">{job.job_type}</span>
              {!isTerminal && (
                <span className="badge bg-primary-subtle text-primary border border-primary-subtle">
                  <span className="spinner-border spinner-border-sm me-1" style={{ width: '0.6rem', height: '0.6rem' }} />
                  Polling
                </span>
              )}
            </div>
          </div>
          <div className="d-flex gap-2">
            {canCancel && (
              <button
                className="btn btn-sm btn-outline-warning"
                onClick={() => setShowConfirm('cancel')}
                disabled={cancelMutation.isPending}
              >
                Cancel
              </button>
            )}
            {canRetry && (
              <button
                className="btn btn-sm btn-outline-primary"
                onClick={() => setShowConfirm('retry')}
                disabled={retryMutation.isPending}
              >
                Retry
              </button>
            )}
          </div>
        </div>

        <div className="card-body">
          <div className="row g-3">
            <div className="col-md-6">
              <dl className="mb-0">
                <dt className="small text-muted">Job ID</dt>
                <dd className="font-monospace small">{job.job_id}</dd>
                <dt className="small text-muted">Created</dt>
                <dd className="small">{job.created_at ? new Date(job.created_at).toLocaleString() : '-'}</dd>
                {job.started_at && (
                  <>
                    <dt className="small text-muted">Started</dt>
                    <dd className="small">{new Date(job.started_at).toLocaleString()}</dd>
                  </>
                )}
                {job.ended_at && (
                  <>
                    <dt className="small text-muted">Ended</dt>
                    <dd className="small">{new Date(job.ended_at).toLocaleString()}</dd>
                  </>
                )}
              </dl>
            </div>
            <div className="col-md-6">
              {job.error_message && (
                <div className="alert alert-danger small mb-2">
                  <strong>{job.error_code}:</strong> {job.error_message}
                </div>
              )}
              {job.params_json && (
                <div>
                  <dt className="small text-muted">Parameters</dt>
                  <dd>
                    <pre className="bg-light p-2 rounded small mb-0" style={{ maxHeight: '150px', overflow: 'auto' }}>
                      {JSON.stringify(JSON.parse(job.params_json), null, 2)}
                    </pre>
                  </dd>
                </div>
              )}
              {job.metrics_json && (
                <div>
                  <dt className="small text-muted">Metrics</dt>
                  <dd>
                    <pre className="bg-light p-2 rounded small mb-0" style={{ maxHeight: '150px', overflow: 'auto' }}>
                      {JSON.stringify(JSON.parse(job.metrics_json), null, 2)}
                    </pre>
                  </dd>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Events Timeline */}
      <div className="card shadow-sm border-0">
        <div className="card-header bg-white py-3">
          <h3 className="h6 fw-bold mb-0">Events Timeline</h3>
        </div>
        <div className="card-body">
          {eventsQuery.isLoading ? (
            <div className="text-center p-3">
              <div className="spinner-border spinner-border-sm text-primary" />
            </div>
          ) : eventsQuery.isError ? (
            <ErrorBanner error={eventsQuery.error} title="Failed to load events" />
          ) : !eventsQuery.data?.events?.length ? (
            <div className="text-center p-3 text-muted small">No events recorded yet</div>
          ) : (
            <div className="timeline">
              {eventsQuery.data.events.map((event: { event_id: string; event_type: string; timestamp: string; details_json: string }) => (
                <div key={event.event_id} className="d-flex gap-3 mb-3 pb-3 border-bottom">
                  <div className="flex-shrink-0">
                    <span className="badge bg-secondary-subtle text-secondary border rounded-pill">
                      {event.event_type}
                    </span>
                  </div>
                  <div className="flex-grow-1">
                    <div className="small text-muted">
                      {new Date(event.timestamp).toLocaleString()}
                    </div>
                    {event.details_json && (
                      <pre className="bg-light p-2 rounded small mt-1 mb-0" style={{ maxHeight: '100px', overflow: 'auto' }}>
                        {(() => {
                          try {
                            return JSON.stringify(JSON.parse(event.details_json), null, 2);
                          } catch {
                            return event.details_json;
                          }
                        })()}
                      </pre>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Confirmation Dialog */}
      {showConfirm && (
        <div className="modal d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}>
          <div className="modal-dialog modal-dialog-centered">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">
                  {showConfirm === 'cancel' ? 'Cancel Job?' : 'Retry Job?'}
                </h5>
                <button
                  type="button"
                  className="btn-close"
                  onClick={() => setShowConfirm(null)}
                />
              </div>
              <div className="modal-body">
                {showConfirm === 'cancel'
                  ? 'Are you sure you want to cancel this job? This cannot be undone.'
                  : 'This will create a new job with the same parameters. Continue?'}
              </div>
              <div className="modal-footer">
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setShowConfirm(null)}
                >
                  No, go back
                </button>
                <button
                  className={`btn btn-sm ${showConfirm === 'cancel' ? 'btn-warning' : 'btn-primary'}`}
                  onClick={() => {
                    if (showConfirm === 'cancel') cancelMutation.mutate();
                    else retryMutation.mutate();
                  }}
                  disabled={cancelMutation.isPending || retryMutation.isPending}
                >
                  {(cancelMutation.isPending || retryMutation.isPending)
                    ? 'Processing...'
                    : showConfirm === 'cancel' ? 'Yes, cancel' : 'Yes, retry'}
                </button>
              </div>
              {(cancelMutation.isError || retryMutation.isError) && (
                <div className="modal-footer pt-0">
                  <ErrorBanner
                    error={cancelMutation.error || retryMutation.error}
                    title="Action failed"
                    className="alert alert-danger w-100 mb-0"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
