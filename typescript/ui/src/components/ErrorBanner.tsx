import { ApiError } from '../api/client';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBannerProps {
  error: Error | ApiError | unknown;
  title?: string;
  className?: string;
  onRetry?: () => void;
  onReset?: () => void;
}

function getContextualHint(error: unknown): string | null {
  if (!(error instanceof ApiError)) return null;

  const status = error.upstreamStatus ?? error.statusCode;

  if (status === 400 && error.message.toLowerCase().includes('pagination')) {
    return 'Try resetting filters or refreshing the page.';
  }
  if (status === 404) {
    return 'The requested resource was not found.';
  }
  if (status === 503) {
    return 'The service is temporarily unavailable. Please retry in a moment.';
  }
  if (status === 504 || error.code === 'GATEWAY_TIMEOUT') {
    return 'The query took too long. Try narrowing your filters.';
  }
  if (error.code === 'NETWORK_ERROR') {
    return 'Could not reach the server. Check your connection and retry.';
  }
  return null;
}

export function ErrorBanner({ error, title = 'An error occurred', className = 'alert alert-danger', onRetry, onReset }: ErrorBannerProps) {
  const message = error instanceof Error ? error.message : String(error);
  const hint = getContextualHint(error);
  const isPaginationConflict =
    error instanceof ApiError &&
    (error.upstreamStatus ?? error.statusCode) === 400 &&
    error.message.toLowerCase().includes('pagination');

  return (
    <div className={className}>
      <div className="d-flex gap-3 align-items-start">
        <AlertTriangle className="flex-shrink-0 mt-1" size={20} />
        <div className="flex-grow-1">
          <h4 className="h6 fw-bold mb-1">{title}</h4>
          <p className="mb-0">{message}</p>
          {hint && <p className="mb-0 mt-1 small text-muted">{hint}</p>}
          {error instanceof ApiError && (
            <div className="mt-2 pt-2 border-top border-danger-subtle small font-monospace opacity-75">
              {error.upstreamStatus && <div>Upstream Status: {error.upstreamStatus}</div>}
              {error.requestId && <div>Request ID: {error.requestId}</div>}
              {error.code && <div>Code: {error.code}</div>}
            </div>
          )}
          {(onRetry || (isPaginationConflict && onReset)) && (
            <div className="mt-2 d-flex gap-2">
              {onRetry && (
                <button className="btn btn-sm btn-outline-danger" onClick={onRetry}>
                  <RefreshCw size={12} className="me-1" /> Retry
                </button>
              )}
              {isPaginationConflict && onReset && (
                <button className="btn btn-sm btn-outline-secondary" onClick={onReset}>
                  Reset
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
