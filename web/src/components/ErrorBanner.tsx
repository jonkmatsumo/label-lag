import { ApiError } from '../api/client';
import { AlertTriangle } from 'lucide-react';

interface ErrorBannerProps {
  error: Error | ApiError | unknown;
  title?: string;
  className?: string;
}

export function ErrorBanner({ error, title = 'An error occurred', className = 'alert alert-danger' }: ErrorBannerProps) {
  const message = error instanceof Error ? error.message : String(error);

  return (
    <div className={className}>
      <div className="d-flex gap-3 align-items-start">
        <AlertTriangle className="flex-shrink-0 mt-1" size={20} />
        <div className="flex-grow-1">
          <h4 className="h6 fw-bold mb-1">{title}</h4>
          <p className="mb-0">{message}</p>
          {error instanceof ApiError && (
            <div className="mt-2 pt-2 border-top border-danger-subtle small font-monospace opacity-75">
              {error.upstreamStatus && <div>Upstream Status: {error.upstreamStatus}</div>}
              {error.requestId && <div>Request ID: {error.requestId}</div>}
              {error.code && <div>Code: {error.code}</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}