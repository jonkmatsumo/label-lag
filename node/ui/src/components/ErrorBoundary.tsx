import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCcw, Home } from 'lucide-react';

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  private handleGoHome = () => {
    this.setState({ hasError: false, error: null });
    window.location.href = '/';
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="d-flex align-items-center justify-content-center min-vh-100 bg-light p-4">
          <div className="card shadow-lg border-0" style={{ maxWidth: '500px', width: '100%' }}>
            <div className="card-body p-5 text-center">
              <div className="bg-danger bg-opacity-10 text-danger rounded-circle d-inline-flex p-3 mb-4">
                <AlertTriangle size={48} />
              </div>
              <h2 className="h4 fw-bold mb-3">Something went wrong</h2>
              <p className="text-muted mb-4">
                An unexpected error occurred while rendering this page.
                {this.state.error && <code className="d-block mt-2 p-2 bg-light rounded small text-danger">{this.state.error.message}</code>}
              </p>
              <div className="d-grid gap-2">
                <button onClick={this.handleReset} className="btn btn-primary d-flex align-items-center justify-content-center gap-2">
                  <RefreshCcw size={18} /> Reload Page
                </button>
                <button onClick={this.handleGoHome} className="btn btn-outline-secondary d-flex align-items-center justify-content-center gap-2">
                  <Home size={18} /> Back to Dashboard
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}