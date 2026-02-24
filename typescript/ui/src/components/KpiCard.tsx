import { AlertCircle } from 'lucide-react';

interface KpiCardProps {
    label: string;
    value: string | number;
    loading?: boolean;
    error?: unknown;
    suffix?: string;
    formatter?: (val: unknown) => string;
    badge?: React.ReactNode;
    deltaLabel?: string;
    deltaTone?: 'positive' | 'negative' | 'neutral';
}

export function KpiCard({
    label,
    value,
    loading,
    error,
    suffix,
    formatter,
    badge,
    deltaLabel,
    deltaTone = 'neutral',
}: KpiCardProps) {
    const displayValue = formatter ? formatter(value) : value;
    const deltaClassName =
        deltaTone === 'positive'
            ? 'text-success'
            : deltaTone === 'negative'
              ? 'text-danger'
              : 'text-muted';

    return (
        <div className="card shadow-sm border-0 h-100">
            <div className="card-body p-3">
                <div className="d-flex justify-content-between align-items-center mb-1">
                    <div className="text-muted small fw-bold text-uppercase">{label}</div>
                    {badge}
                </div>

                {loading ? (
                    <div className="placeholder-glow">
                        <span className="placeholder col-8 h4 mb-0"></span>
                    </div>
                ) : error ? (
                    <div className="text-danger small d-flex align-items-center gap-1">
                        <AlertCircle size={14} /> Failed to load
                    </div>
                ) : (
                    <div className="h4 mb-0 fw-bold">
                        {displayValue}
                        {suffix && <span className="small text-muted ms-1 fw-normal">{suffix}</span>}
                    </div>
                )}
                {!loading && !error && deltaLabel && (
                    <div className={`small mt-1 ${deltaClassName}`}>
                        {deltaLabel}
                    </div>
                )}
            </div>
        </div>
    );
}
