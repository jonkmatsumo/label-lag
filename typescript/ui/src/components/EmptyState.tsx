import type { ReactNode } from 'react';

interface EmptyStateProps {
    title: string;
    description?: string;
    icon?: ReactNode;
    action?: ReactNode;
}

export function EmptyState({ title, description, icon, action }: EmptyStateProps) {
    return (
        <div className="text-center p-5">
            {icon && <div className="mb-3 text-muted" style={{ fontSize: '2rem' }}>{icon}</div>}
            <h5 className="text-muted fw-bold">{title}</h5>
            {description && <p className="text-muted mb-4">{description}</p>}
            {action && <div>{action}</div>}
        </div>
    );
}
