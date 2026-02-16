import { useState, useEffect } from 'react';
import { useLocation, useSearchParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { useTenant } from '../context/TenantContext';

export function DebugDrawer() {
    const [isOpen, setIsOpen] = useState(false);
    const [searchParams] = useSearchParams();
    const location = useLocation();
    const { tenantId, setTenantId } = useTenant();
    const queryClient = useQueryClient();

    // Auto-show if ?debug=true
    useEffect(() => {
        if (searchParams.get('debug') === 'true') {
            setIsOpen(true);
        }
    }, [searchParams]);

    if (!isOpen && searchParams.get('debug') !== 'true') return null;

    return (
        <div
            className={`position-fixed top-0 end-0 h-100 bg-white shadow-lg border-start transition-transform ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}
            style={{ width: '300px', zIndex: 1050, transform: isOpen ? 'translateX(0)' : 'translateX(100%)', transition: 'transform 0.3s ease-in-out' }}
        >
            <div className="d-flex justify-content-between align-items-center p-3 border-bottom bg-light">
                <h6 className="mb-0 fw-bold text-danger">Debug Panel</h6>
                <button className="btn-close" onClick={() => setIsOpen(false)}></button>
            </div>

            <div className="p-3">
                <div className="mb-4">
                    <label className="form-label small fw-bold">Current Tenant</label>
                    <input
                        type="text"
                        className="form-control form-control-sm font-monospace"
                        value={tenantId}
                        onChange={(e) => setTenantId(e.target.value)}
                    />
                    <div className="form-text small">Changes applied immediately.</div>
                </div>

                <div className="mb-4">
                    <label className="form-label small fw-bold">Current Route</label>
                    <div className="p-2 bg-light rounded small font-monospace text-wrap text-break">
                        {location.pathname}
                    </div>
                    {location.search && (
                        <div className="mt-1 p-2 bg-light rounded small font-monospace text-wrap text-break">
                            {location.search}
                        </div>
                    )}
                </div>

                <div className="mb-4">
                    <label className="form-label small fw-bold">Query Cache</label>
                    <div className="d-grid gap-2">
                        <button
                            className="btn btn-sm btn-outline-secondary"
                            onClick={() => queryClient.invalidateQueries()}
                        >
                            Invalidate All Queries
                        </button>
                        <button
                            className="btn btn-sm btn-outline-danger"
                            onClick={() => queryClient.clear()}
                        >
                            Clear Cache
                        </button>
                    </div>
                </div>

                <hr />

                <div className="small text-muted">
                    <p className="mb-1"><strong>Environment:</strong></p>
                    <ul className="ps-3 mb-0">
                        <li>Mode: {import.meta.env.MODE}</li>
                        <li>Dev: {import.meta.env.DEV ? 'Yes' : 'No'}</li>
                    </ul>
                </div>
            </div>

            {/* Toggle Button (visible when closed but debug mode enabled via other means, or just always hidden if closed via X? Requirement says ?debug=true enables it. I'll add a floating toggle if closed but debug param present? Or just rely on URL.) */}
        </div>
    );
}

// Floating toggle button to re-open if closed
export function DebugToggle({ onClick }: { onClick: () => void }) {
    const [searchParams] = useSearchParams();
    if (searchParams.get('debug') !== 'true') return null;

    return (
        <button
            className="btn btn-sm btn-danger position-fixed bottom-0 end-0 m-3 rounded-circle shadow"
            style={{ width: '40px', height: '40px', zIndex: 1060 }}
            onClick={onClick}
        >
            🐛
        </button>
    );
}
