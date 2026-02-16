import { Link, useLocation, Outlet } from 'react-router-dom';
import { useTenant } from '../hooks/useTenant';
import { DebugDrawer, DebugToggle } from './DebugDrawer';

interface NavItem {
  path: string;
  label: string;
  icon: string;
}

const navItems: NavItem[] = [
  { path: '/', label: 'Live Scoring', icon: '⚡' },
  { path: '/analytics', label: 'Historical Analytics', icon: '📊' },
  { path: '/dataset', label: 'Synthetic Dataset', icon: '🗃️' },
  { path: '/model-lab', label: 'Model Lab', icon: '🧪' },
  { path: '/rules', label: 'Rule Inspector', icon: '📋' },
  { path: '/what-if', label: 'What-If Simulation', icon: '🔮' },
  { path: '/jobs', label: 'Jobs', icon: '⚙️' },
  { path: '/decisions', label: 'Decisions', icon: '⚖️' },
  { path: '/training', label: 'Training Runs', icon: '🏋️' },
  { path: '/models', label: 'Model Registry', icon: '🧠' },
  { path: '/dataset/profiles', label: 'Dataset Profiles', icon: '📈' },
];

export function Layout() {
  const location = useLocation();
  const { tenantId } = useTenant();

  return (
    <div className="layout">
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1>Label Lag</h1>
          <span className="subtitle">Fraud Detection Platform</span>
          <span className="badge bg-secondary mt-1" style={{ fontSize: '0.7rem' }}>
            Tenant: {tenantId}
          </span>
        </div>
        <ul className="nav-list">
          {navItems.map((item) => {
            const isActive =
              item.path === '/'
                ? location.pathname === '/'
                : location.pathname.startsWith(item.path);

            return (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`nav-link ${isActive ? 'active' : ''}`}
                >
                  <span className="nav-icon">{item.icon}</span>
                  <span className="nav-label">{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <main className="content">
        <Outlet />
      </main>
      <DebugDrawer />
      <DebugToggle onClick={() => {
        // This is a bit hacky, the drawer state is internal.
        // Ideally we hoist state or use context.
        // For now, I'll let the user open it via URL parameter?
        // Actually, DebugToggle should probably just set the URL param if not present?
        const url = new URL(window.location.href);
        url.searchParams.set('debug', 'true');
        window.history.pushState({}, '', url);
        // Force re-render or listen to URL change?
        // DebugDrawer listens to useSearchParams.
      }} />
    </div>
  );
}
