import { Link, useLocation, Outlet } from 'react-router-dom';

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
];

export function Layout() {
  const location = useLocation();

  return (
    <div className="layout">
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1>Label Lag</h1>
          <span className="subtitle">Fraud Detection Platform</span>
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
    </div>
  );
}
