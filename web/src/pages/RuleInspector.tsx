import { Outlet, Link, useLocation } from 'react-router-dom';

const ruleTabs = [
  { path: '/rules', label: 'Management', exact: true },
  { path: '/rules/sandbox', label: 'Sandbox' },
  { path: '/rules/shadow', label: 'Shadow Metrics' },
  { path: '/rules/backtests', label: 'Backtests' },
  { path: '/rules/suggestions', label: 'Suggestions' },
];

export function RuleInspector() {
  const location = useLocation();

  return (
    <div className="page">
      <h2>Rule Inspector</h2>
      <div className="tabs">
        {ruleTabs.map((tab) => {
          const isActive = tab.exact
            ? location.pathname === tab.path
            : location.pathname.startsWith(tab.path);

          return (
            <Link
              key={tab.path}
              to={tab.path}
              className={`tab ${isActive ? 'active' : ''}`}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
      <div className="tab-content">
        <Outlet />
      </div>
    </div>
  );
}

export function RuleManagement() {
  return (
    <div>
      <h3>Rule Management</h3>
      <p>View production rules and manage lifecycles</p>
      {/* P0 implementation coming in Phase 1 */}
    </div>
  );
}

export function RuleSandbox() {
  return (
    <div>
      <h3>Sandbox</h3>
      <p>Test rules against sample transactions</p>
      {/* P0 implementation coming in Phase 1 */}
    </div>
  );
}

export function RuleShadow() {
  return (
    <div>
      <h3>Shadow Metrics</h3>
      <p>Compare production vs shadow mode</p>
      {/* P1 implementation coming in Phase 2 */}
    </div>
  );
}

export function RuleBacktests() {
  return (
    <div>
      <h3>Backtests</h3>
      <p>Run and view backtest results</p>
      {/* P1 implementation coming in Phase 2 */}
    </div>
  );
}

export function RuleSuggestions() {
  return (
    <div>
      <h3>Suggestions</h3>
      <p>AI-generated rule recommendations</p>
      {/* P2 implementation coming in Phase 2 */}
    </div>
  );
}
