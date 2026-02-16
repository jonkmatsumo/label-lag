import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components';
import { ErrorBoundary } from './components/ErrorBoundary';
import { TenantProvider } from './context/TenantContext';
import {
  LiveScoring,
  Analytics,
  Dataset,
  ModelLab,
  RuleInspector,
  RuleManagement,
  RuleSandbox,
  RuleShadow,
  RuleBacktests,
  RuleSuggestions,
  WhatIf,
  Jobs,
  JobDetail,
  DecisionsPage,
  DecisionDetail,
  TrainingRunsPage,
  TrainingRunDetail,
  ModelVersionsPage,
  ModelVersionDetail,
} from './pages';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TenantProvider>
        <ErrorBoundary>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Layout />}>
                <Route index element={<LiveScoring />} />
                <Route path="analytics" element={<Analytics />} />
                <Route path="dataset" element={<Dataset />} />
                <Route path="model-lab" element={<ModelLab />} />
                <Route path="rules" element={<RuleInspector />}>
                  <Route index element={<RuleManagement />} />
                  <Route path="sandbox" element={<RuleSandbox />} />
                  <Route path="shadow" element={<RuleShadow />} />
                  <Route path="backtests" element={<RuleBacktests />} />
                  <Route path="suggestions" element={<RuleSuggestions />} />
                </Route>
                <Route path="what-if" element={<WhatIf />} />
                <Route path="jobs" element={<Jobs />} />
                <Route path="jobs/:id" element={<JobDetail />} />
                <Route path="decisions" element={<DecisionsPage />} />
                <Route path="decisions/:id" element={<DecisionDetail />} />
                <Route path="training" element={<TrainingRunsPage />} />
                <Route path="training-runs/:id" element={<TrainingRunDetail />} />
                <Route path="models" element={<ModelVersionsPage />} />
                <Route path="models/:version" element={<ModelVersionDetail />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ErrorBoundary>
      </TenantProvider>
    </QueryClientProvider>
  );
}

export default App;
