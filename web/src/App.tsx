import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components';
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
} from './pages';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000, // 30 seconds
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
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
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
