import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from './stores/authStore';
import { useEffect } from 'react';

import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import MarketExplorer from './pages/MarketExplorer';
import CurveViewer from './pages/CurveViewer';
import ScenarioBuilder from './pages/ScenarioBuilder';
import Downloads from './pages/Downloads';
import Login from './pages/Login';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  const { isAuthenticated, initialize } = useAuthStore();

  useEffect(() => {
    initialize();
  }, [initialize]);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          
          {isAuthenticated ? (
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="explorer" element={<MarketExplorer />} />
              <Route path="curves" element={<CurveViewer />} />
              <Route path="scenarios" element={<ScenarioBuilder />} />
              <Route path="downloads" element={<Downloads />} />
            </Route>
          ) : (
            <Route path="*" element={<Navigate to="/login" replace />} />
          )}
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;

