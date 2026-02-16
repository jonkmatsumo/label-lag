import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';

interface TenantContextValue {
  tenantId: string;
  setTenantId: (id: string) => void;
}

const TenantContext = createContext<TenantContextValue | null>(null);

const DEFAULT_TENANT = import.meta.env.VITE_DEFAULT_TENANT ?? 'default';

export function TenantProvider({ children }: { children: ReactNode }) {
  const [tenantId, setTenantIdState] = useState(DEFAULT_TENANT);
  const queryClient = useQueryClient();

  // Sync tenant to API client and invalidate all queries on change
  const setTenantId = useCallback((id: string) => {
    setTenantIdState(id);
    apiClient.setTenantId(id);
    queryClient.invalidateQueries();
  }, [queryClient]);

  // Set initial tenant on the API client
  apiClient.setTenantId(tenantId);

  return (
    <TenantContext.Provider value={{ tenantId, setTenantId }}>
      {children}
    </TenantContext.Provider>
  );
}

export function useTenant(): TenantContextValue {
  const ctx = useContext(TenantContext);
  if (!ctx) {
    throw new Error('useTenant must be used within a TenantProvider');
  }
  return ctx;
}
