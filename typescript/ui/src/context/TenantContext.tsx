import { useState, useCallback, type ReactNode } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { DEFAULT_TENANT } from './types';
import { TenantContext } from './TenantContextDefinition';

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
