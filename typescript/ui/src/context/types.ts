export interface TenantContextValue {
    tenantId: string;
    setTenantId: (id: string) => void;
}

export const DEFAULT_TENANT = import.meta.env.VITE_DEFAULT_TENANT ?? 'default';
