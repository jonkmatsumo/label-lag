import { useContext } from 'react';
import { TenantContext } from '../context/TenantContextDefinition';
import type { TenantContextValue } from '../context/types';

export function useTenant(): TenantContextValue {
    const ctx = useContext(TenantContext);
    if (!ctx) {
        throw new Error('useTenant must be used within a TenantProvider');
    }
    return ctx;
}
