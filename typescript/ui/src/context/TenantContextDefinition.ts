import { createContext } from 'react';
import { TenantContextValue } from './types';

export const TenantContext = createContext<TenantContextValue | null>(null);
