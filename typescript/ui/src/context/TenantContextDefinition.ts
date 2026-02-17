import { createContext } from 'react';
import type { TenantContextValue } from './types';

export const TenantContext = createContext<TenantContextValue | null>(null);
