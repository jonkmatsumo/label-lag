import { useSearchParams } from 'react-router-dom';
import { useCallback } from 'react';

export function useAnalyticsSearchParams() {
    const [searchParams, setSearchParams] = useSearchParams();

    const getParam = (key: string) => searchParams.get(key) || '';
    const getNumParam = (key: string) => {
        const val = searchParams.get(key);
        return val ? Number(val) : undefined;
    };

    const updateParams = useCallback((newParams: Record<string, unknown>) => {
        setSearchParams((prev) => {
            const updated = new URLSearchParams(prev);
            Object.entries(newParams).forEach(([key, value]) => {
                if (value === undefined || value === '' || value === null) {
                    updated.delete(key);
                } else {
                    updated.set(key, String(value));
                }
            });
            return updated;
        }, { replace: true });
    }, [setSearchParams]);

    return {
        filters: {
            user_id: getParam('user_id'),
            transaction_id: getParam('transaction_id'),
            start_date: getParam('start_date'),
            end_date: getParam('end_date'),
            min_amount: getNumParam('min_amount'),
            max_amount: getNumParam('max_amount'),
            min_score: getNumParam('min_score'),
            max_score: getNumParam('max_score'),
            is_fraudulent: searchParams.get('is_fraudulent') === 'true' ? true : searchParams.get('is_fraudulent') === 'false' ? false : undefined,
        },
        updateParams,
    };
}
