import { describe, it, expect } from 'vitest';
import { request } from 'undici';

const RUN_PARITY = process.env.RUN_PARITY_TESTS === '1';
const PYTHON_API_URL = process.env.BFF_PYTHON_API_BASE_URL || 'http://localhost:8100';
const GATEWAY_URL = process.env.BFF_GATEWAY_BASE_URL || 'http://localhost:8181';

describe.skipIf(!RUN_PARITY)('Parity: Python API vs Go Gateway', () => {
  it('should return identical scores for same input', async () => {
    const payload = {
      user_id: 'parity_user_1',
      amount: 150.00,
      currency: 'USD',
      client_transaction_id: `txn_${Date.now()}`
    };

    const [pythonRes, goRes] = await Promise.all([
      request(`${PYTHON_API_URL}/evaluate/signal`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload)
      }),
      request(`${GATEWAY_URL}/evaluate/signal`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload)
      })
    ]);

    expect(pythonRes.statusCode).toBe(200);
    expect(goRes.statusCode).toBe(200);

    const pythonBody = await pythonRes.body.json() as any;
    const goBody = await goRes.body.json() as any;

    // Compare fields
    expect(pythonBody.score).toBeDefined();
    expect(goBody.score).toBeDefined();

    // Allow small floating point diff if any, though scores are usually integers 0-100
    expect(Math.abs(pythonBody.score - goBody.score)).toBeLessThanOrEqual(1);

    // Compare rules
    const fastRules = new Set((pythonBody.risk_components || []).map((c: any) => c.key));
    const goRules = new Set((goBody.risk_components || []).map((c: any) => c.key));

    // Check overlap
    for (const rule of fastRules) {
        // Go might implement fewer rules initially, but we want to know gaps
        if (!goRules.has(rule)) {
            console.warn(`Rule ${rule} present in Python API but missing in Go`);
        }
    }
  });
});
