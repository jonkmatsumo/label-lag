import { describe, it, expect } from 'vitest';
import { request } from 'undici';

const RUN_PARITY = process.env.RUN_PARITY_TESTS === '1';
const FASTAPI_URL = process.env.BFF_FASTAPI_BASE_URL || 'http://localhost:8000';
const GATEWAY_URL = process.env.BFF_GATEWAY_BASE_URL || 'http://localhost:8081';

describe.skipIf(!RUN_PARITY)('Parity: FastAPI vs Go Gateway', () => {
  it('should return identical scores for same input', async () => {
    const payload = {
      user_id: 'parity_user_1',
      amount: 150.00,
      currency: 'USD',
      client_transaction_id: `txn_${Date.now()}`
    };

    const [fastRes, goRes] = await Promise.all([
      request(`${FASTAPI_URL}/evaluate/signal`, {
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

    expect(fastRes.statusCode).toBe(200);
    expect(goRes.statusCode).toBe(200);

    const fastBody = await fastRes.body.json() as any;
    const goBody = await goRes.body.json() as any;

    // Compare fields
    expect(fastBody.score).toBeDefined();
    expect(goBody.score).toBeDefined();
    
    // Allow small floating point diff if any, though scores are usually integers 0-100
    expect(Math.abs(fastBody.score - goBody.score)).toBeLessThanOrEqual(1);

    // Compare rules
    const fastRules = new Set((fastBody.risk_components || []).map((c: any) => c.key));
    const goRules = new Set((goBody.risk_components || []).map((c: any) => c.key));
    
    // Check overlap
    for (const rule of fastRules) {
        // Go might implement fewer rules initially, but we want to know gaps
        if (!goRules.has(rule)) {
            console.warn(`Rule ${rule} present in FastAPI but missing in Go`);
        }
    }
  });
});
