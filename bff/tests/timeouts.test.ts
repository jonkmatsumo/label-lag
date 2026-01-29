import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import Fastify, { FastifyInstance } from 'fastify';
import { MockAgent, setGlobalDispatcher } from 'undici';
import { HttpClient } from '../src/services/http-client';
import pino from 'pino';

describe('HttpClient Timeouts and Retries', () => {
  let mockAgent: MockAgent;
  let logger: pino.Logger;

  beforeAll(() => {
    mockAgent = new MockAgent();
    mockAgent.disableNetConnect();
    setGlobalDispatcher(mockAgent);
    logger = pino({ level: 'silent' });
  });

  const config = {
    port: 3000,
    host: 'localhost',
    fastApiBaseUrl: 'http://api',
    mlflowTrackingUri: 'http://mlflow',
    inferenceMode: 'fastapi' as const,
    gatewayBaseUrl: 'http://gateway',
    requestTimeout: 1000, // Long timeout
    upstreamTimeout: 50,  // Short timeout default
    logLevel: 'silent',
    testMode: true
  };

  it('should timeout if upstream takes too long', async () => {
    const client = new HttpClient({ config, logger });
    const mockPool = mockAgent.get('http://api');
    
    mockPool.intercept({
      path: '/slow',
      method: 'GET'
    }).reply(200, { ok: true }).delay(100).persist(); // Delay > upstreamTimeout, persist for retries

    await expect(client.request({
      method: 'GET',
      path: '/slow',
      requestId: 'req1'
    })).rejects.toThrow(/timed out/);
  });

  it('should retry GET requests on 503', async () => {
    const client = new HttpClient({ config, logger });
    const mockPool = mockAgent.get('http://api');
    
    // First attempt fails
    mockPool.intercept({
      path: '/flaky',
      method: 'GET'
    }).reply(503, { error: 'Service Unavailable' });

    // Second attempt succeeds
    mockPool.intercept({
      path: '/flaky',
      method: 'GET'
    }).reply(200, { ok: true });

    const response = await client.request<{ ok: boolean }>({
      method: 'GET',
      path: '/flaky',
      requestId: 'req2'
    });

    expect(response.statusCode).toBe(200);
    expect(response.data.ok).toBe(true);
  });

  it('should NOT retry POST requests', async () => {
    const client = new HttpClient({ config, logger });
    const mockPool = mockAgent.get('http://api');
    
    mockPool.intercept({
      path: '/mutate',
      method: 'POST'
    }).reply(503, { error: 'Service Unavailable' });

    await expect(client.request({
      method: 'POST',
      path: '/mutate',
      requestId: 'req3',
      body: {}
    })).rejects.toThrow('Service Unavailable');
  });

  it('should use override timeout if provided', async () => {
    const client = new HttpClient({ config, logger });
    const mockPool = mockAgent.get('http://api');
    
    mockPool.intercept({
      path: '/long-job',
      method: 'POST'
    }).reply(200, { ok: true }).delay(100);

    // Override timeout to 200ms (longer than delay)
    const response = await client.request<{ ok: boolean }>({
      method: 'POST',
      path: '/long-job',
      requestId: 'req4',
      timeout: 200
    });

    expect(response.statusCode).toBe(200);
  });
});
