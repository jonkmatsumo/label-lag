import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import Fastify, { FastifyInstance } from 'fastify';
import { MockAgent, setGlobalDispatcher } from 'undici';
import { datasetRoutes } from '../src/routes/dataset';
import { HttpClient } from '../src/services/http-client';
import { requestIdMiddleware } from '../src/middleware/request-id';

describe('Dataset Routes', () => {
  let fastify: FastifyInstance;
  let mockAgent: MockAgent;

  beforeAll(async () => {
    mockAgent = new MockAgent();
    mockAgent.disableNetConnect();
    setGlobalDispatcher(mockAgent);

    fastify = Fastify();
    fastify.addHook('onRequest', requestIdMiddleware);
    
    const httpClient = new HttpClient({
      config: {
        fastApiBaseUrl: 'http://api:8000',
        mlflowTrackingUri: 'http://mlflow:5000',
        requestTimeout: 1000,
        logLevel: 'silent',
        port: 3000,
        host: 'localhost',
        inferenceMode: 'fastapi',
        gatewayBaseUrl: '',
        testMode: true,
        upstreamTimeout: 1000,
        cacheEnabled: false,
        cacheTtlMs: 1000
      },
      logger: fastify.log
    });

    await fastify.register(datasetRoutes, { httpClient });
  });

  afterAll(async () => {
    await fastify.close();
  });

  it('GET /bff/v1/dataset/overview should return metrics', async () => {
    const mockPool = mockAgent.get('http://api:8000');
    mockPool.intercept({
      path: '/analytics/overview',
      method: 'GET'
    }).reply(200, {
      total_records: 1000,
      fraud_rate: 0.05
    });

    const response = await fastify.inject({
      method: 'GET',
      url: '/bff/v1/dataset/overview'
    });

    expect(response.statusCode).toBe(200);
    const body = JSON.parse(response.payload);
    expect(body.total_records).toBe(1000);
  });

  it('POST /bff/v1/dataset/generate should trigger generation', async () => {
    const mockPool = mockAgent.get('http://api:8000');
    mockPool.intercept({
      path: '/data/generate',
      method: 'POST'
    }).reply(200, {
      success: true,
      total_records: 500
    });

    const response = await fastify.inject({
      method: 'POST',
      url: '/bff/v1/dataset/generate',
      payload: { num_users: 100, fraud_rate: 0.1 }
    });

    expect(response.statusCode).toBe(200);
    expect(JSON.parse(response.payload).success).toBe(true);
  });

  it('GET /bff/v1/dataset/sample should return feature samples', async () => {
    const mockPool = mockAgent.get('http://api:8000');
    mockPool.intercept({
      path: /\/analytics\/feature-sample.*/,
      method: 'GET'
    }).reply(200, {
      samples: [{ velocity_24h: 5.0, is_fraudulent: false }]
    });

    const response = await fastify.inject({
      method: 'GET',
      url: '/bff/v1/dataset/sample',
      query: { sample_size: '10', stratify: 'true' }
    });

    expect(response.statusCode).toBe(200);
    expect(JSON.parse(response.payload).samples).toHaveLength(1);
  });
});
