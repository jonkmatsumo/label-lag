import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import Fastify, { FastifyInstance } from 'fastify';
import { MockAgent, setGlobalDispatcher } from 'undici';
import { mlflowRoutes } from '../src/routes/mlflow';
import { HttpClient } from '../src/services/http-client';

describe('MLflow Routes', () => {
  let fastify: FastifyInstance;
  let mockAgent: MockAgent;

  beforeAll(async () => {
    mockAgent = new MockAgent();
    mockAgent.disableNetConnect();
    setGlobalDispatcher(mockAgent);

    fastify = Fastify();
    
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

    await fastify.register(mlflowRoutes, { 
      httpClient, 
      mlflowTrackingUri: 'http://mlflow:5000' 
    });
  });

  afterAll(async () => {
    await fastify.close();
  });

  it('GET /bff/v1/mlflow/model-versions/search should proxy to MLflow', async () => {
    const mockPool = mockAgent.get('http://mlflow:5000');
    mockPool.intercept({
      path: '/api/2.0/mlflow/model-versions/search',
      method: 'GET'
    }).reply(200, {
      model_versions: [{ version: "1", current_stage: "Production" }]
    });

    const response = await fastify.inject({
      method: 'GET',
      url: '/bff/v1/mlflow/model-versions/search'
    });

    expect(response.statusCode).toBe(200);
    const body = JSON.parse(response.payload);
    expect(body.model_versions[0].version).toBe("1");
  });

  it('GET /bff/v1/mlflow/runs/:id/artifacts should return content', async () => {
    const mockPool = mockAgent.get('http://mlflow:5000');
    mockPool.intercept({
      path: /\/get-artifact.*/,
      method: 'GET'
    }).reply(200, { accuracy: 0.9 });

    const response = await fastify.inject({
      method: 'GET',
      url: '/bff/v1/mlflow/runs/abc/artifacts',
      query: { path: 'metrics.json' }
    });

    expect(response.statusCode).toBe(200);
    expect(JSON.parse(response.payload).accuracy).toBe(0.9);
  });
});
