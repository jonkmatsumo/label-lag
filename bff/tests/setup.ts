import Fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';
import { MockAgent, setGlobalDispatcher, getGlobalDispatcher, Dispatcher } from 'undici';
import { requestIdMiddleware } from '../src/middleware/request-id.js';
import { HttpClient } from '../src/services/http-client.js';
import { Config } from '../src/config.js';
import {
  healthRoutes,
  evaluateRoutes,
  modelRoutes,
  rulesRoutes,
  backtestRoutes,
  analyticsRoutes,
  monitoringRoutes,
  rulesDetailRoutes,
} from '../src/routes/index.js';
import pino from 'pino';

export interface TestContext {
  app: FastifyInstance;
  config: Config;
  mockAgent: MockAgent;
  mockPool: ReturnType<MockAgent['get']>;
  originalDispatcher: Dispatcher;
}

export function createTestConfig(): Config {
  return {
    port: 3001,
    host: '127.0.0.1',
    fastApiBaseUrl: 'http://mock-api:8000',
    mlflowTrackingUri: 'http://mock-mlflow:5000',
    inferenceMode: 'fastapi',
    gatewayBaseUrl: 'http://mock-gateway:8081',
    requestTimeout: 5000,
    logLevel: 'silent',
    testMode: true,
  };
}

export async function createTestApp(config?: Config): Promise<TestContext> {
  const testConfig = config ?? createTestConfig();

  // Store original dispatcher and set up mock agent
  const originalDispatcher = getGlobalDispatcher();
  const mockAgent = new MockAgent();
  mockAgent.disableNetConnect();
  setGlobalDispatcher(mockAgent);

  // Create a mock pool for the FastAPI backend
  const mockPool = mockAgent.get('http://mock-api:8000');

  const logger = pino({ level: 'silent' });

  const app = Fastify({
    logger: false,
  });

  await app.register(cors, {
    origin: true,
    credentials: true,
  });

  app.addHook('onRequest', requestIdMiddleware);

  const httpClient = new HttpClient({ config: testConfig, logger });

  await app.register(healthRoutes, { httpClient });
  await app.register(evaluateRoutes, { httpClient, config: testConfig });
  await app.register(modelRoutes, { httpClient });
  await app.register(rulesRoutes, { httpClient });
  await app.register(backtestRoutes, { httpClient });
  await app.register(analyticsRoutes, { httpClient });
  await app.register(monitoringRoutes, { httpClient });
  await app.register(rulesDetailRoutes, { httpClient });

  return { app, config: testConfig, mockAgent, mockPool, originalDispatcher };
}

export function restoreDispatcher(originalDispatcher: Dispatcher): void {
  setGlobalDispatcher(originalDispatcher);
}
