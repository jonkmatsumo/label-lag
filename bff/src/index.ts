import Fastify from 'fastify';
import cors from '@fastify/cors';
import pino from 'pino';
import { loadConfig } from './config.js';
import { requestIdMiddleware } from './middleware/request-id.js';
import { HttpClient, UpstreamError } from './services/http-client.js';
import {
  healthRoutes,
  evaluateRoutes,
  modelRoutes,
  rulesRoutes,
  backtestRoutes,
  analyticsRoutes,
  monitoringRoutes,
  rulesDetailRoutes,
  datasetRoutes,
  mlflowRoutes,
} from './routes/index.js';
import { ErrorResponse } from './types/api.js';

async function main(): Promise<void> {
  const config = loadConfig();

  const logger = pino({
    level: config.logLevel,
    transport:
      process.env.NODE_ENV !== 'production'
        ? { target: 'pino-pretty', options: { colorize: true } }
        : undefined,
  });

  const fastify = Fastify({
    logger,
    disableRequestLogging: false,
  });

  // Register CORS
  await fastify.register(cors, {
    origin: true, // Allow all origins in development
    credentials: true,
  });

  // Register request ID middleware
  fastify.addHook('onRequest', requestIdMiddleware);

  // Create HTTP client for upstream calls
  const httpClient = new HttpClient({ config, logger });

  // Global error handler
  fastify.setErrorHandler((error: unknown, request, reply) => {
    const requestId = request.requestId ?? 'unknown';

    if (error instanceof UpstreamError) {
      logger.warn(
        { requestId, error: error.message, code: error.apiError.code },
        'Upstream error'
      );
      return reply.status(error.statusCode).send(error.toResponse());
    }

    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    const errorStack = error instanceof Error ? error.stack : undefined;

    logger.error(
      { requestId, error: errorMessage, stack: errorStack },
      'Unhandled error'
    );

    const response: ErrorResponse = {
      error: {
        code: 'INTERNAL_ERROR',
        message: 'An unexpected error occurred',
      },
    };

    return reply.status(500).send(response);
  });

  // Register routes
  await fastify.register(healthRoutes, { httpClient });
  await fastify.register(evaluateRoutes, { httpClient, config });
  await fastify.register(modelRoutes, { httpClient });
  await fastify.register(rulesRoutes, { httpClient });
  await fastify.register(backtestRoutes, { httpClient });
  await fastify.register(analyticsRoutes, { httpClient });
  await fastify.register(monitoringRoutes, { httpClient });
  await fastify.register(rulesDetailRoutes, { httpClient });
  await fastify.register(datasetRoutes, { httpClient });
  await fastify.register(mlflowRoutes, { httpClient, mlflowTrackingUri: config.mlflowTrackingUri });

  // Start server
  try {
    const address = await fastify.listen({
      port: config.port,
      host: config.host,
    });

    logger.info(
      {
        address,
        inferenceMode: config.inferenceMode,
        fastApiBaseUrl: config.fastApiBaseUrl,
        mlflowUri: config.mlflowTrackingUri,
      },
      'BFF server started'
    );
  } catch (err) {
    logger.fatal(err, 'Failed to start server');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
