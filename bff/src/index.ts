import Fastify from 'fastify';
import cors from '@fastify/cors';
import jwt from '@fastify/jwt';
import pino from 'pino';
import { loadConfig } from './config.js';
import { requestIdMiddleware } from './middleware/request-id.js';
import { HttpClient, UpstreamError } from './services/http-client.js';
import { SimpleCache } from './services/cache.js';
import { authenticate, authorize } from './middleware/auth.js';
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
    loggerInstance: logger,
    disableRequestLogging: false,
  });

  // Register CORS
  await fastify.register(cors, {
    origin: (origin, cb) => {
      const allowed = config.corsOrigin;
      if (allowed === 'true') {
        cb(null, true);
        return;
      }
      if (!origin) {
        cb(null, true); // Allow requests with no origin (like curl)
        return;
      }
      const origins = allowed.split(',').map(s => s.trim());
      if (origins.includes(origin)) {
        cb(null, true);
      } else {
        cb(new Error('Not allowed by CORS'), false);
      }
    },
    credentials: true,
  });

  // Register JWT
  await fastify.register(jwt, {
    secret: config.authJwtSecret,
  });

  // Register request ID middleware
  fastify.addHook('onRequest', requestIdMiddleware);

  // Create HTTP client for upstream calls
  const httpClient = new HttpClient({ config, logger });
  const cache = new SimpleCache(config, logger);

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
        // @ts-ignore - extending ErrorResponse locally for now or we update the type def
        request_id: requestId,
      },
    };

    return reply.status(500).send(response);
  });

  // Register routes
  await fastify.register(healthRoutes, { httpClient });

  // Dev-only login endpoint
  fastify.post('/bff/v1/auth/dev-login', async (request) => {
    const { role = 'admin' } = request.body as { role?: string };
    const token = fastify.jwt.sign({ 
      sub: 'dev-user', 
      role: role as 'admin' | 'scoring-only',
      name: 'Dev User' 
    });
    return { token };
  });

  await fastify.register(evaluateRoutes, { httpClient, config });
  
  // Protected routes
  await fastify.register(async (protectedPart) => {
    protectedPart.addHook('preHandler', authenticate);
    
    await protectedPart.register(analyticsRoutes, { httpClient, cache });
    await protectedPart.register(monitoringRoutes, { httpClient });
    
    // Admin-only routes
    await protectedPart.register(async (adminPart) => {
      adminPart.addHook('preHandler', authorize(['admin']));
      
      await adminPart.register(modelRoutes, { httpClient });
      await adminPart.register(rulesRoutes, { httpClient });
      await adminPart.register(backtestRoutes, { httpClient });
      await adminPart.register(rulesDetailRoutes, { httpClient });
      await adminPart.register(datasetRoutes, { httpClient });
      await adminPart.register(mlflowRoutes, { httpClient, mlflowTrackingUri: config.mlflowTrackingUri });
    });
  });

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
