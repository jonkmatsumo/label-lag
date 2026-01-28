import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import { HealthResponse } from '../types/api.js';

export interface HealthRoutesOptions {
  httpClient: HttpClient;
}

/**
 * Health check routes
 */
export async function healthRoutes(
  fastify: FastifyInstance,
  options: HealthRoutesOptions
): Promise<void> {
  const { httpClient } = options;

  // BFF own health - always returns healthy if the process is running
  fastify.get('/health', async (_request: FastifyRequest, reply: FastifyReply) => {
    const uptimeSeconds = process.uptime();

    return reply.send({
      status: 'healthy',
      version: '1.0.0',
      uptime_seconds: Math.floor(uptimeSeconds),
      timestamp: new Date().toISOString(),
    });
  });

  // Proxy to FastAPI health - provides model status
  fastify.get(
    '/bff/v1/health',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request<HealthResponse>({
          method: 'GET',
          path: '/health',
          requestId: request.requestId,
        });

        return reply.status(response.statusCode).send(response.data);
      } catch (error) {
        if (error instanceof UpstreamError) {
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );
}
