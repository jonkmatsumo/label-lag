import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';

export interface DatasetRoutesOptions {
  httpClient: HttpClient;
}

export async function datasetRoutes(
  fastify: FastifyInstance,
  options: DatasetRoutesOptions
): Promise<void> {
  const { httpClient } = options;

  // GET /bff/v1/dataset/overview
  fastify.get(
    '/bff/v1/dataset/overview',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'GET',
          path: '/analytics/overview', // Proxies to Python API
          requestId: request.requestId,
          authToken: request.headers.authorization,
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

  // GET /bff/v1/dataset/schema
  fastify.get(
    '/bff/v1/dataset/schema',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'GET',
          path: '/analytics/schema',
          requestId: request.requestId,
          authToken: request.headers.authorization,
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

  // POST /bff/v1/dataset/generate
  fastify.post(
    '/bff/v1/dataset/generate',
    {
      schema: {
        body: {
          type: 'object',
          required: ['num_users', 'fraud_rate'],
          properties: {
            num_users: { type: 'number', minimum: 1 },
            fraud_rate: { type: 'number', minimum: 0, maximum: 1 },
            drop_existing: { type: 'boolean' },
          },
        },
      },
    },
    async (request: FastifyRequest<{ Body: { num_users: number; fraud_rate: number; drop_existing?: boolean } }>, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'POST',
          path: '/data/generate',
          body: request.body,
          requestId: request.requestId,
          timeout: 300000, // 5 min timeout for generation
          authToken: request.headers.authorization,
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

  // DELETE /bff/v1/dataset/clear
  fastify.delete(
    '/bff/v1/dataset/clear',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'DELETE',
          path: '/data/clear',
          requestId: request.requestId,
          timeout: 60000, // 1 min timeout
          authToken: request.headers.authorization,
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

  // GET /bff/v1/dataset/sample
  fastify.get(
    '/bff/v1/dataset/sample',
    {
        schema: {
            querystring: {
                type: 'object',
                properties: {
                    sample_size: { type: 'integer', default: 1000 },
                    stratify: { type: 'boolean', default: true }
                }
            }
        }
    },
    async (request: FastifyRequest<{ Querystring: { sample_size: number; stratify: boolean } }>, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'GET',
          path: '/analytics/feature-sample',
          query: request.query as Record<string, string | number | boolean>,
          requestId: request.requestId,
          authToken: request.headers.authorization,
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
