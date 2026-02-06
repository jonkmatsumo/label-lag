import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError, RequestOptions } from '../services/http-client.js';
import { ShadowService } from '../services/shadow.js';

export interface DatasetRoutesOptions {
  httpClient: HttpClient;
  shadowService: ShadowService;
}

export async function datasetRoutes(
  fastify: FastifyInstance,
  options: DatasetRoutesOptions
): Promise<void> {
  const { httpClient, shadowService } = options;

  // GET /bff/v1/dataset/overview
  fastify.get(
    '/bff/v1/dataset/overview',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'GET',
          path: '/analytics/overview',
          requestId: request.requestId,
          target: 'gateway',
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
          target: 'gateway',
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
            seed: { type: 'number' }, // Optional seed for determinism
          },
        },
      },
    },
    async (request: FastifyRequest<{ Body: { num_users: number; fraud_rate: number; drop_existing?: boolean; seed?: number } }>, reply: FastifyReply) => {
      try {
        const goRequest: RequestOptions = {
          method: 'POST',
          path: '/analytics/generate',
          body: request.body,
          requestId: request.requestId,
          timeout: 300000, // 5 min timeout for generation
          target: 'gateway',
        };

        const pythonRequest: RequestOptions = {
          method: 'POST',
          path: '/data/generate',
          body: request.body,
          requestId: request.requestId,
          timeout: 300000,
          target: 'python',
        };

        if (httpClient.config.enableGoDatasetGenerate) {
          const response = await shadowService.executeWithShadow(
            {
              primary: goRequest,
              shadow: pythonRequest,
            }
          );
          return reply.status(response.statusCode).send(response.data);
        } else {
          const response = await httpClient.request(pythonRequest);
          return reply.status(response.statusCode).send(response.data);
        }
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
        const goRequest: RequestOptions = {
          method: 'DELETE',
          path: '/data/clear',
          requestId: request.requestId,
          timeout: 60000,
          target: 'gateway',
        };

        const pythonRequest: RequestOptions = {
          method: 'DELETE',
          path: '/data/clear',
          requestId: request.requestId,
          timeout: 60000,
          target: 'python',
        };

        if (httpClient.config.enableGoDatasetClear) {
          const response = await shadowService.executeWithShadow(
            {
              primary: goRequest,
              shadow: pythonRequest,
            }
          );
          return reply.status(response.statusCode).send(response.data);
        } else {
          const response = await httpClient.request(pythonRequest);
          return reply.status(response.statusCode).send(response.data);
        }
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
          target: 'gateway',
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

  // GET /bff/v1/dataset/relationships
  fastify.get(
    '/bff/v1/dataset/relationships',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            sample_size: { type: 'integer', default: 500 },
            target_column: { type: 'string', default: 'is_fraudulent' }
          }
        }
      }
    },
    async (request: FastifyRequest<{ Querystring: { sample_size: number; target_column: string } }>, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'GET',
          path: '/analytics/relationships',
          query: request.query as Record<string, string | number | boolean>,
          requestId: request.requestId,
          target: 'python',
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

  // GET /bff/v1/dataset/correlations
  fastify.get(
    '/bff/v1/dataset/correlations',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            sample_size: { type: 'integer', default: 1000 },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: { sample_size: number } }>,
      reply: FastifyReply
    ) => {
      try {
        const response = await httpClient.request({
          method: 'GET',
          path: '/analytics/correlations',
          query: request.query as Record<string, string | number | boolean>,
          requestId: request.requestId,
          target: 'python',
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
