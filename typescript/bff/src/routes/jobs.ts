import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';

export interface JobsRoutesOptions {
  httpClient: HttpClient;
}

interface ListJobsQuery {
  job_type?: string;
  status?: string;
  limit?: number;
  cursor?: string;
}

interface JobIdParams {
  id: string;
}

/**
 * Jobs management routes — proxied to orchestrator
 */
export async function jobsRoutes(
  fastify: FastifyInstance,
  options: JobsRoutesOptions
): Promise<void> {
  const { httpClient } = options;

  // GET /bff/v1/jobs - List jobs with cursor pagination
  fastify.get<{ Querystring: ListJobsQuery }>(
    '/bff/v1/jobs',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            job_type: { type: 'string' },
            status: { type: 'string' },
            limit: { type: 'integer', minimum: 1, maximum: 100, default: 25 },
            cursor: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: ListJobsQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { job_type, status, limit = 25, cursor } = request.query;
        const query: Record<string, string | number | undefined> = { limit };
        if (job_type) query.job_type = job_type;
        if (status) query.status = status;
        if (cursor) query.cursor = cursor;

        const response = await httpClient.request({
          method: 'GET',
          path: '/jobs',
          query,
          requestId: request.requestId,
          tenantId: request.tenantId,
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

  // GET /bff/v1/jobs/:id - Get job detail
  fastify.get<{ Params: JobIdParams }>(
    '/bff/v1/jobs/:id',
    {
      schema: {
        params: {
          type: 'object',
          required: ['id'],
          properties: { id: { type: 'string' } },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: JobIdParams }>,
      reply: FastifyReply
    ) => {
      try {
        const { id } = request.params;
        const response = await httpClient.request({
          method: 'GET',
          path: `/jobs/${encodeURIComponent(id)}`,
          requestId: request.requestId,
          tenantId: request.tenantId,
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

  // GET /bff/v1/jobs/:id/events - Get job events
  fastify.get<{ Params: JobIdParams }>(
    '/bff/v1/jobs/:id/events',
    {
      schema: {
        params: {
          type: 'object',
          required: ['id'],
          properties: { id: { type: 'string' } },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: JobIdParams }>,
      reply: FastifyReply
    ) => {
      try {
        const { id } = request.params;
        const response = await httpClient.request({
          method: 'GET',
          path: `/jobs/${encodeURIComponent(id)}/events`,
          requestId: request.requestId,
          tenantId: request.tenantId,
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

  // POST /bff/v1/jobs/:id/cancel - Cancel a job
  fastify.post<{ Params: JobIdParams }>(
    '/bff/v1/jobs/:id/cancel',
    {
      schema: {
        params: {
          type: 'object',
          required: ['id'],
          properties: { id: { type: 'string' } },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: JobIdParams }>,
      reply: FastifyReply
    ) => {
      try {
        const { id } = request.params;
        const response = await httpClient.request({
          method: 'POST',
          path: `/jobs/${encodeURIComponent(id)}/cancel`,
          requestId: request.requestId,
          tenantId: request.tenantId,
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

  // POST /bff/v1/jobs/:id/retry - Retry a failed job
  fastify.post<{ Params: JobIdParams }>(
    '/bff/v1/jobs/:id/retry',
    {
      schema: {
        params: {
          type: 'object',
          required: ['id'],
          properties: { id: { type: 'string' } },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: JobIdParams }>,
      reply: FastifyReply
    ) => {
      try {
        const { id } = request.params;
        const response = await httpClient.request({
          method: 'POST',
          path: `/jobs/${encodeURIComponent(id)}/retry`,
          requestId: request.requestId,
          tenantId: request.tenantId,
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
}
