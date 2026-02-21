import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import { SimpleCache } from '../services/cache.js';
import type { GetJobSummaryResponse } from '../types/api.js';
import { parseInt64, timestampToIso } from '../utils/protojson.js';
import { getRequestAbortSignal } from '../utils/request-signal.js';
import { normalizeAnalyticsMeta } from '../utils/analytics-meta.js';

export interface JobsRoutesOptions {
  httpClient: HttpClient;
  cache: SimpleCache;
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

interface JobSummaryQuery {
  start_time: string;
  end_time: string;
}

const HOT_JOBS_CACHE_TTL_MS = 20_000;

/**
 * Jobs management routes — proxied to orchestrator
 */
export async function jobsRoutes(
  fastify: FastifyInstance,
  options: JobsRoutesOptions
): Promise<void> {
  const { httpClient, cache } = options;

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
        const requestSignal = getRequestAbortSignal(request, reply);
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
          signal: requestSignal,
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { id } = request.params;
        const response = await httpClient.request({
          method: 'GET',
          path: `/jobs/${encodeURIComponent(id)}`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          target: 'gateway',
          signal: requestSignal,
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { id } = request.params;
        const response = await httpClient.request({
          method: 'GET',
          path: `/jobs/${encodeURIComponent(id)}/events`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          target: 'gateway',
          signal: requestSignal,
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { id } = request.params;
        const response = await httpClient.request({
          method: 'POST',
          path: `/jobs/${encodeURIComponent(id)}/cancel`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          target: 'gateway',
          signal: requestSignal,
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { id } = request.params;
        const response = await httpClient.request({
          method: 'POST',
          path: `/jobs/${encodeURIComponent(id)}/retry`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          target: 'gateway',
          signal: requestSignal,
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

  // GET /bff/v1/jobs/summary - Get jobs summary (aggregated by hour)
  fastify.get(
    '/bff/v1/jobs/summary',
    {
      schema: {
        querystring: {
          type: 'object',
          required: ['start_time', 'end_time'],
          additionalProperties: false,
          properties: {
            start_time: {
              type: 'string',
              pattern: '^\\d{4}-\\d{2}-\\d{2}',
              description: 'ISO date or datetime string (YYYY-MM-DD)',
            },
            end_time: {
              type: 'string',
              pattern: '^\\d{4}-\\d{2}-\\d{2}',
              description: 'ISO date or datetime string (YYYY-MM-DD)',
            },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: JobSummaryQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const { start_time, end_time } = request.query;
        const tenantId =
          (request as FastifyRequest & { tenantId?: string }).tenantId ??
          (typeof request.headers['x-tenant-id'] === 'string' ? request.headers['x-tenant-id'] : undefined) ??
          'default';

        // Validation
        const start = new Date(start_time);
        const end = new Date(end_time);
        if (start >= end) {
          return reply.status(400).send({
            error: {
              code: 'INVALID_RANGE',
              message: 'start_time must be before end_time',
            }
          });
        }
        const daysDiff = (end.getTime() - start.getTime()) / (1000 * 3600 * 24);
        if (daysDiff > 90) {
          return reply.status(400).send({
            error: {
              code: 'INVALID_RANGE',
              message: 'Time range cannot exceed 90 days',
            }
          });
        }

        const cacheKey = `jobs:summary:${tenantId}:${start_time ?? ''}:${end_time ?? ''}`;
        const searchParams = new URLSearchParams();
        if (start_time) searchParams.set('start_time', start_time);
        if (end_time) searchParams.set('end_time', end_time);

        const normalized = await cache.getOrLoad<GetJobSummaryResponse>(
          cacheKey,
          tenantId,
          async () => {
            const response = await httpClient.request<any>({
              method: 'GET',
              path: `/jobs/summary?${searchParams.toString()}`,
              requestId: request.requestId,
              tenantId: request.tenantId,
              target: 'gateway',
              signal: requestSignal,
            });

            const raw = response.data;
            const summaries = (raw.summaries ?? []).map((s: any) => ({
              bucket_time: timestampToIso(s.bucket_time),
              total_jobs: parseInt64(s.total_jobs) ?? 0,
              completed_jobs: parseInt64(s.completed_jobs) ?? 0,
              failed_jobs: parseInt64(s.failed_jobs) ?? 0,
            }));

            return {
              summaries,
              meta: normalizeAnalyticsMeta({
                raw,
                startTime: start_time,
                endTime: end_time,
                hasData: summaries.length > 0,
              }),
            };
          },
          HOT_JOBS_CACHE_TTL_MS
        );

        return reply.send(normalized);
      } catch (error) {
        if (error instanceof UpstreamError) {
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );
}
