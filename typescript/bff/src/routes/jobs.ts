import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import { SimpleCache } from '../services/cache.js';
import type { GetJobSummaryResponse } from '../types/api.js';
import { parseInt64, timestampToIso } from '../utils/protojson.js';
import { getRequestAbortSignal } from '../utils/request-signal.js';
import { normalizeAnalyticsMeta } from '../utils/analytics-meta.js';
import { resolveAnalyticsQueryInput } from '../utils/analytics-query-envelope.js';

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
  start_time?: string;
  end_time?: string;
  granularity?: 'hour' | 'day';
  query?: string;
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
            limit: { type: 'integer', minimum: 1, maximum: 10000, default: 25 },
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

        if (
          typeof response.data !== 'object' ||
          response.data === null ||
          Array.isArray(response.data)
        ) {
          return reply.status(response.statusCode).send(response.data);
        }

        const raw = response.data as Record<string, unknown>;
        const jobs = Array.isArray(raw.jobs) ? raw.jobs : [];
        const normalized = {
          ...raw,
          meta: normalizeAnalyticsMeta({
            raw,
            hasData: jobs.length > 0,
          }),
        };

        return reply.status(response.statusCode).send(normalized);
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
            granularity: { type: 'string', enum: ['hour', 'day'] },
            query: { type: 'string' },
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
        const { start_time, end_time, granularity, query } = request.query;
        const validatedQuery = resolveAnalyticsQueryInput({
          query,
          legacy: {
            start_time,
            end_time,
            granularity,
          },
          options: {
            startField: 'start_time',
            endField: 'end_time',
            required: true,
          },
        });
        if (!validatedQuery.ok) {
          return reply.status(validatedQuery.statusCode).send(validatedQuery.body);
        }
        const queryEnvelope = validatedQuery.value;
        const tenantId =
          (request as FastifyRequest & { tenantId?: string }).tenantId ??
          (typeof request.headers['x-tenant-id'] === 'string' ? request.headers['x-tenant-id'] : undefined) ??
          'default';

        const cacheKey = `jobs:summary:${tenantId}:${queryEnvelope.start_time}:${queryEnvelope.end_time}`;
        const searchParams = new URLSearchParams();
        searchParams.set('start_time', queryEnvelope.start_time);
        searchParams.set('end_time', queryEnvelope.end_time);

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
