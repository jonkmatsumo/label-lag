import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import { SimpleCache } from '../services/cache.js';
import type {
  AnalyticsOverviewResponse,
  DailyStatsResponse,
  TransactionDetailsResponse,
  RecentAlertsResponse,
  DatasetFingerprintResponse,
  FeatureSampleResponse,
  RuleAnalyticsResponse,
  RuleAttributionResponse,
} from '../types/api.js';

export interface AnalyticsRoutesOptions {
  httpClient: HttpClient;
  cache: SimpleCache;
}

interface DailyStatsQuery {
  days?: number;
}

interface TransactionDetailsQuery {
  days?: number;
  limit?: number;
}

interface RecentAlertsQuery {
  limit?: number;
}

interface FeatureSampleQuery {
  sample_size?: number;
  stratify?: boolean;
}

interface RuleAnalyticsParams {
  rule_id: string;
}

interface RuleAnalyticsQuery {
  days?: number;
}

interface RuleAttributionQuery {
  rule_id: string;
  days?: number;
}

/**
 * Analytics routes for operational metrics and dataset insights
 */
export async function analyticsRoutes(
  fastify: FastifyInstance,
  options: AnalyticsRoutesOptions
): Promise<void> {
  const { httpClient, cache } = options;

  // GET /bff/v1/analytics/overview - Get dataset overview metrics
  fastify.get(
    '/bff/v1/analytics/overview',
    async (request: FastifyRequest, reply: FastifyReply) => {
      const cacheKey = 'analytics:overview';
      const cached = cache.get<AnalyticsOverviewResponse>(cacheKey);
      if (cached) return reply.send(cached);

      try {
        const response = await httpClient.request<AnalyticsOverviewResponse>({
          method: 'GET',
          path: '/analytics/overview',
          requestId: request.requestId,
        });

        cache.set(cacheKey, response.data);
        return reply.status(response.statusCode).send(response.data);
      } catch (error) {
        if (error instanceof UpstreamError) {
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );

  // GET /bff/v1/analytics/daily-stats - Get daily transaction statistics
  fastify.get<{ Querystring: DailyStatsQuery }>(
    '/bff/v1/analytics/daily-stats',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            days: { type: 'integer', minimum: 1, maximum: 90, default: 30 },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: DailyStatsQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { days = 30 } = request.query;
        const cacheKey = `analytics:daily-stats:${days}`;
        const cached = cache.get<DailyStatsResponse>(cacheKey);
        if (cached) return reply.send(cached);

        const response = await httpClient.request<DailyStatsResponse>({
          method: 'GET',
          path: `/analytics/daily-stats?days=${days}`,
          requestId: request.requestId,
        });

        cache.set(cacheKey, response.data);
        return reply.status(response.statusCode).send(response.data);
      } catch (error) {
        if (error instanceof UpstreamError) {
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );

  // GET /bff/v1/analytics/transactions - Get transaction details
  fastify.get<{ Querystring: TransactionDetailsQuery }>(
    '/bff/v1/analytics/transactions',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            days: { type: 'integer', minimum: 1, maximum: 30, default: 7 },
            limit: { type: 'integer', minimum: 1, maximum: 5000, default: 1000 },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: TransactionDetailsQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { days = 7, limit = 1000 } = request.query;

        const response = await httpClient.request<TransactionDetailsResponse>({
          method: 'GET',
          path: `/analytics/transactions?days=${days}&limit=${limit}`,
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

  // GET /bff/v1/analytics/recent-alerts - Get recent high-risk alerts
  fastify.get<{ Querystring: RecentAlertsQuery }>(
    '/bff/v1/analytics/recent-alerts',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            limit: { type: 'integer', minimum: 1, maximum: 200, default: 50 },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: RecentAlertsQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { limit = 50 } = request.query;

        const response = await httpClient.request<RecentAlertsResponse>({
          method: 'GET',
          path: `/analytics/recent-alerts?limit=${limit}`,
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

  // GET /bff/v1/analytics/fingerprint - Get dataset fingerprint
  fastify.get(
    '/bff/v1/analytics/fingerprint',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request<DatasetFingerprintResponse>({
          method: 'GET',
          path: '/analytics/fingerprint',
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

  // GET /bff/v1/analytics/feature-sample - Get sampled features for diagnostics
  fastify.get<{ Querystring: FeatureSampleQuery }>(
    '/bff/v1/analytics/feature-sample',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            sample_size: { type: 'integer', minimum: 1, maximum: 1000, default: 100 },
            stratify: { type: 'boolean', default: true },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: FeatureSampleQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { sample_size = 100, stratify = true } = request.query;

        const response = await httpClient.request<FeatureSampleResponse>({
          method: 'GET',
          path: `/analytics/feature-sample?sample_size=${sample_size}&stratify=${stratify}`,
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

  // GET /bff/v1/analytics/rules/:rule_id - Get rule health & stats
  fastify.get<{ Params: RuleAnalyticsParams; Querystring: RuleAnalyticsQuery }>(
    '/bff/v1/analytics/rules/:rule_id',
    {
      schema: {
        params: {
          type: 'object',
          required: ['rule_id'],
          properties: {
            rule_id: { type: 'string' },
          },
        },
        querystring: {
          type: 'object',
          properties: {
            days: { type: 'integer', minimum: 1, maximum: 90, default: 7 },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: RuleAnalyticsParams; Querystring: RuleAnalyticsQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { rule_id } = request.params;
        const { days = 7 } = request.query;

        const response = await httpClient.request<RuleAnalyticsResponse>({
          method: 'GET',
          path: `/analytics/rules/${encodeURIComponent(rule_id)}?days=${days}`,
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

  // GET /bff/v1/analytics/attribution - Get rule attribution metrics
  fastify.get<{ Querystring: RuleAttributionQuery }>(
    '/bff/v1/analytics/attribution',
    {
      schema: {
        querystring: {
          type: 'object',
          required: ['rule_id'],
          properties: {
            rule_id: { type: 'string' },
            days: { type: 'integer', minimum: 1, maximum: 90, default: 7 },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: RuleAttributionQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { rule_id, days = 7 } = request.query;

        const response = await httpClient.request<RuleAttributionResponse>({
          method: 'GET',
          path: `/analytics/attribution?rule_id=${encodeURIComponent(rule_id)}&days=${days}`,
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
