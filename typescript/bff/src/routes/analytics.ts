import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import { SimpleCache } from '../services/cache.js';
import { ShadowService } from '../services/shadow.js';
import type {
  AnalyticsOverviewResponse,
  DailyStatsResponse,
  TransactionDetailsResponse,
  TransactionSearchRequest,
  TransactionSearchResponse,
  RecentAlertsResponse,
  DatasetFingerprintResponse,
  FeatureSampleResponse,
  RuleAnalyticsResponse,
  RuleAttributionResponse,
  KpisResponse,
  VolumeSeriesResponse,
  ConfusionMatrixResponse,
  GetRuleImpactResponse,
} from '../types/api.js';
import { parseInt64, timestampToIso } from '../utils/protojson.js';
import { getRequestAbortSignal } from '../utils/request-signal.js';
import { normalizeAnalyticsMeta } from '../utils/analytics-meta.js';

export interface AnalyticsRoutesOptions {
  httpClient: HttpClient;
  cache: SimpleCache;
  shadowService: ShadowService;
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

interface KpiQuery {
  start_time: string;
  end_time: string;
  group_by?: 'hour' | 'day';
}

interface VolumeQuery {
  start_time: string;
  end_time: string;
  granularity?: 'hour' | 'day';
}

interface ConfusionMatrixQuery {
  start_time: string;
  end_time: string;
  threshold?: number;
  model_version?: string;
}

interface RuleImpactQuery {
  start_time: string;
  end_time: string;
}

const HOT_ANALYTICS_CACHE_TTL_MS = 20_000;

/**
 * Analytics routes for operational metrics and dataset insights
 */
export async function analyticsRoutes(
  fastify: FastifyInstance,
  options: AnalyticsRoutesOptions
): Promise<void> {
  const { httpClient, cache, shadowService } = options;

  // GET /bff/v1/analytics/overview - Get dataset overview metrics
  fastify.get(
    '/bff/v1/analytics/overview',
    async (request: FastifyRequest, reply: FastifyReply) => {
      const cacheKey = 'analytics:overview';
      const requestSignal = getRequestAbortSignal(request, reply);
      // @ts-ignore - tenantId is added by middleware
      const tenantId = request.tenantId || 'default';
      const cached = cache.get<AnalyticsOverviewResponse>(cacheKey, tenantId);
      if (cached) return reply.send(cached);

      try {
        // Shadow Mode: Gateway (Primary) vs Python (Shadow)
        const response = await shadowService.executeWithShadow<AnalyticsOverviewResponse>({
          primary: {
            method: 'GET',
            path: '/analytics/overview',
            requestId: request.requestId,
            tenantId: request.tenantId,
            target: 'gateway',
            signal: requestSignal,
          },
          shadow: {
            method: 'GET',
            path: '/analytics/overview',
            requestId: request.requestId,
            tenantId: request.tenantId,
            target: 'python',
            signal: requestSignal,
          },
        });

        cache.set(cacheKey, response.data, tenantId);
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { days = 30 } = request.query;
        const cacheKey = `analytics:daily-stats:${days}`;
        // @ts-ignore - tenantId is added by middleware
        const tenantId = request.tenantId || 'default';
        const cached = cache.get<DailyStatsResponse>(cacheKey, tenantId);
        if (cached) return reply.send(cached);

        const response = await httpClient.request<DailyStatsResponse>({
          method: 'GET',
          path: `/analytics/daily-stats?days=${days}`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          target: 'gateway',
          signal: requestSignal,
        });

        cache.set(cacheKey, response.data, tenantId);
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { days = 7, limit = 1000 } = request.query;

        const response = await httpClient.request<TransactionDetailsResponse>({
          method: 'GET',
          path: `/analytics/transactions?days=${days}&limit=${limit}`,
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { limit = 50 } = request.query;

        const response = await httpClient.request<RecentAlertsResponse>({
          method: 'GET',
          path: `/analytics/recent-alerts?limit=${limit}`,
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

  // GET /bff/v1/analytics/fingerprint - Get dataset fingerprint
  fastify.get(
    '/bff/v1/analytics/fingerprint',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const response = await httpClient.request<DatasetFingerprintResponse>({
          method: 'GET',
          path: '/analytics/fingerprint',
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { sample_size = 100, stratify = true } = request.query;

        const response = await httpClient.request<FeatureSampleResponse>({
          method: 'GET',
          path: `/analytics/feature-sample?sample_size=${sample_size}&stratify=${stratify}`,
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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { rule_id } = request.params;
        const { days = 7 } = request.query;

        const options: any = {
          method: 'GET',
          path: `/analytics/rules/${encodeURIComponent(rule_id)}?days=${days}`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          signal: requestSignal,
        };

        if (httpClient.config.enableGoRulesControlPlane) {
          options.target = 'gateway';
        } else {
          options.target = 'python';
        }

        const response = await httpClient.request<RuleAnalyticsResponse>(options);

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
        const requestSignal = getRequestAbortSignal(request, reply);
        const { rule_id, days = 7 } = request.query;

        const options: any = {
          method: 'GET',
          path: `/analytics/attribution?rule_id=${encodeURIComponent(rule_id)}&days=${days}`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          signal: requestSignal,
        };

        if (httpClient.config.enableGoRulesControlPlane) {
          options.target = 'gateway';
        } else {
          options.target = 'python';
        }

        const response = await httpClient.request<RuleAttributionResponse>(options);

        return reply.status(response.statusCode).send(response.data);
      } catch (error) {
        if (error instanceof UpstreamError) {
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );

  // POST /bff/v1/analytics/transactions/search
  fastify.post<{ Body: TransactionSearchRequest }>(
    '/bff/v1/analytics/transactions/search',
    {
      schema: {
        body: {
          type: 'object',
          properties: {
            user_id: { type: 'string' },
            transaction_id: { type: 'string' },
            min_amount: { type: 'number' },
            max_amount: { type: 'number' },
            start_date: { type: 'string' },
            end_date: { type: 'string' },
            is_fraudulent: { type: 'boolean' },
            min_score: { type: 'integer' },
            max_score: { type: 'integer' },
            limit: { type: 'integer', default: 100 },
            cursor: { type: 'string' },
            include_features: { type: 'boolean', default: false },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Body: TransactionSearchRequest }>,
      reply: FastifyReply
    ) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const payload = {
          ...request.body,
          include_features: request.body.include_features ?? false,
        };

        const response = await httpClient.request<any>({
          method: 'POST',
          path: '/analytics/transactions/search',
          body: payload,
          target: 'gateway',
          requestId: request.requestId,
          tenantId: request.tenantId,
          signal: requestSignal,
        });

        const raw = response.data;
        const normalized: TransactionSearchResponse = {
          items: raw.transactions || [],
          next_cursor: raw.next_cursor,
          truncated: raw.truncated ?? false,
          total: raw.total,
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

  // GET /bff/v1/kpis - Get performance KPIs
  fastify.get<{ Querystring: KpiQuery }>(
    '/bff/v1/kpis',
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
              description: 'ISO date or datetime string (YYYY-MM-DD or YYYY-MM-DDTHH:mm:ssZ)',
            },
            end_time: {
              type: 'string',
              pattern: '^\\d{4}-\\d{2}-\\d{2}',
              description: 'ISO date or datetime string (YYYY-MM-DD or YYYY-MM-DDTHH:mm:ssZ)',
            },
            group_by: { type: 'string', enum: ['hour', 'day'], default: 'day' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: KpiQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const { start_time, end_time, group_by = 'day' } = request.query;
        const tenantId = (request as FastifyRequest & { tenantId?: string }).tenantId ?? 'default';
        const cacheKey = `analytics:kpis:${tenantId}:${start_time}:${end_time}:${group_by}`;
        const cached = cache.get<KpisResponse>(cacheKey, tenantId);
        if (cached) return reply.send(cached);

        const response = await httpClient.request<any>({
          method: 'GET',
          path: `/kpis?start_time=${encodeURIComponent(start_time)}&end_time=${encodeURIComponent(end_time)}&group_by=${group_by}`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          target: 'gateway',
          signal: requestSignal,
        });

        // Normalize protojson int64 and timestamps
        const raw = response.data;
        const normalized: KpisResponse = {
          total_decisions: parseInt64(raw.total_decisions) ?? 0,
          total_alerts: parseInt64(raw.total_alerts) ?? 0,
          alert_rate: raw.alert_rate ?? 0,
          avg_score: raw.avg_score ?? 0,
          rules_fired_total: parseInt64(raw.rules_fired_total) ?? 0,
          buckets: raw.buckets?.map((b: any) => ({
            timestamp: timestampToIso(b.timestamp),
            decisions: parseInt64(b.decisions) ?? 0,
            alerts: parseInt64(b.alerts) ?? 0,
            rules_fired: parseInt64(b.rules_fired) ?? 0,
          })),
        };

        cache.set(cacheKey, normalized, tenantId, 30000); // 30s TTL
        return reply.status(response.statusCode).send(normalized);
      } catch (error) {
        if (error instanceof UpstreamError) {
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );

  // GET /bff/v1/volume - Get transaction volume timeseries
  fastify.get<{ Querystring: VolumeQuery }>(
    '/bff/v1/volume',
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
              description: 'ISO date or datetime string (YYYY-MM-DD or YYYY-MM-DDTHH:mm:ssZ)',
            },
            end_time: {
              type: 'string',
              pattern: '^\\d{4}-\\d{2}-\\d{2}',
              description: 'ISO date or datetime string (YYYY-MM-DD or YYYY-MM-DDTHH:mm:ssZ)',
            },
            granularity: { type: 'string', enum: ['hour', 'day'], default: 'day' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: VolumeQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const { start_time, end_time, granularity = 'day' } = request.query;
        const tenantId = (request as FastifyRequest & { tenantId?: string }).tenantId ?? 'default';
        const cacheKey = `analytics:volume:${tenantId}:${start_time}:${end_time}:${granularity}`;
        const cached = cache.get<VolumeSeriesResponse>(cacheKey, tenantId);
        if (cached) return reply.send(cached);

        const response = await httpClient.request<any>({
          method: 'GET',
          path: `/volume?start_time=${encodeURIComponent(start_time)}&end_time=${encodeURIComponent(end_time)}&granularity=${granularity}`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          target: 'gateway',
          signal: requestSignal,
        });

        // Normalize protojson int64 and timestamps
        const raw = response.data;
        const normalized: VolumeSeriesResponse = {
          points: (raw.points || []).map((p: any) => ({
            timestamp: timestampToIso(p.timestamp),
            count: parseInt64(p.count) ?? 0,
            alerts: parseInt64(p.alerts) ?? 0,
          })),
        };

        cache.set(cacheKey, normalized, tenantId, 30000); // 30s TTL
        return reply.status(response.statusCode).send(normalized);
      } catch (error) {
        if (error instanceof UpstreamError) {
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );

  // GET /bff/v1/analytics/confusion-matrix - Get model confusion matrix
  fastify.get<{ Querystring: ConfusionMatrixQuery }>(
    '/bff/v1/analytics/confusion-matrix',
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
              description: 'ISO date or datetime string (YYYY-MM-DD or YYYY-MM-DDTHH:mm:ssZ)',
            },
            end_time: {
              type: 'string',
              pattern: '^\\d{4}-\\d{2}-\\d{2}',
              description: 'ISO date or datetime string (YYYY-MM-DD or YYYY-MM-DDTHH:mm:ssZ)',
            },
            threshold: { type: 'number', minimum: 0, maximum: 100 },
            model_version: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: ConfusionMatrixQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const { start_time, end_time, threshold, model_version } = request.query;
        const tenantId = (request as FastifyRequest & { tenantId?: string }).tenantId ?? 'default';
        const cacheKey = `analytics:confusion-matrix:${tenantId}:${start_time}:${end_time}:${threshold ?? ''}:${model_version ?? ''}`;
        const cached = cache.get<ConfusionMatrixResponse>(cacheKey, tenantId);
        if (cached) return reply.send(cached);

        const searchParams = new URLSearchParams();
        searchParams.set('start_time', start_time);
        searchParams.set('end_time', end_time);
        if (threshold !== undefined) searchParams.set('threshold', String(threshold));
        if (model_version) searchParams.set('model_version', model_version);

        const response = await httpClient.request<any>({
          method: 'GET',
          path: `/analytics/confusion-matrix?${searchParams.toString()}`,
          requestId: request.requestId,
          tenantId: request.tenantId,
          target: 'gateway',
          signal: requestSignal,
        });

        // Normalize protojson int64 count fields to JS numbers
        const raw = response.data;
        const normalized: ConfusionMatrixResponse = {
          true_positives: parseInt64(raw.true_positives) ?? 0,
          false_positives: parseInt64(raw.false_positives) ?? 0,
          true_negatives: parseInt64(raw.true_negatives) ?? 0,
          false_negatives: parseInt64(raw.false_negatives) ?? 0,
          precision: raw.precision ?? 0,
          recall: raw.recall ?? 0,
          f1_score: raw.f1_score ?? 0,
          insufficient_labels: raw.insufficient_labels ?? false,
        };

        cache.set(cacheKey, normalized, tenantId, 30000); // 30s TTL
        return reply.status(response.statusCode).send(normalized);
      } catch (error) {
        if (error instanceof UpstreamError) {
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );

  // GET /bff/v1/analytics/rules/:rule_id/impact - Get impact metrics for a rule
  fastify.get(
    '/bff/v1/analytics/rules/:rule_id/impact',
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
      request: FastifyRequest<{
        Params: RuleAnalyticsParams;
        Querystring: RuleImpactQuery;
      }>,
      reply: FastifyReply
    ) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const { rule_id } = request.params;
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

        const cacheKey = `analytics:rules:${rule_id}:impact:${tenantId}:${start_time ?? ''}:${end_time ?? ''}`;
        const searchParams = new URLSearchParams();
        if (start_time) searchParams.set('start_date', start_time);
        if (end_time) searchParams.set('end_date', end_time);

        const normalized = await cache.getOrLoad<GetRuleImpactResponse>(
          cacheKey,
          tenantId,
          async () => {
            const response = await httpClient.request<any>({
              method: 'GET',
              path: `/analytics/rules/${encodeURIComponent(rule_id)}/impact?${searchParams.toString()}`,
              requestId: request.requestId,
              tenantId: request.tenantId,
              target: 'gateway',
              signal: requestSignal,
            });

            const raw = response.data;
            const dailyBuckets = (raw.daily_buckets ?? []).map((b: any) => ({
              date: b.date,
              trigger_count: parseInt64(b.trigger_count) ?? 0,
              avg_score_delta: Number(b.avg_score_delta) || 0,
              decisions_changed_count: parseInt64(b.decisions_changed_count) ?? 0,
            })).sort((a: any, b: any) => a.date.localeCompare(b.date));

            const totalTriggers = parseInt64(raw.total_triggers) ?? 0;
            return {
              rule_id: raw.rule_id,
              total_triggers: totalTriggers,
              avg_score_delta: Number(raw.avg_score_delta) || 0,
              daily_buckets: dailyBuckets,
              truncated: false,
              meta: normalizeAnalyticsMeta({
                raw,
                startTime: start_time,
                endTime: end_time,
                hasData: totalTriggers > 0 || dailyBuckets.length > 0,
              }),
            };
          },
          HOT_ANALYTICS_CACHE_TTL_MS
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
