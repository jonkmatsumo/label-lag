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
import {
  resolveAnalyticsQueryInput,
} from '../utils/analytics-query-envelope.js';

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
  start_time?: string;
  end_time?: string;
  group_by?: 'hour' | 'day';
  granularity?: 'hour' | 'day';
  query?: string;
}

interface VolumeQuery {
  start_time?: string;
  end_time?: string;
  granularity?: 'hour' | 'day';
  query?: string;
}

interface ConfusionMatrixQuery {
  start_time?: string;
  end_time?: string;
  threshold?: number;
  model_version?: string;
  granularity?: 'hour' | 'day';
  query?: string;
}

interface RuleImpactQuery {
  start_time?: string;
  end_time?: string;
  granularity?: 'hour' | 'day';
  query?: string;
}

const HOT_ANALYTICS_CACHE_TTL_MS = 20_000;
const DOWNSAMPLE_FRIENDLY_MAX_WINDOW_DAYS = {
  day: 3650,
  hour: 3650,
} as const;

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

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
      try {
        const cacheKey = 'analytics:overview';
        const requestSignal = getRequestAbortSignal(request, reply);
        // @ts-ignore - tenantId is added by middleware
        const tenantId = request.tenantId || 'default';
        const normalized = await cache.getOrLoad<AnalyticsOverviewResponse>(
          cacheKey,
          tenantId,
          async () => {
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
            return response.data;
          },
          60000 // 1m TTL for overview
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

        const normalizedResponse = await cache.getOrLoad<DailyStatsResponse>(
          cacheKey,
          tenantId,
          async () => {
            const response = await httpClient.request<any>({
              method: 'GET',
              path: `/analytics/daily-stats?days=${days}`,
              requestId: request.requestId,
              tenantId: request.tenantId,
              target: 'gateway',
              signal: requestSignal,
            });

            const raw = response.data;
            return {
              stats: (raw.stats ?? []).map((s: any) => ({
                date: s.date,
                total_transactions: parseInt64(s.total_transactions) ?? 0,
                fraud_transactions: parseInt64(s.fraud_transactions) ?? 0,
                fraud_rate: s.fraud_rate ?? 0,
                total_amount: s.total_amount ?? 0,
              })),
              period_days: raw.period_days ?? days,
            };
          },
          HOT_ANALYTICS_CACHE_TTL_MS
        );

        return reply.send(normalizedResponse);
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
        const tenantId =
          (request as FastifyRequest & { tenantId?: string }).tenantId ??
          (typeof request.headers['x-tenant-id'] === 'string' ? request.headers['x-tenant-id'] : undefined) ??
          'default';
        const cacheKey = `analytics:recent-alerts:${tenantId}:${limit}`;

        const normalized = await cache.getOrLoad<RecentAlertsResponse>(
          cacheKey,
          tenantId,
          async () => {
            const response = await httpClient.request<any>({
              method: 'GET',
              path: `/analytics/recent-alerts?limit=${limit}`,
              requestId: request.requestId,
              tenantId: request.tenantId,
              target: 'gateway',
              signal: requestSignal,
            });

            const raw = response.data;
            return {
              alerts: (raw.alerts ?? []).map((a: any) => ({
                id: a.id,
                user_id: a.user_id,
                amount: a.amount ?? 0,
                score: a.score ?? 0,
                created_at: timestampToIso(a.created_at),
                matched_rules: a.matched_rules ?? [],
              })),
              total: parseInt64(raw.total) ?? 0,
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
            limit: { type: 'integer', minimum: 1, maximum: 10000, default: 100 },
            cursor: { type: 'string' },
            include_features: { type: 'boolean', default: false },
            query: {
              type: 'object',
              additionalProperties: false,
              properties: {
                start_time: { type: 'string' },
                end_time: { type: 'string' },
                granularity: { type: 'string', enum: ['hour', 'day'] },
              },
            },
          },
          additionalProperties: false,
        },
      },
    },
    async (
      request: FastifyRequest<{ Body: TransactionSearchRequest }>,
      reply: FastifyReply
    ) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const { query, ...restBody } = request.body;
        const validatedQuery = resolveAnalyticsQueryInput({
          query,
          legacy: {
            start_time: restBody.start_date,
            end_time: restBody.end_date,
          },
          options: {
            required: false,
            startField: 'start_date',
            endField: 'end_date',
          },
        });
        if (!validatedQuery.ok) {
          return reply.status(validatedQuery.statusCode).send(validatedQuery.body);
        }

        const payload = {
          ...restBody,
          include_features: restBody.include_features ?? false,
          ...(validatedQuery.value.start_time ? { start_date: validatedQuery.value.start_time } : {}),
          ...(validatedQuery.value.end_time ? { end_date: validatedQuery.value.end_time } : {}),
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
        const normalizedItems = (raw.transactions || []).map((tx: any) => ({
          id: tx.record_id,
          record_id: tx.record_id,
          user_id: tx.user_id,
          amount: toFiniteNumber(tx.amount),
          timestamp: timestampToIso(tx.created_at),
          created_at: timestampToIso(tx.created_at),
          is_fraud: tx.is_fraudulent,
          is_fraudulent: tx.is_fraudulent,
          fraud_type: tx.fraud_type,
          merchant_risk_score: toFiniteNumber(tx.merchant_risk_score),
          velocity_24h: toFiniteNumber(tx.velocity_24h),
          amount_to_avg_ratio_30d: toFiniteNumber(tx.amount_to_avg_ratio_30d),
          balance_volatility_z_score: toFiniteNumber(tx.balance_volatility_z_score),
          is_off_hours_txn: tx.is_off_hours_txn,
        }));
        const normalized: TransactionSearchResponse = {
          items: normalizedItems,
          next_cursor: raw.next_cursor ?? raw.nextCursor ?? '',
          truncated: !!(raw.truncated || raw.meta?.truncated),
          meta: normalizeAnalyticsMeta({
            raw,
            hasData: normalizedItems.length > 0,
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

  // GET /bff/v1/kpis - Get performance KPIs
  fastify.get<{ Querystring: KpiQuery }>(
    '/bff/v1/kpis',
    {
      schema: {
        querystring: {
          type: 'object',
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
            group_by: { type: 'string', enum: ['hour', 'day'] },
            granularity: { type: 'string', enum: ['hour', 'day'] },
            query: { type: 'string' },
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
        const { start_time, end_time, group_by, granularity, query } = request.query;
        if (granularity && group_by && granularity !== group_by) {
          return reply.status(400).send({
            error: {
              code: 'INVALID_RANGE',
              message: 'group_by and granularity must match when both are provided',
            },
          });
        }

        const validatedQuery = resolveAnalyticsQueryInput({
          query,
          legacy: {
            start_time,
            end_time,
            granularity: granularity ?? group_by,
          },
          options: {
            startField: 'start_time',
            endField: 'end_time',
            maxWindowDaysByGranularity: DOWNSAMPLE_FRIENDLY_MAX_WINDOW_DAYS,
          },
        });
        if (!validatedQuery.ok) {
          return reply.status(validatedQuery.statusCode).send(validatedQuery.body);
        }

        const queryEnvelope = validatedQuery.value;
        const groupBy = queryEnvelope.granularity;
        const tenantId = (request as FastifyRequest & { tenantId?: string }).tenantId ?? 'default';
        const cacheKey = `analytics:kpis:${tenantId}:${queryEnvelope.start_time}:${queryEnvelope.end_time}:${groupBy}`;
        const normalized = await cache.getOrLoad<KpisResponse>(
          cacheKey,
          tenantId,
          async () => {
            const response = await httpClient.request<any>({
              method: 'GET',
              path: `/kpis?start_time=${encodeURIComponent(queryEnvelope.start_time)}&end_time=${encodeURIComponent(queryEnvelope.end_time)}&group_by=${groupBy}`,
              requestId: request.requestId,
              tenantId: request.tenantId,
              target: 'gateway',
              signal: requestSignal,
            });

            const raw = response.data;
            return {
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
              meta: normalizeAnalyticsMeta({
                raw,
                hasData: (parseInt64(raw.total_decisions) ?? 0) > 0,
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

  // GET /bff/v1/volume - Get transaction volume timeseries
  fastify.get<{ Querystring: VolumeQuery }>(
    '/bff/v1/volume',
    {
      schema: {
        querystring: {
          type: 'object',
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
            granularity: { type: 'string', enum: ['hour', 'day'] },
            query: { type: 'string' },
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
            maxWindowDaysByGranularity: DOWNSAMPLE_FRIENDLY_MAX_WINDOW_DAYS,
          },
        });
        if (!validatedQuery.ok) {
          return reply.status(validatedQuery.statusCode).send(validatedQuery.body);
        }

        const queryEnvelope = validatedQuery.value;
        const tenantId = (request as FastifyRequest & { tenantId?: string }).tenantId ?? 'default';
        const cacheKey = `analytics:volume:${tenantId}:${queryEnvelope.start_time}:${queryEnvelope.end_time}:${queryEnvelope.granularity}`;
        const normalized = await cache.getOrLoad<VolumeSeriesResponse>(
          cacheKey,
          tenantId,
          async () => {
            const response = await httpClient.request<any>({
              method: 'GET',
              path: `/volume?start_time=${encodeURIComponent(queryEnvelope.start_time)}&end_time=${encodeURIComponent(queryEnvelope.end_time)}&granularity=${queryEnvelope.granularity}`,
              requestId: request.requestId,
              tenantId: request.tenantId,
              target: 'gateway',
              signal: requestSignal,
            });

            const raw = response.data;
            return {
              points: (raw.points ?? []).map((p: any) => ({
                timestamp: timestampToIso(p.timestamp),
                count: parseInt64(p.count) ?? 0,
                alerts: parseInt64(p.alerts) ?? 0,
              })),
              meta: normalizeAnalyticsMeta({
                raw,
                hasData: (raw.points?.length ?? 0) > 0,
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

  // GET /bff/v1/analytics/confusion-matrix - Get model confusion matrix
  fastify.get<{ Querystring: ConfusionMatrixQuery }>(
    '/bff/v1/analytics/confusion-matrix',
    {
      schema: {
        querystring: {
          type: 'object',
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
            granularity: { type: 'string', enum: ['hour', 'day'] },
            query: { type: 'string' },
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
        const { start_time, end_time, threshold, model_version, granularity, query } = request.query;
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
            maxWindowDaysByGranularity: DOWNSAMPLE_FRIENDLY_MAX_WINDOW_DAYS,
          },
        });
        if (!validatedQuery.ok) {
          return reply.status(validatedQuery.statusCode).send(validatedQuery.body);
        }

        const queryEnvelope = validatedQuery.value;
        const tenantId = (request as FastifyRequest & { tenantId?: string }).tenantId ?? 'default';
        const cacheKey = `analytics:confusion-matrix:${tenantId}:${queryEnvelope.start_time}:${queryEnvelope.end_time}:${threshold ?? ''}:${model_version ?? ''}`;
        const normalized = await cache.getOrLoad<ConfusionMatrixResponse>(
          cacheKey,
          tenantId,
          async () => {
            const searchParams = new URLSearchParams();
            searchParams.set('start_time', queryEnvelope.start_time);
            searchParams.set('end_time', queryEnvelope.end_time);
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
            return {
              true_positives: parseInt64(raw.true_positives) ?? 0,
              false_positives: parseInt64(raw.false_positives) ?? 0,
              true_negatives: parseInt64(raw.true_negatives) ?? 0,
              false_negatives: parseInt64(raw.false_negatives) ?? 0,
              precision: raw.precision ?? 0,
              recall: raw.recall ?? 0,
              f1_score: raw.f1_score ?? 0,
              insufficient_labels: !!raw.insufficient_labels,
              meta: normalizeAnalyticsMeta({
                raw,
                hasData: (parseInt64(raw.true_positives) ?? 0) + (parseInt64(raw.false_positives) ?? 0) + (parseInt64(raw.true_negatives) ?? 0) + (parseInt64(raw.false_negatives) ?? 0) > 0,
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
      request: FastifyRequest<{
        Params: RuleAnalyticsParams;
        Querystring: RuleImpactQuery;
      }>,
      reply: FastifyReply
    ) => {
      try {
        const requestSignal = getRequestAbortSignal(request, reply);
        const { rule_id } = request.params;
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

        const cacheKey = `analytics:rules:${rule_id}:impact:${tenantId}:${queryEnvelope.start_time}:${queryEnvelope.end_time}`;
        const searchParams = new URLSearchParams();
        searchParams.set('start_date', queryEnvelope.start_time);
        searchParams.set('end_date', queryEnvelope.end_time);

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
            const truncated = raw.truncated === true;
            return {
              rule_id: raw.rule_id,
              total_triggers: totalTriggers,
              avg_score_delta: Number(raw.avg_score_delta) || 0,
              daily_buckets: dailyBuckets,
              truncated,
              meta: normalizeAnalyticsMeta({
                raw,
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
