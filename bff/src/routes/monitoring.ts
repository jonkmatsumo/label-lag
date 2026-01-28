import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import type {
  DriftStatusResponse,
  ShadowComparisonResponse,
  BacktestResultsListResponse,
} from '../types/api.js';

export interface MonitoringRoutesOptions {
  httpClient: HttpClient;
}

interface DriftQuery {
  hours?: number;
  threshold?: number;
  force_refresh?: boolean;
}

interface ShadowComparisonQuery {
  start_date: string;
  end_date: string;
  rule_ids?: string;
}

interface BacktestResultsQuery {
  rule_id?: string;
  start_date?: string;
  end_date?: string;
  limit?: number;
}

/**
 * Monitoring routes for drift detection and shadow metrics
 */
export async function monitoringRoutes(
  fastify: FastifyInstance,
  options: MonitoringRoutesOptions
): Promise<void> {
  const { httpClient } = options;

  // GET /bff/v1/monitoring/drift - Check dataset drift status
  fastify.get<{ Querystring: DriftQuery }>(
    '/bff/v1/monitoring/drift',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            hours: { type: 'integer', minimum: 1, maximum: 168, default: 24 },
            threshold: { type: 'number', minimum: 0.01, maximum: 1.0, default: 0.25 },
            force_refresh: { type: 'boolean', default: false },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: DriftQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { hours = 24, threshold = 0.25, force_refresh = false } = request.query;

        const queryParams = new URLSearchParams({
          hours: String(hours),
          threshold: String(threshold),
          force_refresh: String(force_refresh),
        });

        const response = await httpClient.request<DriftStatusResponse>({
          method: 'GET',
          path: `/monitoring/drift?${queryParams.toString()}`,
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

  // GET /bff/v1/metrics/shadow/comparison - Get shadow mode comparison metrics
  fastify.get<{ Querystring: ShadowComparisonQuery }>(
    '/bff/v1/metrics/shadow/comparison',
    {
      schema: {
        querystring: {
          type: 'object',
          required: ['start_date', 'end_date'],
          properties: {
            start_date: { type: 'string', format: 'date' },
            end_date: { type: 'string', format: 'date' },
            rule_ids: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: ShadowComparisonQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { start_date, end_date, rule_ids } = request.query;

        const queryParams = new URLSearchParams({
          start_date,
          end_date,
        });

        if (rule_ids) {
          queryParams.set('rule_ids', rule_ids);
        }

        const response = await httpClient.request<ShadowComparisonResponse>({
          method: 'GET',
          path: `/metrics/shadow/comparison?${queryParams.toString()}`,
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

  // GET /bff/v1/backtest/results - List backtest results
  fastify.get<{ Querystring: BacktestResultsQuery }>(
    '/bff/v1/backtest/results',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            rule_id: { type: 'string' },
            start_date: { type: 'string', format: 'date' },
            end_date: { type: 'string', format: 'date' },
            limit: { type: 'integer', minimum: 1, maximum: 100, default: 50 },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Querystring: BacktestResultsQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { rule_id, start_date, end_date, limit = 50 } = request.query;

        const queryParams = new URLSearchParams({ limit: String(limit) });

        if (rule_id) {
          queryParams.set('rule_id', rule_id);
        }
        if (start_date) {
          queryParams.set('start_date', start_date);
        }
        if (end_date) {
          queryParams.set('end_date', end_date);
        }

        const response = await httpClient.request<BacktestResultsListResponse>({
          method: 'GET',
          path: `/backtest/results?${queryParams.toString()}`,
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
