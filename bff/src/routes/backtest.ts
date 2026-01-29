import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import type { BacktestCompareRequest, BacktestCompareResponse, BacktestResultsListResponse } from '../types/api.js';

export interface BacktestRoutesOptions {
  httpClient: HttpClient;
}

interface BacktestCompareBody {
  base_version: string;
  candidate_version: string;
  start_date: string;
  end_date: string;
  rule_id?: string;
}

interface BacktestResultsQuery {
  rule_id?: string;
  start_date?: string;
  end_date?: string;
  limit?: number;
}

/**
 * Backtest / What-if simulation routes
 */
export async function backtestRoutes(
  fastify: FastifyInstance,
  options: BacktestRoutesOptions
): Promise<void> {
  const { httpClient } = options;

  // GET /bff/v1/backtest/results - List backtest results
  fastify.get<{ Querystring: BacktestResultsQuery }>(
    '/bff/v1/backtest/results',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            rule_id: { type: 'string' },
            start_date: { type: 'string' },
            end_date: { type: 'string' },
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
        
        const query: Record<string, string | number | undefined> = { limit };
        if (rule_id) query.rule_id = rule_id;
        if (start_date) query.start_date = start_date;
        if (end_date) query.end_date = end_date;

        const response = await httpClient.request<BacktestResultsListResponse>({
          method: 'GET',
          path: '/backtest/results',
          query,
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

  // POST /bff/v1/backtest/compare - Compare two backtest rulesets (what-if)
  fastify.post<{ Body: BacktestCompareBody }>(
    '/bff/v1/backtest/compare',
    {
      schema: {
        body: {
          type: 'object',
          required: ['base_version', 'candidate_version', 'start_date', 'end_date'],
          properties: {
            base_version: { type: 'string' },
            candidate_version: { type: 'string' },
            start_date: { type: 'string' },
            end_date: { type: 'string' },
            rule_id: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Body: BacktestCompareBody }>,
      reply: FastifyReply
    ) => {
      try {
        const compareRequest: BacktestCompareRequest = request.body;

        const response = await httpClient.request<BacktestCompareResponse>({
          method: 'POST',
          path: '/backtest/compare',
          body: compareRequest,
          requestId: request.requestId,
          timeout: 120000, // 2 minutes for backtest
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
