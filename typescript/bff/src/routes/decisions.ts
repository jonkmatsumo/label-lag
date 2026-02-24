import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import { normalizeAnalyticsMeta } from '../utils/analytics-meta.js';

export interface DecisionsRoutesOptions {
    httpClient: HttpClient;
}

interface ListDecisionsQuery {
    user_id?: string;
    decision?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
    cursor?: string;
}

interface DecisionIdParams {
    id: string;
}

/**
 * Decisions management routes — proxied to orchestrator
 */
export async function decisionsRoutes(
    fastify: FastifyInstance,
    options: DecisionsRoutesOptions
): Promise<void> {
    const { httpClient } = options;

    // GET /bff/v1/decisions - List decisions with cursor pagination
    fastify.get<{ Querystring: ListDecisionsQuery }>(
        '/bff/v1/decisions',
        {
            schema: {
                querystring: {
                    type: 'object',
                    properties: {
                        user_id: { type: 'string' },
                        decision: { type: 'string' },
                        start_time: { type: 'string', format: 'date-time' },
                        end_time: { type: 'string', format: 'date-time' },
                        limit: { type: 'integer', minimum: 1, maximum: 10000, default: 25 },
                        cursor: { type: 'string' },
                    },
                },
            },
        },
        async (
            request: FastifyRequest<{ Querystring: ListDecisionsQuery }>,
            reply: FastifyReply
        ) => {
            try {
                const { user_id, decision, start_time, end_time, limit = 25, cursor } = request.query;
                const query: Record<string, string | number | undefined> = { limit };
                if (user_id) query.user_id = user_id;
                if (decision) query.decision = decision;
                if (start_time) query.start_time = start_time;
                if (end_time) query.end_time = end_time;
                if (cursor) query.cursor = cursor;

                const response = await httpClient.request({
                    method: 'GET',
                    path: '/decisions',
                    query,
                    requestId: request.requestId,
                    tenantId: request.tenantId,
                    target: 'gateway',
                });

                if (
                    typeof response.data !== 'object' ||
                    response.data === null ||
                    Array.isArray(response.data)
                ) {
                    return reply.status(response.statusCode).send(response.data);
                }

                const raw = response.data as Record<string, unknown>;
                const decisions = Array.isArray(raw.decisions) ? raw.decisions : [];
                const normalized = {
                    ...raw,
                    meta: normalizeAnalyticsMeta({
                        raw,
                        hasData: decisions.length > 0,
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

    // GET /bff/v1/decisions/:id - Get decision detail
    fastify.get<{ Params: DecisionIdParams }>(
        '/bff/v1/decisions/:id',
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
            request: FastifyRequest<{ Params: DecisionIdParams }>,
            reply: FastifyReply
        ) => {
            try {
                const { id } = request.params;
                const response = await httpClient.request({
                    method: 'GET',
                    path: `/decisions/${encodeURIComponent(id)}`,
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

    // GET /bff/v1/decisions/:id/trace - Get decision trace
    fastify.get<{ Params: DecisionIdParams }>(
        '/bff/v1/decisions/:id/trace',
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
            request: FastifyRequest<{ Params: DecisionIdParams }>,
            reply: FastifyReply
        ) => {
            try {
                const { id } = request.params;
                const response = await httpClient.request({
                    method: 'GET',
                    path: `/decisions/${encodeURIComponent(id)}/trace`,
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
