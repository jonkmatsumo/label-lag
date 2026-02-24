import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import { normalizeAnalyticsMeta } from '../utils/analytics-meta.js';

export interface TrainingRoutesOptions {
    httpClient: HttpClient;
}

interface ListTrainingRunsQuery {
    model_name?: string;
    status?: string;
    limit?: number;
    cursor?: string;
}

interface TrainingRunIdParams {
    id: string;
}

/**
 * Training runs management routes — proxied to orchestrator
 */
export async function trainingRoutes(
    fastify: FastifyInstance,
    options: TrainingRoutesOptions
): Promise<void> {
    const { httpClient } = options;

    // GET /bff/v1/training-runs - List training runs
    fastify.get<{ Querystring: ListTrainingRunsQuery }>(
        '/bff/v1/training-runs',
        {
            schema: {
                querystring: {
                    type: 'object',
                    properties: {
                        model_name: { type: 'string' },
                        status: { type: 'string' },
                        limit: { type: 'integer', minimum: 1, maximum: 10000, default: 25 },
                        cursor: { type: 'string' },
                    },
                },
            },
        },
        async (
            request: FastifyRequest<{ Querystring: ListTrainingRunsQuery }>,
            reply: FastifyReply
        ) => {
            try {
                const { model_name, status, limit = 25, cursor } = request.query;
                const query: Record<string, string | number | undefined> = { limit };
                if (model_name) query.model_name = model_name;
                if (status) query.status = status;
                if (cursor) query.cursor = cursor;

                const response = await httpClient.request({
                    method: 'GET',
                    path: '/training-runs',
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
                const runs = Array.isArray(raw.runs) ? raw.runs : [];
                const normalized = {
                    ...raw,
                    meta: normalizeAnalyticsMeta({
                        raw,
                        hasData: runs.length > 0,
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

    // GET /bff/v1/training-runs/:id - Get training run detail
    fastify.get<{ Params: TrainingRunIdParams }>(
        '/bff/v1/training-runs/:id',
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
            request: FastifyRequest<{ Params: TrainingRunIdParams }>,
            reply: FastifyReply
        ) => {
            try {
                const { id } = request.params;
                const response = await httpClient.request({
                    method: 'GET',
                    path: `/training-runs/${encodeURIComponent(id)}`,
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
