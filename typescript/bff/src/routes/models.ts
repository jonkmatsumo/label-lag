import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';

export interface ModelsRoutesOptions {
    httpClient: HttpClient;
}

interface ListModelsQuery {
    model_name?: string;
    limit?: number;
    cursor?: string;
}

/**
 * Model versions management routes — proxied to orchestrator
 */
export async function modelsRoutes(
    fastify: FastifyInstance,
    options: ModelsRoutesOptions
): Promise<void> {
    const { httpClient } = options;

    // GET /bff/v1/models/versions - List model versions
    fastify.get<{ Querystring: ListModelsQuery }>(
        '/bff/v1/models/versions',
        {
            schema: {
                querystring: {
                    type: 'object',
                    properties: {
                        model_name: { type: 'string' },
                        limit: { type: 'integer', minimum: 1, maximum: 100, default: 25 },
                        cursor: { type: 'string' },
                    },
                },
            },
        },
        async (
            request: FastifyRequest<{ Querystring: ListModelsQuery }>,
            reply: FastifyReply
        ) => {
            try {
                const { model_name, limit = 25, cursor } = request.query;
                const query: Record<string, string | number | undefined> = { limit };
                if (model_name) query.model_name = model_name;
                if (cursor) query.cursor = cursor;

                const response = await httpClient.request({
                    method: 'GET',
                    path: '/models/versions',
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

    // GET /bff/v1/models/versions/:version - Get model version details
    fastify.get<{ Params: { version: string } }>(
        '/bff/v1/models/versions/:version',
        {
            schema: {
                params: {
                    type: 'object',
                    required: ['version'],
                    properties: {
                        version: { type: 'string' },
                    },
                },
            },
        },
        async (
            request: FastifyRequest<{ Params: { version: string } }>,
            reply: FastifyReply
        ) => {
            try {
                const { version } = request.params;
                const response = await httpClient.request({
                    method: 'GET',
                    path: `/models/versions/${encodeURIComponent(version)}`,
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
