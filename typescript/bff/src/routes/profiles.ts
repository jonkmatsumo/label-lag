import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';

export interface ProfilesRoutesOptions {
    httpClient: HttpClient;
}

interface ListProfilesQuery {
    limit?: number;
    cursor?: string;
}

interface ProfileIdParams {
    id: string;
}

interface ProfileSummaryQuery {
    profile_id?: string;
}

interface CompareProfilesQuery {
    base_id: string;
    target_id: string;
}

/**
 * Dataset profiles management routes — proxied to orchestrator
 */
export async function profilesRoutes(
    fastify: FastifyInstance,
    options: ProfilesRoutesOptions
): Promise<void> {
    const { httpClient } = options;

    // GET /bff/v1/dataset/profiles - List dataset profiles
    fastify.get<{ Querystring: ListProfilesQuery }>(
        '/bff/v1/dataset/profiles',
        {
            schema: {
                querystring: {
                    type: 'object',
                    properties: {
                        limit: { type: 'integer', minimum: 1, maximum: 100, default: 25 },
                        cursor: { type: 'string' },
                    },
                },
            },
        },
        async (
            request: FastifyRequest<{ Querystring: ListProfilesQuery }>,
            reply: FastifyReply
        ) => {
            try {
                const { limit = 25, cursor } = request.query;
                const query: Record<string, string | number | undefined> = { limit };
                if (cursor) query.cursor = cursor;

                const response = await httpClient.request({
                    method: 'GET',
                    path: '/dataset/profiles',
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

    // GET /bff/v1/dataset/profiles/:id - Get dataset profile detail
    fastify.get<{ Params: ProfileIdParams }>(
        '/bff/v1/dataset/profiles/:id',
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
            request: FastifyRequest<{ Params: ProfileIdParams }>,
            reply: FastifyReply
        ) => {
            try {
                const { id } = request.params;
                const response = await httpClient.request({
                    method: 'GET',
                    path: `/dataset/profiles/${encodeURIComponent(id)}`,
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

    // GET /bff/v1/dataset/summary - Get dataset summary (latest or specific profile)
    fastify.get<{ Querystring: ProfileSummaryQuery }>(
        '/bff/v1/dataset/summary',
        {
            schema: {
                querystring: {
                    type: 'object',
                    properties: {
                        profile_id: { type: 'string' },
                    },
                },
            },
        },
        async (
            request: FastifyRequest<{ Querystring: ProfileSummaryQuery }>,
            reply: FastifyReply
        ) => {
            try {
                const { profile_id } = request.query;
                const query: Record<string, string | undefined> = {};
                if (profile_id) query.profile_id = profile_id;

                const response = await httpClient.request({
                    method: 'GET',
                    path: '/dataset/summary',
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

    // GET /bff/v1/dataset/profiles/compare - Compare profiles
    fastify.get<{ Querystring: CompareProfilesQuery }>(
        '/bff/v1/dataset/profiles/compare',
        {
            schema: {
                querystring: {
                    type: 'object',
                    required: ['base_id', 'target_id'],
                    properties: {
                        base_id: { type: 'string' },
                        target_id: { type: 'string' },
                    },
                },
            },
        },
        async (
            request: FastifyRequest<{ Querystring: CompareProfilesQuery }>,
            reply: FastifyReply
        ) => {
            try {
                const { base_id, target_id } = request.query;
                const query = { base_id, target_id };

                const response = await httpClient.request({
                    method: 'GET',
                    path: '/dataset/profiles/compare',
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
}
