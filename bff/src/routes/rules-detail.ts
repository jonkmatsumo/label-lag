import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import type {
  ReadinessReportResponse,
  RuleVersionListResponse,
  RuleVersionResponse,
  RuleDiffResponse,
  ProductionRulesResponse,
} from '../types/api.js';

export interface RulesDetailRoutesOptions {
  httpClient: HttpClient;
}

interface RuleIdParams {
  rule_id: string;
}

interface VersionParams {
  rule_id: string;
  version_id: string;
}

interface RuleDiffQuery {
  version_a?: string;
  version_b?: string;
}

/**
 * Rules detail routes for readiness, versions, and diffs
 */
export async function rulesDetailRoutes(
  fastify: FastifyInstance,
  options: RulesDetailRoutesOptions
): Promise<void> {
  const { httpClient } = options;

  // GET /bff/v1/rules - Get production ruleset
  fastify.get(
    '/bff/v1/rules',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request<ProductionRulesResponse>({
          method: 'GET',
          path: '/rules',
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

  // GET /bff/v1/rules/:rule_id/readiness - Check promotion readiness
  fastify.get<{ Params: RuleIdParams }>(
    '/bff/v1/rules/:rule_id/readiness',
    {
      schema: {
        params: {
          type: 'object',
          required: ['rule_id'],
          properties: {
            rule_id: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: RuleIdParams }>,
      reply: FastifyReply
    ) => {
      try {
        const { rule_id } = request.params;

        const response = await httpClient.request<ReadinessReportResponse>({
          method: 'GET',
          path: `/rules/${encodeURIComponent(rule_id)}/readiness`,
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

  // GET /bff/v1/rules/:rule_id/versions - List all versions of a rule
  fastify.get<{ Params: RuleIdParams }>(
    '/bff/v1/rules/:rule_id/versions',
    {
      schema: {
        params: {
          type: 'object',
          required: ['rule_id'],
          properties: {
            rule_id: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: RuleIdParams }>,
      reply: FastifyReply
    ) => {
      try {
        const { rule_id } = request.params;

        const response = await httpClient.request<RuleVersionListResponse>({
          method: 'GET',
          path: `/rules/${encodeURIComponent(rule_id)}/versions`,
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

  // GET /bff/v1/rules/:rule_id/versions/:version_id - Get a specific version
  fastify.get<{ Params: VersionParams }>(
    '/bff/v1/rules/:rule_id/versions/:version_id',
    {
      schema: {
        params: {
          type: 'object',
          required: ['rule_id', 'version_id'],
          properties: {
            rule_id: { type: 'string' },
            version_id: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: VersionParams }>,
      reply: FastifyReply
    ) => {
      try {
        const { rule_id, version_id } = request.params;

        const response = await httpClient.request<RuleVersionResponse>({
          method: 'GET',
          path: `/rules/${encodeURIComponent(rule_id)}/versions/${encodeURIComponent(version_id)}`,
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

  // GET /bff/v1/rules/:rule_id/diff - Compare two versions of a rule
  fastify.get<{ Params: RuleIdParams; Querystring: RuleDiffQuery }>(
    '/bff/v1/rules/:rule_id/diff',
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
            version_a: { type: 'string' },
            version_b: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: RuleIdParams; Querystring: RuleDiffQuery }>,
      reply: FastifyReply
    ) => {
      try {
        const { rule_id } = request.params;
        const { version_a, version_b } = request.query;

        const queryParams = new URLSearchParams();
        if (version_a) {
          queryParams.set('version_a', version_a);
        }
        if (version_b) {
          queryParams.set('version_b', version_b);
        }

        const queryString = queryParams.toString();
        const path = `/rules/${encodeURIComponent(rule_id)}/diff${queryString ? `?${queryString}` : ''}`;

        const response = await httpClient.request<RuleDiffResponse>({
          method: 'GET',
          path,
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
