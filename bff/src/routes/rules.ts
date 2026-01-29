import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import type {
  DraftRulesResponse,
  PublishRuleResponse,
  SandboxEvaluateRequest,
  SandboxEvaluateResponse,
} from '../types/api.js';

export interface RulesRoutesOptions {
  httpClient: HttpClient;
}

interface PublishRuleParams {
  id: string;
}

interface SandboxEvaluateBody {
  base_score: number;
  features: Record<string, unknown>;
  rule_ids?: string[];
}

/**
 * Rules management routes
 */
export async function rulesRoutes(
  fastify: FastifyInstance,
  options: RulesRoutesOptions
): Promise<void> {
  const { httpClient } = options;

  // GET /bff/v1/rules/draft - List draft rules
  fastify.get(
    '/bff/v1/rules/draft',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request<DraftRulesResponse>({
          method: 'GET',
          path: '/rules/draft',
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

  // POST /bff/v1/rules/:id/publish - Publish approved rule to production
  fastify.post<{ Params: PublishRuleParams }>(
    '/bff/v1/rules/:id/publish',
    {
      schema: {
        params: {
          type: 'object',
          required: ['id'],
          properties: {
            id: { type: 'string' },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Params: PublishRuleParams }>,
      reply: FastifyReply
    ) => {
      try {
        const { id } = request.params;

        const response = await httpClient.request<PublishRuleResponse>({
          method: 'POST',
          path: `/rules/${id}/publish`,
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

  // POST /bff/v1/rules/sandbox/evaluate - Evaluate rules in sandbox mode
  fastify.post<{ Body: SandboxEvaluateBody }>(
    '/bff/v1/rules/sandbox/evaluate',
    {
      schema: {
        body: {
          type: 'object',
          required: ['base_score', 'features'],
          properties: {
            base_score: { type: 'number' },
            features: { type: 'object' },
            rule_ids: { type: 'array', items: { type: 'string' } },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{ Body: SandboxEvaluateBody }>,
      reply: FastifyReply
    ) => {
      try {
        const sandboxRequest: SandboxEvaluateRequest = request.body;

        const response = await httpClient.request<SandboxEvaluateResponse>({
          method: 'POST',
          path: '/rules/sandbox/evaluate',
          body: sandboxRequest,
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

  // GET /bff/v1/suggestions/heuristic
  fastify.get(
    '/bff/v1/suggestions/heuristic',
    async (request: FastifyRequest<{ Querystring: { field?: string; min_confidence?: number } }>, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'GET',
          path: '/suggestions/heuristic',
          query: request.query as Record<string, string | number>,
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

  // POST /bff/v1/suggestions/accept
  fastify.post(
    '/bff/v1/suggestions/accept',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await httpClient.request({
          method: 'POST',
          path: '/suggestions/accept',
          body: request.body,
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
