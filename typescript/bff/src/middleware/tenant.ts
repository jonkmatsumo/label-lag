import { FastifyRequest, FastifyReply, HookHandlerDoneFunction } from 'fastify';
import { Config } from '../config.js';

declare module 'fastify' {
  interface FastifyRequest {
    tenantId: string;
  }
}

/**
 * Tenant middleware factory.
 * Extracts X-Tenant-Id from the request header.
 * Falls back to BFF_DEFAULT_TENANT_ID env var in dev.
 * Returns 400 if tenant cannot be resolved.
 */
export function createTenantMiddleware(config: Config) {
  return function tenantMiddleware(
    request: FastifyRequest,
    reply: FastifyReply,
    done: HookHandlerDoneFunction
  ): void {
    const header = request.headers['x-tenant-id'];
    const tenantId = typeof header === 'string' ? header : config.defaultTenantId;

    if (!tenantId) {
      reply.status(400).send({
        error: {
          code: 'MISSING_TENANT',
          message: 'X-Tenant-Id header is required',
          request_id: request.requestId,
        },
      });
      return;
    }

    request.tenantId = tenantId;
    done();
  };
}
