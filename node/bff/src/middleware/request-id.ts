import { FastifyRequest, FastifyReply, HookHandlerDoneFunction } from 'fastify';
import { v4 as uuidv4 } from 'uuid';

declare module 'fastify' {
  interface FastifyRequest {
    requestId: string;
  }
}

/**
 * Request ID middleware
 * Uses incoming X-Request-Id header or generates a new UUID
 * Adds request ID to response headers and request object
 */
export function requestIdMiddleware(
  request: FastifyRequest,
  reply: FastifyReply,
  done: HookHandlerDoneFunction
): void {
  const incomingRequestId = request.headers['x-request-id'];
  const requestId = typeof incomingRequestId === 'string'
    ? incomingRequestId
    : uuidv4();

  request.requestId = requestId;
  reply.header('X-Request-Id', requestId);

  done();
}
