import { FastifyReply, FastifyRequest } from 'fastify';

const REQUEST_ABORT_SIGNAL = Symbol('requestAbortSignal');

type RequestWithAbortSignal = FastifyRequest & {
  [REQUEST_ABORT_SIGNAL]?: AbortSignal;
};

/**
 * Build a per-request AbortSignal that trips when the downstream client
 * disconnects before the response is fully written.
 */
export function getRequestAbortSignal(request: FastifyRequest, reply: FastifyReply): AbortSignal {
  const requestWithSignal = request as RequestWithAbortSignal;
  if (requestWithSignal[REQUEST_ABORT_SIGNAL]) {
    return requestWithSignal[REQUEST_ABORT_SIGNAL];
  }

  const controller = new AbortController();

  const cleanup = () => {
    request.raw.off('aborted', onRequestAborted);
    request.raw.off('close', onRequestClose);
    reply.raw.off('close', onReplyClose);
    reply.raw.off('finish', cleanup);
  };

  const abort = () => {
    if (!controller.signal.aborted) {
      controller.abort();
    }
    cleanup();
  };

  const onRequestAborted = () => {
    abort();
  };

  const onRequestClose = () => {
    if (request.raw.aborted || !reply.raw.writableEnded) {
      abort();
      return;
    }
    cleanup();
  };

  const onReplyClose = () => {
    if (!reply.raw.writableEnded) {
      abort();
      return;
    }
    cleanup();
  };

  request.raw.once('aborted', onRequestAborted);
  request.raw.once('close', onRequestClose);
  reply.raw.once('close', onReplyClose);
  reply.raw.once('finish', cleanup);

  requestWithSignal[REQUEST_ABORT_SIGNAL] = controller.signal;
  return controller.signal;
}
