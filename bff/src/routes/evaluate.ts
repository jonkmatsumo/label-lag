import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import { Config } from '../config.js';
import type { SignalRequest, SignalResponse } from '../types/api.js';

export interface EvaluateRoutesOptions {
  httpClient: HttpClient;
  config: Config;
}

interface EvaluateSignalBody {
  user_id: string;
  amount: number;
  currency: string;
  client_transaction_id: string;
}

/**
 * Signal evaluation routes
 */
export async function evaluateRoutes(
  fastify: FastifyInstance,
  options: EvaluateRoutesOptions
): Promise<void> {
  const { httpClient, config } = options;

  // POST /bff/v1/evaluate/signal - Risk evaluation
  fastify.post<{ Body: EvaluateSignalBody }>(
    '/bff/v1/evaluate/signal',
    {
      schema: {
        body: {
          type: 'object',
          required: ['user_id', 'amount', 'currency', 'client_transaction_id'],
          properties: {
            user_id: { type: 'string' },
            amount: { type: 'number' },
            currency: { type: 'string' },
            client_transaction_id: { type: 'string' },
          },
        },
      },
    },
    async (request: FastifyRequest<{ Body: EvaluateSignalBody }>, reply: FastifyReply) => {
      try {
        const signalRequest: SignalRequest = request.body;

        // Route based on inference mode
        const target = config.inferenceMode === 'gateway' ? 'gateway' : 'fastapi';

        const response = await httpClient.request<SignalResponse>({
          method: 'POST',
          path: '/evaluate/signal',
          body: signalRequest,
          requestId: request.requestId,
          target,
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
