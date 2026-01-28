import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';
import type { TrainRequest, TrainResponse, DeployRequest, DeployResponse } from '../types/api.js';

export interface ModelRoutesOptions {
  httpClient: HttpClient;
}

interface TrainBody {
  name?: string;
  test_size?: number;
  random_seed?: number;
  selected_columns?: string[];
}

interface DeployBody {
  model_version?: string;
  run_id?: string;
}

/**
 * Model training and deployment routes
 */
export async function modelRoutes(
  fastify: FastifyInstance,
  options: ModelRoutesOptions
): Promise<void> {
  const { httpClient } = options;

  // POST /bff/v1/train - Train new model
  fastify.post<{ Body: TrainBody }>(
    '/bff/v1/train',
    {
      schema: {
        body: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            test_size: { type: 'number' },
            random_seed: { type: 'number' },
            selected_columns: { type: 'array', items: { type: 'string' } },
          },
        },
      },
    },
    async (request: FastifyRequest<{ Body: TrainBody }>, reply: FastifyReply) => {
      try {
        const trainRequest: TrainRequest = request.body;

        const response = await httpClient.request<TrainResponse>({
          method: 'POST',
          path: '/train',
          body: trainRequest,
          requestId: request.requestId,
          timeout: 300000, // 5 minutes for training
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

  // POST /bff/v1/models/deploy - Deploy model to production
  fastify.post<{ Body: DeployBody }>(
    '/bff/v1/models/deploy',
    {
      schema: {
        body: {
          type: 'object',
          properties: {
            model_version: { type: 'string' },
            run_id: { type: 'string' },
          },
        },
      },
    },
    async (request: FastifyRequest<{ Body: DeployBody }>, reply: FastifyReply) => {
      try {
        const deployRequest: DeployRequest = request.body;

        const response = await httpClient.request<DeployResponse>({
          method: 'POST',
          path: '/models/deploy',
          body: deployRequest,
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
