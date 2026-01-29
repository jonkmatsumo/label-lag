import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { HttpClient, UpstreamError } from '../services/http-client.js';

export interface MlflowRoutesOptions {
  httpClient: HttpClient;
  mlflowTrackingUri: string;
}

export async function mlflowRoutes(
  fastify: FastifyInstance,
  options: MlflowRoutesOptions
): Promise<void> {
  const { httpClient, mlflowTrackingUri } = options;

  // Helper to create MLflow client specific for this route
  // We need a separate client because the base URL is different (mlflow vs api)
  const mlflowClient = new HttpClient({
    config: {
      ...httpClient.config,
      fastApiBaseUrl: mlflowTrackingUri, // Override base URL to point to MLflow
    },
    logger: fastify.log,
  });

  // GET /bff/v1/mlflow/experiments/search
  fastify.get(
    '/bff/v1/mlflow/experiments/search',
    async (request: FastifyRequest<{ Querystring: { filter?: string } }>, reply: FastifyReply) => {
      try {
        const response = await mlflowClient.request({
          method: 'GET',
          path: '/api/2.0/mlflow/experiments/search',
          query: request.query as Record<string, string>,
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

  // POST /bff/v1/mlflow/runs/search
  fastify.post(
    '/bff/v1/mlflow/runs/search',
    async (request: FastifyRequest, reply: FastifyReply) => {
      try {
        const response = await mlflowClient.request({
          method: 'POST',
          path: '/api/2.0/mlflow/runs/search',
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

  // GET /bff/v1/mlflow/model-versions/search
  fastify.get(
    '/bff/v1/mlflow/model-versions/search',
    async (request: FastifyRequest<{ Querystring: { filter?: string } }>, reply: FastifyReply) => {
      try {
        const response = await mlflowClient.request({
          method: 'GET',
          path: '/api/2.0/mlflow/model-versions/search',
          query: request.query as Record<string, string>,
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

  // GET /bff/v1/mlflow/runs/:run_id/artifacts
  // Proxies artifact content (JSON only)
  fastify.get(
    '/bff/v1/mlflow/runs/:run_id/artifacts',
    async (request: FastifyRequest<{ Params: { run_id: string }; Querystring: { path: string } }>, reply: FastifyReply) => {
      try {
        const { run_id } = request.params;
        const { path } = request.query;

        // Security: Validate run_id (alphanumeric/dashes)
        if (!/^[a-zA-Z0-9-]+$/.test(run_id)) {
          return reply.status(400).send({ error: { code: 'INVALID_INPUT', message: 'Invalid run_id' } });
        }

        // Security: Sanitize path to prevent traversal
        if (path.includes('..') || path.startsWith('/') || path.includes('\\')) {
          return reply.status(400).send({ error: { code: 'INVALID_INPUT', message: 'Invalid artifact path' } });
        }

        // Allow only specific file extensions (JSON, PNG, CSV) to prevent fetching dangerous content or massive binaries
        if (!/\.(json|csv|png|txt|yaml|yml)$/i.test(path)) {
           return reply.status(400).send({ error: { code: 'INVALID_INPUT', message: 'Artifact type not allowed' } });
        }

        const response = await mlflowClient.request({
          method: 'GET',
          path: '/get-artifact',
          query: {
            path,
            run_uuid: run_id,
          },
          requestId: request.requestId,
        });
        
        // Return raw data
        return reply.status(response.statusCode).send(response.data);
      } catch (error) {
        if (error instanceof UpstreamError) {
          // 404 means artifact not found
          return reply.status(error.statusCode).send(error.toResponse());
        }
        throw error;
      }
    }
  );
}
