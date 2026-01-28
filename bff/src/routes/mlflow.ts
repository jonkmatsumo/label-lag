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
        // MLflow 2.0 API for artifacts is tricky. It often returns a signed URL or requires direct access to S3.
        // However, the standard server supports /get-artifact?path=...&run_uuid=...
        // This endpoint returns the raw bytes.
        
        const response = await mlflowClient.request({
          method: 'GET',
          path: '/get-artifact',
          query: {
            path: request.query.path,
            run_uuid: request.params.run_id,
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
