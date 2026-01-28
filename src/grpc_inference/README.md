# grpc-inference service

Standalone gRPC service that hosts model scoring for the Go inference gateway.

## Local run

```bash
python -m grpc_inference.server
```

## Configuration

All configuration uses the `GRPC_INFERENCE_*` namespace.

- `GRPC_INFERENCE_HOST` (default: `0.0.0.0`)
- `GRPC_INFERENCE_PORT` (default: `50052`)
- `GRPC_INFERENCE_DB_DSN` (optional; SQLAlchemy DSN)
- `GRPC_INFERENCE_MLFLOW_TRACKING_URI` (optional)
- `GRPC_INFERENCE_MODEL_NAME` (reserved)
- `GRPC_INFERENCE_MODEL_STAGE` (default: `Production`)
- `GRPC_INFERENCE_INCLUDE_FEATURES_USED` (default: `false`)
- `GRPC_INFERENCE_SCORE_TIMEOUT_MS` (default: `1500`)
- `GRPC_INFERENCE_CONFIG_PATH` (optional dotenv-style file)

If `GRPC_INFERENCE_CONFIG_PATH` is set, its key/value pairs are loaded before
reading environment variables.
