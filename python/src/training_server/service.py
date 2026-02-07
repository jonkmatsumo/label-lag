import logging

import grpc

from model.train import train_model
from training_server.crud_client import get_crud_client
from training_server.proto.training.v1 import training_pb2, training_pb2_grpc

logger = logging.getLogger(__name__)


class TrainingService(training_pb2_grpc.TrainingServiceServicer):
    def ClearData(self, request, context):  # noqa: N802
        """Clear all data from the database via Analytics service."""
        try:
            client = get_crud_client()
            resp = client.clear_all_data()
            return training_pb2.ClearDataResponse(
                success=resp.success,
                tables_cleared=list(resp.tables_cleared),
            )
        except Exception as e:
            logger.exception("Data clearing failed")
            context.abort(grpc.StatusCode.INTERNAL, f"Data clearing failed: {e}")

    def Train(self, request, context):  # noqa: N802
        """Train a new model with specified parameters."""
        try:
            # Map parameters from request.parameters (map<string, string>)
            # or use defaults if not provided.
            # For now, we'll try to extract common parameters if they were in the map
            # but the existing TrainRequest had named fields.
            # I should probably update the proto to have named fields for clarity
            # if they are stable. For now, I'll use defaults or the map.

            # Legacy fields from schemas.py:
            # max_depth, training_window_days, selected_feature_columns, etc.

            params = request.parameters or {}

            run_id = train_model(
                max_depth=int(params.get("max_depth", 6)),
                training_window_days=int(params.get("training_window_days", 90)),
                # ... other parameters
            )
            return training_pb2.TrainResponse(
                job_id=run_id,
                status="COMPLETED",
                model_version=f"v{run_id[:8]}",  # Placeholder
            )
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Training failed")
            context.abort(grpc.StatusCode.INTERNAL, f"Training failed: {e}")
