import logging

import grpc

from model.train import train_model
from training.crud_client import get_crud_client
from training.proto.training.v1 import training_pb2, training_pb2_grpc

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
            # Map proto fields to internal function arguments
            # Note: tuning_config is a message, likely needs conversion to dict
            # if train_model expects one.
            # But based on the proto -> internal mapping standard,
            # let's pass fields directly.

            # Helper to convert TuningConfig message to dict if needed by train_model
            tuning_config = None
            if request.HasField("tuning_config"):
                tuning_config = {
                    "enabled": request.tuning_config.enabled,
                    "strategy": request.tuning_config.strategy,
                    "n_trials": request.tuning_config.n_trials,
                    "timeout_minutes": request.tuning_config.timeout_minutes,
                    "metric": request.tuning_config.metric,
                    "direction": request.tuning_config.direction,
                    "selected_trial_number": (
                        request.tuning_config.selected_trial_number
                        if request.tuning_config.HasField("selected_trial_number")
                        else None
                    ),
                }

            # Helper for SplitConfig
            split_config = None
            if request.HasField("split_config"):
                split_config = {
                    "strategy": request.split_config.strategy,
                    "n_folds": request.split_config.n_folds,
                    "stratify_column": (
                        request.split_config.stratify_column
                        if request.split_config.HasField("stratify_column")
                        else None
                    ),
                    "group_column": request.split_config.group_column,
                    "validation_fraction": request.split_config.validation_fraction,
                    "seed": request.split_config.seed,
                }

            run_id = train_model(
                max_depth=request.max_depth,
                training_window_days=request.training_window_days,
                feature_columns=list(request.selected_feature_columns),
                split_config=split_config,
                n_estimators=request.n_estimators,
                learning_rate=request.learning_rate,
                min_child_weight=request.min_child_weight,
                subsample=request.subsample,
                colsample_bytree=request.colsample_bytree,
                gamma=request.gamma,
                reg_alpha=request.reg_alpha,
                reg_lambda=request.reg_lambda,
                random_state=request.random_state,
                early_stopping_rounds=(
                    request.early_stopping_rounds
                    if request.HasField("early_stopping_rounds")
                    else None
                ),
                tuning_config=tuning_config,
            )
            return training_pb2.TrainResponse(
                success=True,
                run_id=run_id,
                status="COMPLETED",
                model_version=f"v{run_id[:8]}",  # Placeholder
            )
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Training failed")
            context.abort(grpc.StatusCode.INTERNAL, f"Training failed: {e}")
