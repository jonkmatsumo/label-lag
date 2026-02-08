import logging

import grpc
from pydantic import ValidationError

from features.registry import FeatureRegistry
from features.store import get_feature_store
from model.loader import DataLoader
from model.train import train_model
from training.crud_client import get_crud_client
from training.proto.training.v1 import training_pb2, training_pb2_grpc
from training.schemas import SplitConfig, TuningConfig

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
            # -1. Feature Set Resolution (FF1)
            # If feature_set_id is present, we must not have inline features/groups
            if request.HasField("feature_set_id"):
                if request.selected_feature_columns or request.feature_groups:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Cannot specify both feature_set_id and inline features/groups",
                    )

                store = get_feature_store()
                spec = store.get(request.feature_set_id)
                if not spec:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Feature set {request.feature_set_id} not found",
                    )
                # Use features from spec
                resolved_features = set(spec.features)
            else:
                # 0. Resolve Features from Groups (Commit 9)
                resolved_features = set()
                if request.selected_feature_columns:
                    resolved_features.update(request.selected_feature_columns)

                if request.feature_groups:
                    valid_groups = {"transaction", "user", "merchant", "network"}
                    for group in request.feature_groups:
                        # Canonicalize (FF4)
                        c_group = group.strip().lower()

                        if c_group not in valid_groups:
                            if request.feature_resolution_mode == "strict":
                                context.abort(
                                    grpc.StatusCode.INVALID_ARGUMENT,
                                    f"Unknown feature group: '{group}'. "
                                    f"Valid groups: {sorted(list(valid_groups))}",
                                )
                            logger.warning(f"Ignoring unknown feature group: {group}")
                            continue

                        group_features = FeatureRegistry.expand_group(c_group)
                        if not group_features:
                            if request.feature_resolution_mode == "strict":
                                context.abort(
                                    grpc.StatusCode.INVALID_ARGUMENT,
                                    f"Feature group '{group}' "
                                    "expanded to zero features",
                                )
                            # best_effort: ignore empty groups
                        resolved_features.update(group_features)

                final_feature_list = sorted(list(resolved_features))

            # 1. Validate Feature Columns
            if final_feature_list:
                # If "strict", we already validated groups.
                # Now check if features exist in DataLoader/Registry?
                # DataLoader.FEATURE_COLUMNS is the "known available" list
                # for training data.
                # Registry is the "known definitions".
                # Commit 1 validation checked against DataLoader.FEATURE_COLUMNS.
                # If we expanded from registry, they exist in registry.
                # But do they exist in DataLoader?
                # DataLoader.FEATURE_COLUMNS was updated to be "default".
                # Actually, we should check if they are *available* in the data.
                # For now, let's stick to checking if they are known.
                unknown_features = [
                    f for f in final_feature_list if f not in DataLoader.FEATURE_COLUMNS
                ]
                # If strict, fail. If best_effort, drop unknown.
                if request.feature_resolution_mode == "best_effort":
                    final_feature_list = [
                        f for f in final_feature_list if f in DataLoader.FEATURE_COLUMNS
                    ]
                elif unknown_features:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Unknown features: {unknown_features}. "
                        f"Available: {DataLoader.FEATURE_COLUMNS}",
                    )

            # 2. Validate Configs using Pydantic Models
            tuning_config = None
            if request.HasField("tuning_config"):
                try:
                    tuning_data = {
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
                        "search_space": dict(request.tuning_config.search_space),
                    }
                    tuning_config = TuningConfig(**tuning_data)
                except ValidationError as e:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT, f"Invalid TuningConfig: {e}"
                    )

            split_config = None
            if request.HasField("split_config"):
                try:
                    split_data = {
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
                    split_config = SplitConfig(**split_data)
                except ValidationError as e:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT, f"Invalid SplitConfig: {e}"
                    )

            run_id = train_model(
                max_depth=request.max_depth,
                training_window_days=request.training_window_days,
                feature_columns=final_feature_list if final_feature_list else None,
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
                feature_set_id=request.feature_set_id
                if request.HasField("feature_set_id")
                else None,
                feature_resolution_mode=request.feature_resolution_mode,
                feature_groups=(
                    list(request.feature_groups) if request.feature_groups else None
                ),
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
