from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

import grpc
import mlflow
from pydantic import ValidationError

from features.registry import FeatureRegistry
from features.store import get_feature_store
from model.loader import DataLoader
from model.train import train_model
from training.crud_client import get_crud_client
from training.job_queue import JobQueue
from training.job_store import JobStore
from training.jobs import TuningJob, TuningJobStatus
from training.proto.training.v1 import training_pb2, training_pb2_grpc
from training.schemas import SplitConfig, TuningConfig

logger = logging.getLogger(__name__)


class TrainingService(training_pb2_grpc.TrainingServiceServicer):
    def __init__(self, job_store: JobStore, job_queue: JobQueue):
        self.job_store = job_store
        self.job_queue = job_queue

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
            from training.grpc_errors import abort_invalid_argument

            if request.HasField("feature_set_id"):
                if request.selected_feature_columns or request.feature_groups:
                    abort_invalid_argument(
                        context,
                        "feature_set_id",
                        "Cannot specify both feature_set_id and inline features/groups",
                    )

                store = get_feature_store()
                spec = store.get(request.feature_set_id)
                if not spec:
                    abort_invalid_argument(
                        context,
                        "feature_set_id",
                        f"Feature set {request.feature_set_id} not found",
                        "List available feature sets via ListFeatureSets",
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
                                abort_invalid_argument(
                                    context,
                                    "feature_groups",
                                    f"Unknown feature group: '{group}'",
                                    f"Valid groups: {sorted(list(valid_groups))}",
                                )
                            logger.warning(f"Ignoring unknown feature group: {group}")
                            continue

                        group_features = FeatureRegistry.expand_group(c_group)
                        if not group_features:
                            if request.feature_resolution_mode == "strict":
                                abort_invalid_argument(
                                    context,
                                    "feature_groups",
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
                if unknown_features:
                    abort_invalid_argument(
                        context,
                        "selected_feature_columns",
                        f"Unknown features: {unknown_features}",
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

    def GetTrainingRunInfo(self, request, context):  # noqa: N802
        """Fetch TrainingRunSpec from MLflow for a specific run."""
        try:
            if not request.run_id:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "run_id is required")

            client = mlflow.MlflowClient()
            try:
                local_path = client.download_artifacts(
                    request.run_id, "training_run_spec.json"
                )
                with open(local_path) as f:
                    spec_json = f.read()
                return training_pb2.GetTrainingRunInfoResponse(run_spec_json=spec_json)
            except Exception as e:
                logger.warning(f"Run spec not found for {request.run_id}: {e}")
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"Run spec not found for {request.run_id}",
                )
        except grpc.RpcError:
            raise
        except Exception as e:
            logger.exception("GetTrainingRunInfo failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def GetModelInfo(self, request, context):  # noqa: N802
        """Fetch metadata for the active production model."""
        try:
            from forecast.model_manager import MODEL_NAME

            model_name = request.model_name or MODEL_NAME

            client = mlflow.MlflowClient()
            try:
                versions = client.search_model_versions(f"name='{model_name}'")
                prod_v = next(
                    (v for v in versions if v.current_stage == "Production"), None
                )
                if not prod_v:
                    context.abort(
                        grpc.StatusCode.NOT_FOUND,
                        f"No production version for {model_name}",
                    )

                # Try to load metadata from artifacts
                required_features = []
                f_hash = ""
                t_hash = ""

                try:
                    path = client.download_artifacts(
                        prod_v.run_id, "required_features.json"
                    )
                    with open(path) as f:
                        data = json.load(f)
                    required_features = data.get("features", [])
                    f_hash = data.get("feature_set_hash", "")
                    t_hash = data.get("training_config_hash", "")
                except Exception:
                    # Fallback to feature_columns.json
                    try:
                        path = client.download_artifacts(
                            prod_v.run_id, "feature_columns.json"
                        )
                        with open(path) as f:
                            required_features = json.load(f)
                    except Exception:
                        pass

                return training_pb2.GetModelInfoResponse(
                    model_name=model_name,
                    model_version=f"v{prod_v.version}",
                    required_features=required_features,
                    feature_set_hash=f_hash,
                    training_config_hash=t_hash,
                    run_id=prod_v.run_id,
                )
            except grpc.RpcError:
                raise
            except Exception as e:
                logger.warning(f"Model info failed for {model_name}: {e}")
                context.abort(grpc.StatusCode.NOT_FOUND, f"Model info failed: {e}")
        except grpc.RpcError:
            raise
        except Exception as e:
            logger.exception("GetModelInfo failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def ValidateTrainRequest(self, request, context):  # noqa: N802
        """Validate a training request without executing training."""
        try:
            # Shared resolution logic
            from features.spec import FeatureSetSpec
            from model.train import _to_python_type
            from training.grpc_errors import abort_invalid_argument

            warnings = []
            resolved_features_set = set()

            if request.HasField("feature_set_id"):
                if request.selected_feature_columns or request.feature_groups:
                    abort_invalid_argument(
                        context,
                        "feature_set_id",
                        "Cannot specify both feature_set_id and inline features/groups",
                    )
                store = get_feature_store()
                spec = store.get(request.feature_set_id)
                if not spec:
                    abort_invalid_argument(
                        context,
                        "feature_set_id",
                        f"Feature set {request.feature_set_id} not found",
                    )
                resolved_features_set = set(spec.features)
            else:
                if request.selected_feature_columns:
                    resolved_features_set.update(request.selected_feature_columns)

                if request.feature_groups:
                    valid_groups = {"transaction", "user", "merchant", "network"}
                    for group in request.feature_groups:
                        c_group = group.strip().lower()
                        if c_group not in valid_groups:
                            if request.feature_resolution_mode == "strict":
                                abort_invalid_argument(
                                    context,
                                    "feature_groups",
                                    f"Unknown feature group: '{group}'",
                                    f"Valid groups: {sorted(list(valid_groups))}",
                                )
                            warnings.append(f"Ignored unknown feature group: {group}")
                            continue

                        group_features = FeatureRegistry.expand_group(c_group)
                        if not group_features:
                            if request.feature_resolution_mode == "strict":
                                abort_invalid_argument(
                                    context,
                                    "feature_groups",
                                    f"Feature group '{group}' "
                                    "expanded to zero features",
                                )
                            warnings.append(f"Feature group '{group}' is empty")
                        resolved_features_set.update(group_features)

            final_feature_list = sorted(list(resolved_features_set))

            # Validate against DataLoader.FEATURE_COLUMNS
            if final_feature_list:
                unknown_features = [
                    f for f in final_feature_list if f not in DataLoader.FEATURE_COLUMNS
                ]
                if (
                    request.feature_resolution_mode == "best_effort"
                    and unknown_features
                ):
                    warnings.append(f"Dropped unknown features: {unknown_features}")
                    final_feature_list = [
                        f for f in final_feature_list if f in DataLoader.FEATURE_COLUMNS
                    ]
                elif unknown_features:
                    abort_invalid_argument(
                        context,
                        "selected_feature_columns",
                        f"Unknown features: {unknown_features}",
                        f"Available: {DataLoader.FEATURE_COLUMNS}",
                    )

            # Compute preview hashes
            feature_spec = FeatureSetSpec.from_features(final_feature_list)

            # Hyperparams preview
            preview_hyperparams = {
                "scale_pos_weight": 1.0,  # Placeholder as we don't load data
                "max_depth": request.max_depth or 6,
                "n_estimators": request.n_estimators or 100,
                "learning_rate": request.learning_rate or 0.1,
                "min_child_weight": request.min_child_weight or 1,
                "subsample": request.subsample or 1.0,
                "colsample_bytree": request.colsample_bytree or 1.0,
                "gamma": request.gamma or 0.0,
                "reg_alpha": request.reg_alpha or 0.0,
                "reg_lambda": request.reg_lambda or 1.0,
                "random_state": request.random_state or 42,
                "early_stopping_rounds": (
                    request.early_stopping_rounds
                    if request.HasField("early_stopping_rounds")
                    else None
                ),
            }

            # Fake TuningConfig for hashing if provided
            split_dict = None
            if request.HasField("split_config"):
                split_dict = {
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

            config_to_hash = {
                "features": final_feature_list,
                "hyperparameters": preview_hyperparams,
                "split_config": split_dict,
                "training_window_days": request.training_window_days,
            }
            config_json = json.dumps(
                config_to_hash, sort_keys=True, default=_to_python_type
            )
            training_config_hash = hashlib.sha256(
                config_json.encode("utf-8")
            ).hexdigest()

            return training_pb2.ValidateTrainRequestResponse(
                valid=True,
                resolved_features=final_feature_list,
                feature_set_hash=feature_spec.hash,
                training_config_hash=training_config_hash,
                warnings=warnings,
            )
        except grpc.RpcError:
            raise
        except Exception as e:
            logger.exception("ValidateTrainRequest failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def ListFeatureSets(self, request, context):  # noqa: N802
        """List registered feature sets with pagination."""
        try:
            limit = request.limit or 50
            cursor = request.cursor if request.cursor else None

            store = get_feature_store()
            specs, next_cursor = store.list(limit=limit, cursor=cursor)

            summaries = [
                training_pb2.ListFeatureSetsResponse.FeatureSetSummary(
                    id=s.id,
                    hash=s.hash,
                    feature_count=len(s.features),
                    created_at=s.created_at,
                    created_by=s.created_by,
                )
                for s in specs
            ]

            return training_pb2.ListFeatureSetsResponse(
                feature_sets=summaries,
                next_cursor=next_cursor if next_cursor else "",
            )
        except Exception as e:
            logger.exception("ListFeatureSets failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def StartTuningJob(self, request, context):  # noqa: N802
        """Starts an asynchronous hyperparameter tuning job."""
        try:
            # Shared resolution logic
            from training.grpc_errors import abort_invalid_argument

            resolved_features_set = set()

            if request.HasField("feature_set_id"):
                if request.selected_feature_columns or request.feature_groups:
                    abort_invalid_argument(
                        context,
                        "feature_set_id",
                        "Cannot specify both feature_set_id and inline features/groups",
                    )
                store = get_feature_store()
                spec = store.get(request.feature_set_id)
                if not spec:
                    abort_invalid_argument(
                        context,
                        "feature_set_id",
                        f"Feature set {request.feature_set_id} not found",
                    )
                resolved_features_set = set(spec.features)
            else:
                if request.selected_feature_columns:
                    resolved_features_set.update(request.selected_feature_columns)

                if request.feature_groups:
                    valid_groups = {"transaction", "user", "merchant", "network"}
                    for group in request.feature_groups:
                        c_group = group.strip().lower()
                        if c_group not in valid_groups:
                            if request.feature_resolution_mode == "strict":
                                abort_invalid_argument(
                                    context,
                                    "feature_groups",
                                    f"Unknown feature group: '{group}'",
                                )
                            continue

                        group_features = FeatureRegistry.expand_group(c_group)
                        resolved_features_set.update(group_features)

            final_feature_list = sorted(list(resolved_features_set))

            # Validate against DataLoader.FEATURE_COLUMNS
            if final_feature_list:
                unknown_features = [
                    f for f in final_feature_list if f not in DataLoader.FEATURE_COLUMNS
                ]
                if (
                    request.feature_resolution_mode == "best_effort"
                    and unknown_features
                ):
                    final_feature_list = [
                        f for f in final_feature_list if f in DataLoader.FEATURE_COLUMNS
                    ]
                elif unknown_features:
                    abort_invalid_argument(
                        context,
                        "selected_feature_columns",
                        f"Unknown features: {unknown_features}",
                    )

            if not final_feature_list:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No features selected")

            if (
                not request.HasField("tuning_config")
                or not request.tuning_config.enabled
            ):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "Tuning must be enabled"
                )

            # Create MLflow parent run
            from forecast.model_manager import MODEL_NAME

            mlflow.set_experiment(MODEL_NAME)
            with mlflow.start_run(
                run_name=f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ) as run:
                mlflow_run_id = run.info.run_id
                mlflow.set_tag("run_type", "tuning_job")

                # Store config for worker
                job_config = {
                    "training_window_days": request.training_window_days,
                    "feature_columns": final_feature_list,
                    "split_config": {
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
                    },
                    "tuning_config": {
                        "enabled": True,
                        "strategy": request.tuning_config.strategy,
                        "n_trials": request.tuning_config.n_trials,
                        "timeout_minutes": request.tuning_config.timeout_minutes,
                        "metric": request.tuning_config.metric,
                        "direction": request.tuning_config.direction,
                        "search_space": dict(request.tuning_config.search_space),
                    },
                }

                job = TuningJob.create(
                    config=job_config,
                    total_trials=request.tuning_config.n_trials,
                    mlflow_run_id=mlflow_run_id,
                )

                mlflow.set_tag("job_id", job.job_id)
                self.job_store.create(job)
                self.job_queue.enqueue(job.job_id)

                return training_pb2.StartTuningJobResponse(
                    job_id=job.job_id,
                    status=job.status.value,
                    mlflow_run_id=mlflow_run_id,
                )

        except grpc.RpcError:
            raise
        except Exception as e:
            logger.exception("StartTuningJob failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def GetTuningStatus(self, request, context):  # noqa: N802
        """Checks the status of a tuning job."""
        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")

        return training_pb2.TuningJobStatusResponse(
            job_id=job.job_id,
            status=job.status.value,
            completed_trials=job.completed_trials,
            total_trials=job.total_trials,
            pruned_trials=job.pruned_trials,
            best_value=job.best_value if job.best_value is not None else 0.0,
            best_params=job.best_params,
            mlflow_run_id=job.mlflow_run_id or "",
            error_message=job.error_message or "",
            created_at=int(job.created_at.timestamp() * 1000),
            started_at=int(job.started_at.timestamp() * 1000) if job.started_at else 0,
            updated_at=int(job.updated_at.timestamp() * 1000),
            ended_at=int(job.ended_at.timestamp() * 1000) if job.ended_at else 0,
        )

    def ListTrials(self, request, context):  # noqa: N802
        """Lists trials for a specific tuning job."""
        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")

        trials = job.trials
        if request.sort_by == "value":
            trials = sorted(
                trials,
                key=lambda t: t.value if t.value is not None else -1e9,
                reverse=True,
            )

        limit = request.limit or 100
        trials = trials[:limit]

        return training_pb2.ListTrialsResponse(
            trials=[
                training_pb2.TrialRecord(
                    trial_number=t.trial_number,
                    state=t.state,
                    value=t.value if t.value is not None else 0.0,
                    params=t.params,
                    started_at=int(t.started_at.timestamp() * 1000)
                    if t.started_at
                    else 0,
                    ended_at=int(t.ended_at.timestamp() * 1000) if t.ended_at else 0,
                    duration_ms=t.duration_ms or 0.0,
                )
                for t in trials
            ]
        )

    def CancelTuningJob(self, request, context):  # noqa: N802
        """Cancels a running tuning job."""
        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")

        if job.status.is_terminal():
            return self.GetTuningStatus(request, context)

        def set_canceling(j):
            j.status = TuningJobStatus.CANCELING
            j.updated_at = datetime.now(UTC)

        updated_job = self.job_store.update(request.job_id, set_canceling)

        return training_pb2.TuningJobStatusResponse(
            job_id=updated_job.job_id,
            status=updated_job.status.value,
            completed_trials=updated_job.completed_trials,
            total_trials=updated_job.total_trials,
            pruned_trials=updated_job.pruned_trials,
            best_value=updated_job.best_value
            if updated_job.best_value is not None
            else 0.0,
            best_params=updated_job.best_params,
            mlflow_run_id=updated_job.mlflow_run_id or "",
            error_message=updated_job.error_message or "",
            created_at=int(updated_job.created_at.timestamp() * 1000),
            started_at=int(updated_job.started_at.timestamp() * 1000)
            if updated_job.started_at
            else 0,
            updated_at=int(updated_job.updated_at.timestamp() * 1000),
            ended_at=int(updated_job.ended_at.timestamp() * 1000)
            if updated_job.ended_at
            else 0,
        )
