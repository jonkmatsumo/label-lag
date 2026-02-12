from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import UTC, datetime

import grpc
import mlflow
from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import ValidationError

from analytics.v1 import analytics_pb2
from features.registry import FeatureRegistry
from features.store import get_feature_store
from model.loader import DataLoader
from model.train import EXPERIMENT_NAME, train_model
from training.crud_client import get_crud_client
from training.job_queue import JobQueue
from training.job_store import JobStore, encode_jobs_cursor
from training.jobs import (
    TuningJob,
    TuningJobStatus,
    bound_params,
    truncate_error_message,
)
from training.optuna_resume import get_optuna_storage_url, load_existing_tuning_study
from training.schemas import SplitConfig, TuningConfig
from training.v1 import training_pb2, training_pb2_grpc

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class TrainingService(training_pb2_grpc.TrainingServiceServicer):
    def __init__(self, job_store: JobStore, job_queue: JobQueue):
        self.job_store = job_store
        self.job_queue = job_queue
        self.max_tuning_trials = max(1, int(os.getenv("MAX_TUNING_TRIALS", "100")))
        self.max_tuning_timeout_minutes = max(
            1, int(os.getenv("MAX_TUNING_TIMEOUT_MINUTES", "120"))
        )
        self.max_concurrent_tuning_jobs = max(
            1, int(os.getenv("MAX_CONCURRENT_TUNING_JOBS", "1"))
        )
        # ENABLE_TUNING_ADMIN_RPC controls operator-only job repair endpoints.
        self.enable_tuning_admin_rpc = _env_flag("ENABLE_TUNING_ADMIN_RPC", False)

    def _find_trial(self, job_id: str, trial_number: int):
        cursor: str | None = None
        while True:
            trials, next_cursor = self.job_store.list_trials(
                job_id=job_id,
                limit=200,
                cursor=cursor,
                sort_by="trial_number",
            )
            for trial in trials:
                if trial.trial_number == trial_number:
                    return trial
                if trial.trial_number > trial_number:
                    return None
            if not next_cursor:
                return None
            cursor = next_cursor

    @staticmethod
    def _coerce_trial_hyperparameters(params: dict[str, str]) -> dict[str, float | int]:
        int_keys = {"max_depth", "n_estimators", "min_child_weight", "random_state"}
        float_keys = {
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
        }

        coerced: dict[str, float | int] = {}
        for key in int_keys:
            raw = params.get(key)
            if raw is None:
                continue
            try:
                coerced[key] = int(float(raw))
            except ValueError:
                continue

        for key in float_keys:
            raw = params.get(key)
            if raw is None:
                continue
            try:
                coerced[key] = float(raw)
            except ValueError:
                continue

        return coerced

    @staticmethod
    def _split_config_from_job(job: TuningJob) -> SplitConfig | None:
        payload = (
            job.config.get("split_config") if isinstance(job.config, dict) else None
        )
        if not isinstance(payload, dict):
            return None
        try:
            return SplitConfig(**payload)
        except ValidationError:
            return SplitConfig.model_construct(**payload)

    @staticmethod
    def _ensure_registered_model(client: mlflow.MlflowClient, model_name: str) -> None:
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)

    @staticmethod
    def _find_model_version_for_run(
        client: mlflow.MlflowClient, model_name: str, run_id: str
    ) -> str:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception:
            return ""
        for version in versions:
            if version.run_id == run_id:
                return str(version.version)
        return ""

    def _require_admin_rpc_enabled(self, context) -> None:
        if self.enable_tuning_admin_rpc:
            return
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "Admin tuning RPCs are disabled (set ENABLE_TUNING_ADMIN_RPC=true)",
        )

    @staticmethod
    def _build_job_summary(job: TuningJob) -> training_pb2.TuningJobSummary:
        tuning_cfg = (
            job.config.get("tuning_config", {}) if isinstance(job.config, dict) else {}
        )
        summary = training_pb2.TuningJobSummary(
            job_id=job.job_id,
            status=job.status.value,
            created_at=int(job.created_at.timestamp() * 1000),
            started_at=int(job.started_at.timestamp() * 1000) if job.started_at else 0,
            updated_at=int(job.updated_at.timestamp() * 1000),
            ended_at=int(job.ended_at.timestamp() * 1000) if job.ended_at else 0,
            mlflow_run_id=job.mlflow_run_id or "",
            total_trials=job.total_trials,
            completed_trials=job.completed_trials,
            pruned_trials=job.pruned_trials,
            metric=tuning_cfg.get("metric", ""),
            direction=tuning_cfg.get("direction", ""),
        )
        if job.best_value is not None:
            summary.best_value = job.best_value
        if job.requested_by:
            summary.requested_by = job.requested_by
        if job.error_message:
            summary.error_message = truncate_error_message(job.error_message)
        return summary

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
        analytics_run_id = str(uuid.uuid4())
        crud = get_crud_client()

        try:
            # Report Start
            start_ts = Timestamp()
            start_ts.GetCurrentTime()

            run_start = analytics_pb2.TrainingRun(
                run_id=analytics_run_id,
                model_name=EXPERIMENT_NAME,
                status="RUNNING",
                started_at=start_ts,
            )
            try:
                crud.report_training_run(run_start)
            except Exception as e:
                logger.warning(f"Failed to report training start: {e}")

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

            # Report Success
            end_ts = Timestamp()
            end_ts.GetCurrentTime()
            run_complete = analytics_pb2.TrainingRun(
                run_id=analytics_run_id,
                status="COMPLETED",
                ended_at=end_ts,
                mlflow_run_id=run_id,
            )
            try:
                crud.report_training_run(run_complete)
            except Exception as e:
                logger.warning(f"Failed to report training completion: {e}")

            return training_pb2.TrainResponse(
                success=True,
                run_id=run_id,
                status="COMPLETED",
                model_version=f"v{run_id[:8]}",  # Placeholder
            )
        except ValueError as e:
            # Report Failure
            end_ts = Timestamp()
            end_ts.GetCurrentTime()
            try:
                crud.report_training_run(
                    analytics_pb2.TrainingRun(
                        run_id=analytics_run_id,
                        status="FAILED",
                        ended_at=end_ts,
                    )
                )
            except Exception:
                pass
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            # Report Failure
            end_ts = Timestamp()
            end_ts.GetCurrentTime()
            try:
                crud.report_training_run(
                    analytics_pb2.TrainingRun(
                        run_id=analytics_run_id,
                        status="FAILED",
                        ended_at=end_ts,
                    )
                )
            except Exception:
                pass
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
            feature_set_hash = ""

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
                feature_set_hash = spec.hash
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

            requested_trials = request.tuning_config.n_trials
            if requested_trials <= 0:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "tuning_config.n_trials must be greater than zero",
                )
            if requested_trials > self.max_tuning_trials:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    (
                        "tuning_config.n_trials exceeds server limit "
                        f"({requested_trials} > {self.max_tuning_trials})"
                    ),
                )

            requested_timeout_minutes = request.tuning_config.timeout_minutes
            if requested_timeout_minutes <= 0:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "tuning_config.timeout_minutes must be greater than zero",
                )
            if requested_timeout_minutes > self.max_tuning_timeout_minutes:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    (
                        "tuning_config.timeout_minutes exceeds server limit "
                        f"({requested_timeout_minutes} > "
                        f"{self.max_tuning_timeout_minutes})"
                    ),
                )

            active_jobs = self.job_store.list_jobs(
                statuses=[
                    TuningJobStatus.PENDING,
                    TuningJobStatus.RUNNING,
                    TuningJobStatus.CANCELING,
                ],
                limit=self.max_concurrent_tuning_jobs + 1,
            )
            if len(active_jobs) >= self.max_concurrent_tuning_jobs:
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    (
                        "Maximum concurrent tuning jobs reached "
                        f"({self.max_concurrent_tuning_jobs})"
                    ),
                )

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
            tuning_config = {
                "enabled": True,
                "strategy": request.tuning_config.strategy,
                "n_trials": request.tuning_config.n_trials,
                "timeout_minutes": request.tuning_config.timeout_minutes,
                "metric": request.tuning_config.metric,
                "direction": request.tuning_config.direction,
                "search_space": dict(request.tuning_config.search_space),
            }

            training_config_hash = hashlib.sha256(
                json.dumps(
                    {
                        "training_window_days": request.training_window_days,
                        "feature_columns": final_feature_list,
                        "split_config": split_config,
                        "tuning_config": tuning_config,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

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
                    "split_config": split_config,
                    "tuning_config": tuning_config,
                    "optuna_storage_url": get_optuna_storage_url(),
                }

                job = TuningJob.create(
                    config=job_config,
                    total_trials=request.tuning_config.n_trials,
                    mlflow_run_id=mlflow_run_id,
                )

                mlflow.set_tag("job_id", job.job_id)
                mlflow.set_tag("tuning_job_id", job.job_id)
                mlflow.set_tag("tuning_metric", request.tuning_config.metric)
                mlflow.set_tag("tuning_direction", request.tuning_config.direction)
                mlflow.set_tag("tuning_strategy", request.tuning_config.strategy)
                mlflow.set_tag("training_config_hash", training_config_hash)
                if feature_set_hash:
                    mlflow.set_tag("feature_set_hash", feature_set_hash)
                self.job_store.create(job)
                self.job_queue.enqueue(job.job_id)
                logger.info(
                    "tuning_job_enqueued job_id=%s queue_depth=%s total_trials=%s",
                    job.job_id,
                    self.job_queue.depth(),
                    job.total_trials,
                )

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
            best_params=bound_params(job.best_params),
            mlflow_run_id=job.mlflow_run_id or "",
            error_message=truncate_error_message(job.error_message) or "",
            created_at=int(job.created_at.timestamp() * 1000),
            started_at=int(job.started_at.timestamp() * 1000) if job.started_at else 0,
            updated_at=int(job.updated_at.timestamp() * 1000),
            ended_at=int(job.ended_at.timestamp() * 1000) if job.ended_at else 0,
            heartbeat_at=int(job.heartbeat_at.timestamp() * 1000)
            if job.heartbeat_at
            else 0,
        )

    def ListTuningJobs(self, request, context):  # noqa: N802
        """Lists tuning jobs for operators."""
        try:
            sort_statuses = request.statuses if request.statuses else []
            statuses: list[TuningJobStatus] | None = None
            if sort_statuses:
                statuses = []
                for raw_status in sort_statuses:
                    try:
                        statuses.append(TuningJobStatus(raw_status))
                    except ValueError:
                        context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            f"Unknown status filter '{raw_status}'",
                        )

            limit = request.limit or 50
            limit = max(1, min(limit, 200))
            cursor = request.cursor if request.cursor else None
            jobs = self.job_store.list_jobs(
                statuses=statuses,
                limit=limit + 1,
                cursor=cursor,
            )
            has_more = len(jobs) > limit
            page_jobs = jobs[:limit]
            next_cursor = (
                encode_jobs_cursor(page_jobs[-1].created_at, page_jobs[-1].job_id)
                if page_jobs and has_more
                else ""
            )

            summaries = []
            for job in page_jobs:
                summaries.append(self._build_job_summary(job))

            return training_pb2.ListTuningJobsResponse(
                jobs=summaries,
                next_cursor=next_cursor,
            )
        except grpc.RpcError:
            raise
        except Exception as e:
            logger.exception("ListTuningJobs failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def GetQueueDepth(self, request, context):  # noqa: N802
        """Returns tuning queue depth."""
        return training_pb2.GetQueueDepthResponse(depth=self.job_queue.depth())

    def PromoteTrial(self, request, context):  # noqa: N802
        if not request.job_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "job_id is required")
        if request.trial_number < 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "trial_number must be non-negative",
            )

        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")
        if not job.status.is_terminal() and job.status != TuningJobStatus.RUNNING:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Job {job.job_id} must be terminal or RUNNING for promotion",
            )

        trial = self._find_trial(job.job_id, request.trial_number)
        if not trial:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                (f"Trial {request.trial_number} not found for job {request.job_id}"),
            )

        trial_hyperparams = self._coerce_trial_hyperparameters(trial.params)
        target_model_name = request.model_name or EXPERIMENT_NAME
        split_config = self._split_config_from_job(job)
        feature_columns = (
            job.config.get("feature_columns") if isinstance(job.config, dict) else None
        )
        training_window_days = (
            int(job.config.get("training_window_days", 30))
            if isinstance(job.config, dict)
            else 30
        )

        if request.dry_run:
            return training_pb2.PromoteTrialResponse(
                status="COMPLETED",
                error_message=(
                    f"dry-run: would promote trial {request.trial_number} "
                    f"from job {request.job_id} to model '{target_model_name}'"
                ),
            )

        logger.info(
            "promote_trial_started job_id=%s trial_number=%s model_name=%s",
            request.job_id,
            request.trial_number,
            target_model_name,
        )
        try:
            run_id = train_model(
                training_window_days=training_window_days,
                feature_columns=feature_columns,
                split_config=split_config,
                max_depth=int(trial_hyperparams.get("max_depth", 6)),
                n_estimators=int(trial_hyperparams.get("n_estimators", 100)),
                learning_rate=float(trial_hyperparams.get("learning_rate", 0.1)),
                min_child_weight=int(trial_hyperparams.get("min_child_weight", 1)),
                subsample=float(trial_hyperparams.get("subsample", 1.0)),
                colsample_bytree=float(trial_hyperparams.get("colsample_bytree", 1.0)),
                gamma=float(trial_hyperparams.get("gamma", 0.0)),
                reg_alpha=float(trial_hyperparams.get("reg_alpha", 0.0)),
                reg_lambda=float(trial_hyperparams.get("reg_lambda", 1.0)),
                random_state=int(trial_hyperparams.get("random_state", 42)),
            )

            client = mlflow.MlflowClient()
            for key, value in {
                "tuning_job_id": job.job_id,
                "tuning_trial_number": str(request.trial_number),
                "run_type": "promoted_tuning_trial",
                "source_tuning_run_id": job.mlflow_run_id or "",
            }.items():
                client.set_tag(run_id, key, value)

            model_version = self._find_model_version_for_run(
                client, EXPERIMENT_NAME, run_id
            )
            if target_model_name != EXPERIMENT_NAME:
                self._ensure_registered_model(client, target_model_name)
                created_version = client.create_model_version(
                    name=target_model_name,
                    source=f"runs:/{run_id}/model",
                    run_id=run_id,
                )
                model_version = str(created_version.version)

            logger.info(
                "promote_trial_completed job_id=%s trial_number=%s run_id=%s",
                request.job_id,
                request.trial_number,
                run_id,
            )
            return training_pb2.PromoteTrialResponse(
                status="COMPLETED",
                mlflow_run_id=run_id,
                model_version=model_version,
            )
        except Exception as exc:
            logger.exception(
                "promote_trial_failed job_id=%s trial_number=%s",
                request.job_id,
                request.trial_number,
            )
            return training_pb2.PromoteTrialResponse(
                status="FAILED",
                error_message=truncate_error_message(str(exc)) or "unknown error",
            )

    def GetTuningJobInfo(self, request, context):  # noqa: N802
        if not request.job_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "job_id is required")
        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")

        sort_by = request.sort_by or "trial_number"
        if sort_by not in {"trial_number", "value"}:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unsupported sort_by '{sort_by}'. Expected 'trial_number' or 'value'.",
            )

        trials_limit = request.trials_limit or 50
        trials_limit = max(1, min(trials_limit, 200))
        cursor = request.trials_cursor if request.trials_cursor else None
        if sort_by == "value" and cursor:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "trials_cursor is only supported when sort_by=trial_number",
            )

        trials, next_cursor = self.job_store.list_trials(
            request.job_id,
            limit=trials_limit,
            cursor=cursor,
            sort_by=sort_by,
        )

        links: dict[str, str] = {}
        if job.mlflow_run_id:
            links["tuning_run_id"] = job.mlflow_run_id

        return training_pb2.GetTuningJobInfoResponse(
            job=self._build_job_summary(job),
            trials=[
                training_pb2.TrialResult(
                    trial_number=trial.trial_number,
                    state=trial.state,
                    value=trial.value if trial.value is not None else 0.0,
                    params=bound_params(trial.params),
                    started_at=int(trial.started_at.timestamp() * 1000)
                    if trial.started_at
                    else 0,
                    ended_at=int(trial.ended_at.timestamp() * 1000)
                    if trial.ended_at
                    else 0,
                    duration_ms=trial.duration_ms or 0.0,
                )
                for trial in trials
            ],
            next_cursor=next_cursor or "",
            mlflow_links=links,
        )

    def ListTrials(self, request, context):  # noqa: N802
        """Lists trials for a specific tuning job."""
        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")

        sort_by = request.sort_by or "trial_number"
        if sort_by not in {"trial_number", "value"}:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unsupported sort_by '{sort_by}'. Expected 'trial_number' or 'value'.",
            )

        limit = request.limit or 50
        limit = max(1, min(limit, 200))

        cursor = request.cursor if request.cursor else None
        if sort_by == "value" and cursor:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "cursor is only supported when sort_by=trial_number",
            )

        trials, next_cursor = self.job_store.list_trials(
            request.job_id,
            limit=limit,
            cursor=cursor,
            sort_by=sort_by,
        )

        return training_pb2.ListTrialsResponse(
            trials=[
                training_pb2.TrialRecord(
                    trial_number=t.trial_number,
                    state=t.state,
                    value=t.value if t.value is not None else 0.0,
                    params=bound_params(t.params),
                    started_at=int(t.started_at.timestamp() * 1000)
                    if t.started_at
                    else 0,
                    ended_at=int(t.ended_at.timestamp() * 1000) if t.ended_at else 0,
                    duration_ms=t.duration_ms or 0.0,
                )
                for t in trials
            ],
            next_cursor=next_cursor or "",
        )

    def CancelTuningJob(self, request, context):  # noqa: N802
        """Cancels a running tuning job."""
        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")

        if job.status.is_terminal():
            return self.GetTuningStatus(request, context)
        if job.status == TuningJobStatus.CANCELING:
            return self.GetTuningStatus(request, context)

        now = datetime.now(UTC)
        if job.status == TuningJobStatus.PENDING:

            def set_canceled(j):
                j.status = TuningJobStatus.CANCELED
                j.updated_at = now
                j.ended_at = now

            self.job_store.update(request.job_id, set_canceled)
            self.job_queue.cancel(request.job_id)
            return self.GetTuningStatus(request, context)

        def set_canceling(j):
            j.status = TuningJobStatus.CANCELING
            j.updated_at = now

        self.job_store.update(request.job_id, set_canceling)
        return self.GetTuningStatus(request, context)

    def RequeueTuningJob(self, request, context):  # noqa: N802
        self._require_admin_rpc_enabled(context)
        if not request.job_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "job_id is required")

        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")
        if job.status not in {TuningJobStatus.FAILED, TuningJobStatus.CANCELED}:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Only FAILED or CANCELED jobs can be requeued",
            )

        tuning_cfg = (
            job.config.get("tuning_config", {}) if isinstance(job.config, dict) else {}
        )
        storage_url = (
            job.config.get("optuna_storage_url")
            if isinstance(job.config, dict)
            else None
        ) or get_optuna_storage_url()
        if not storage_url:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Resume unavailable: Optuna storage is not configured",
            )
        study = load_existing_tuning_study(
            job_id=job.job_id,
            direction=tuning_cfg.get("direction", "maximize"),
            storage_url=storage_url,
        )
        if not study:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Resume unavailable: Optuna study does not exist",
            )

        now = datetime.now(UTC)

        def mark_pending(current):
            current.status = TuningJobStatus.PENDING
            current.updated_at = now
            current.ended_at = None
            current.error_message = None

        self.job_store.update(job.job_id, mark_pending)
        self.job_queue.enqueue(job.job_id)
        return training_pb2.RequeueTuningJobResponse(
            job_id=job.job_id,
            status=TuningJobStatus.PENDING.value,
            message="job requeued",
        )

    def FinalizeTuningJob(self, request, context):  # noqa: N802
        self._require_admin_rpc_enabled(context)
        if not request.job_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "job_id is required")
        if not request.final_status:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "final_status is required")

        try:
            target_status = TuningJobStatus(request.final_status)
        except ValueError:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unsupported final_status '{request.final_status}'",
            )
        if not target_status.is_terminal():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "final_status must be a terminal status",
            )

        job = self.job_store.get(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")

        now = datetime.now(UTC)

        def finalize(current):
            current.status = target_status
            current.updated_at = now
            current.ended_at = now
            if request.reason:
                current.error_message = truncate_error_message(request.reason)

        self.job_store.update(job.job_id, finalize)
        if target_status == TuningJobStatus.CANCELED:
            self.job_queue.cancel(job.job_id)
        return training_pb2.FinalizeTuningJobResponse(
            job_id=job.job_id,
            status=target_status.value,
            error_message=truncate_error_message(request.reason) or "",
        )

    def GetHealth(self, request, context):  # noqa: N802
        """Fetch health status including spool information."""
        try:
            import os

            var_dir = os.path.join(os.getcwd(), "var")
            log_path = os.path.join(var_dir, "training_run_reports.jsonl")

            spooled_count = 0
            if os.path.exists(log_path):
                with open(log_path) as f:
                    spooled_count = sum(1 for line in f if line.strip())

            return training_pb2.GetHealthResponse(
                ready=True,
                components={"training": "ok"},
                spooled_reports_count=spooled_count,
            )
        except Exception as e:
            logger.exception("GetHealth failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
