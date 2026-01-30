"""Simple background worker for processing jobs from the database."""

import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select

from synthetic_pipeline.db.models import JobDB, JobStatus
from synthetic_pipeline.db.session import DatabaseSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_generate_data(payload):
    """Execution logic for data generation."""
    from api.main import _metadata_to_db, _pydantic_to_db
    from api.schemas import GenerateDataRequest
    from pipeline.materialize_features import FeatureMaterializer
    from synthetic_pipeline.db.models import Base
    from synthetic_pipeline.generator import DataGenerator

    request = GenerateDataRequest(**payload)
    generator = DataGenerator()
    result = generator.generate_dataset_with_sequences(
        num_users=request.num_users,
        fraud_rate=request.fraud_rate,
    )

    fraud_count = sum(1 for r in result.records if r.is_fraudulent)
    db_session = DatabaseSession()

    with db_session.get_session() as session:
        if request.drop_existing:
            Base.metadata.drop_all(db_session.engine)
            Base.metadata.create_all(db_session.engine)
        else:
            Base.metadata.create_all(db_session.engine)

        db_records = [_pydantic_to_db(record) for record in result.records]
        session.bulk_save_objects(db_records)
        meta_records = [_metadata_to_db(meta) for meta in result.metadata]
        session.bulk_save_objects(meta_records)
        session.commit()

    materializer = FeatureMaterializer()
    materialize_stats = materializer.materialize_all()
    features_count = materialize_stats.get("total_processed", 0)

    return {
        "total_records": len(result.records),
        "fraud_records": fraud_count,
        "features_materialized": features_count,
    }


def process_train(payload):
    """Execution logic for model training."""
    from api.schemas import TrainRequest
    from model.train import train_model

    request = TrainRequest(**payload)
    run_id = train_model(
        max_depth=request.max_depth,
        training_window_days=request.training_window_days,
        feature_columns=request.selected_feature_columns,
        split_config=request.split_config,
        n_estimators=request.n_estimators,
        learning_rate=request.learning_rate,
        min_child_weight=request.min_child_weight,
        subsample=request.subsample,
        colsample_bytree=request.colsample_bytree,
        gamma=request.gamma,
        reg_alpha=request.reg_alpha,
        reg_lambda=request.reg_lambda,
        random_state=request.random_state,
        early_stopping_rounds=request.early_stopping_rounds,
        tuning_config=request.tuning_config,
    )
    return {"run_id": run_id}


JOB_PROCESSORS = {"generate_data": process_generate_data, "train": process_train}


def run_worker():
    db_session = DatabaseSession()
    logger.info("Worker started, polling for jobs...")

    while True:
        try:
            with db_session.get_session() as session:
                # Find a pending job
                stmt = (
                    select(JobDB)
                    .where(JobDB.status == JobStatus.PENDING)
                    .order_by(JobDB.created_at.asc())
                    .limit(1)
                )
                job = session.execute(stmt).scalar_one_or_none()

                if not job:
                    time.sleep(2)
                    continue

                # Mark as running
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)
                session.commit()

                logger.info(f"Processing job {job.id} (type: {job.type})")

                try:
                    processor = JOB_PROCESSORS.get(job.type)
                    if not processor:
                        raise ValueError(f"Unknown job type: {job.type}")

                    result = processor(job.payload)

                    # Mark as completed
                    job.status = JobStatus.COMPLETED
                    job.result = result
                    job.finished_at = datetime.now(timezone.utc)
                    logger.info(f"Job {job.id} completed successfully")

                except Exception as e:
                    logger.exception(f"Job {job.id} failed")
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.finished_at = datetime.now(timezone.utc)

                session.commit()

        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run_worker()
