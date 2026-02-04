"""CLI entry point for synthetic data generation via Analytics service."""

from typing import Annotated

import typer

from api.crud_client import get_crud_client
from synthetic_pipeline.generator import DataGenerator, FraudType
from synthetic_pipeline.logging import configure_logging, get_logger
from synthetic_pipeline.models import EvaluationMetadata, GeneratedRecord

app = typer.Typer(
    name="synthetic-data-gen",
    help="Synthetic data generation pipeline for fraud detection.",
    add_completion=False,
)


def pydantic_to_proto(record: GeneratedRecord):
    """Convert a Pydantic GeneratedRecord to proto message."""
    from google.protobuf.timestamp_pb2 import Timestamp

    from api.proto.proto.crud.v1 import analytics_pb2

    ts = Timestamp()
    ts.FromDatetime(record.transaction_timestamp)

    ec_ts = Timestamp()
    if record.identity_changes.email_changed_at:
        ec_ts.FromDatetime(record.identity_changes.email_changed_at)

    pc_ts = Timestamp()
    if record.identity_changes.phone_changed_at:
        pc_ts.FromDatetime(record.identity_changes.phone_changed_at)

    return analytics_pb2.GeneratedRecord(
        record_id=record.record_id,
        user_id=record.user_id,
        full_name=record.full_name,
        email=record.email,
        phone=record.phone,
        transaction_timestamp=ts,
        is_off_hours_txn=record.is_off_hours_txn,
        available_balance=float(record.account.available_balance),
        balance_to_transaction_ratio=float(record.account.balance_to_transaction_ratio),
        avg_available_balance_30d=float(record.behavior.avg_available_balance_30d),
        balance_volatility_z_score=float(record.behavior.balance_volatility_z_score),
        bank_connections_count_24h=record.connection.bank_connections_count_24h,
        bank_connections_count_7d=record.connection.bank_connections_count_7d,
        bank_connections_avg_30d=float(record.connection.bank_connections_avg_30d),
        amount=float(record.transaction.amount),
        amount_to_avg_ratio=float(record.transaction.amount_to_avg_ratio),
        merchant_risk_score=record.transaction.merchant_risk_score,
        is_returned=record.transaction.is_returned,
        email_changed_at=ec_ts if record.identity_changes.email_changed_at else None,
        phone_changed_at=pc_ts if record.identity_changes.phone_changed_at else None,
        is_fraudulent=record.is_fraudulent,
        fraud_type=record.fraud_type or "",
    )


def metadata_to_proto(meta: EvaluationMetadata):
    """Convert a Pydantic EvaluationMetadata to proto message."""
    from google.protobuf.timestamp_pb2 import Timestamp

    from api.proto.proto.crud.v1 import analytics_pb2

    fc_ts = Timestamp()
    if meta.fraud_confirmed_at:
        fc_ts.FromDatetime(meta.fraud_confirmed_at)

    return analytics_pb2.EvaluationMetadata(
        user_id=meta.user_id,
        record_id=meta.record_id,
        sequence_number=meta.sequence_number,
        fraud_confirmed_at=fc_ts if meta.fraud_confirmed_at else None,
        is_pre_fraud=meta.is_pre_fraud,
        days_to_fraud=meta.days_to_fraud or 0,
        is_train_eligible=meta.is_train_eligible,
    )


@app.command()
def seed(
    num_users: Annotated[
        int,
        typer.Option("--users", "-u", help="Number of unique users to generate"),
    ] = 100,
    fraud_rate: Annotated[
        float,
        typer.Option(
            "--fraud-rate",
            "-f",
            help="Fraction of users that should have fraud events (0.0-1.0)",
        ),
    ] = 0.05,
    seed_value: Annotated[
        int | None,
        typer.Option("--seed", "-s", help="Random seed for reproducibility"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for database inserts"),
    ] = 500,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", envvar="DATABASE_URL", help="Database URL (ignored)"
        ),
    ] = None,
    drop_tables: Annotated[
        bool,
        typer.Option("--drop-tables", help="Drop existing tables before seeding"),
    ] = False,
    json_logs: Annotated[
        bool,
        typer.Option("--json-logs", help="Output logs in JSON format"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    legacy_mode: Annotated[
        bool,
        typer.Option(
            "--legacy",
            help="Use legacy generation (single transaction per user, no sequences)",
        ),
    ] = False,
) -> None:
    """Generate synthetic transaction profiles and seed via Analytics service."""
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level, json_format=json_logs)
    log = get_logger("seed")

    # Validate inputs
    if not 0.0 <= fraud_rate <= 1.0:
        log.error("Invalid fraud rate", fraud_rate=fraud_rate)
        raise typer.BadParameter("Fraud rate must be between 0.0 and 1.0")

    if num_users < 1:
        log.error("Invalid user count", num_users=num_users)
        raise typer.BadParameter("User count must be at least 1")

    # Calculate counts
    num_fraud_users = int(num_users * fraud_rate)
    num_legitimate_users = num_users - num_fraud_users

    log.info(
        "Starting data generation",
        num_users=num_users,
        fraud_rate=fraud_rate,
        fraud_users=num_fraud_users,
        legitimate_users=num_legitimate_users,
        seed=seed_value,
        mode="legacy" if legacy_mode else "sequences",
    )

    # Initialize generator
    generator = DataGenerator(seed=seed_value)

    if legacy_mode:
        legitimate_records = generator.generate_legitimate(count=num_legitimate_users)
        fraudulent_records: list[GeneratedRecord] = []

        if num_fraud_users > 0:
            fraud_types = list(FraudType)
            per_type = num_fraud_users // len(fraud_types)
            remainder = num_fraud_users % len(fraud_types)

            for i, fraud_type in enumerate(fraud_types):
                type_count = per_type + (1 if i < remainder else 0)
                if type_count > 0:
                    records = generator.generate_fraudulent(
                        fraud_type, count=type_count
                    )
                    fraudulent_records.extend(records)

        all_records = legitimate_records + fraudulent_records
        all_metadata: list[EvaluationMetadata] = []
    else:
        result = generator.generate_dataset_with_sequences(
            num_users=num_users,
            fraud_rate=fraud_rate,
        )
        all_records = result.records
        all_metadata = result.metadata

    client = get_crud_client()

    try:
        if drop_tables:
            log.warning("Clearing all existing data via Analytics service")
            client.clear_all_data()

        log.info("Converting records to proto messages")
        proto_records = [pydantic_to_proto(r) for r in all_records]
        proto_metadata = [metadata_to_proto(m) for m in all_metadata]

        log.info(
            "Storing records via Analytics service",
            record_count=len(proto_records),
        )

        resp = client.store_generated_data(
            records=proto_records, metadata=proto_metadata
        )
        log.info("Records inserted", count=resp.records_saved)

    except Exception as e:
        log.error("Operation failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(code=1) from e

    # Final summary
    log.info(
        "Seeding complete",
        total_records=len(all_records),
        evaluation_metadata=len(all_metadata),
    )

    typer.echo(f"\nSuccessfully generated {len(all_records)} transaction records")


@app.command()
def init_db(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", envvar="DATABASE_URL", help="Database URL (ignored)"
        ),
    ] = None,
    drop_tables: Annotated[
        bool,
        typer.Option("--drop-tables", help="Drop existing tables (ignored)"),
    ] = False,
) -> None:
    """Initialization is now handled by the Analytics service on startup."""
    typer.echo("Database initialization is now managed by the Analytics service.")


@app.command()
def stats(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", envvar="DATABASE_URL", help="Database URL (ignored)"
        ),
    ] = None,
) -> None:
    """Show statistics via Analytics service."""
    configure_logging()
    log = get_logger("stats")

    client = get_crud_client()
    resp = client.get_overview_metrics()

    log.info(
        "Database statistics",
        total_records=resp.total_records,
        unique_users=resp.unique_users,
        fraudulent=resp.fraud_records,
        fraud_rate=round(resp.fraud_rate, 4),
    )

    typer.echo("\nDatabase Statistics:")
    typer.echo(f"  Total records: {resp.total_records}")
    typer.echo(f"  Unique users: {resp.unique_users}")
    typer.echo(f"  Fraudulent: {resp.fraud_records}")
    typer.echo(f"  Fraud rate: {round(resp.fraud_rate, 2)}%")


if __name__ == "__main__":
    app()
