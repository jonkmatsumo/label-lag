"""CLI entry point for synthetic data generation via Analytics service."""

from typing import Annotated

import typer
from src.logging_util import configure_logging, get_logger

from training_server.crud_client import get_crud_client

app = typer.Typer(
    name="synthetic-data-gen",
    help="Synthetic data generation pipeline for fraud detection.",
    add_completion=False,
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
            help="[DEPRECATED] Ignored. Go generator always uses sequence mode.",
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

    if legacy_mode:
        log.warning("Legacy mode is deprecated and ignored. Using Go generator.")

    log.info(
        "Starting data generation via Analytics service",
        num_users=num_users,
        fraud_rate=fraud_rate,
        seed=seed_value,
        drop_existing=drop_tables,
    )

    client = get_crud_client()

    try:
        resp = client.generate_data(
            num_users=num_users,
            fraud_rate=fraud_rate,
            drop_existing=drop_tables,
            seed=seed_value,
        )

        if not resp.success:
            log.error("Generation failed", error=resp.error)
            raise typer.Exit(code=1)

        # Final summary
        log.info(
            "Seeding complete",
            total_records=resp.total_records,
            fraud_records=resp.fraud_records,
            features_materialized=resp.features_materialized,
        )

        typer.echo(f"\nSuccessfully generated {resp.total_records} transaction records")
        typer.echo(f"Fraud records: {resp.fraud_records}")
        typer.echo(f"Features materialized: {resp.features_materialized}")

    except Exception as e:
        log.error("Operation failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(code=1) from e


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
