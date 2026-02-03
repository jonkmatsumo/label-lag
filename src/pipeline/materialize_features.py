"""Feature materialization pipeline via Analytics service."""

import logging
import os
from typing import Any

from api.crud_client import get_crud_client

logger = logging.getLogger(__name__)


def materialize_features(
    batch_size: int = 1000,
    **kwargs
) -> dict[str, Any]:
    """Materialize features via Analytics service.

    Args:
        batch_size: Maximum records per batch.
        kwargs: Ignored.

    Returns:
        Statistics dict.
    """
    client = get_crud_client()
    try:
        resp = client.materialize_features(batch_size=batch_size)
        return {
            "success": resp.success,
            "total_processed": resp.total_processed,
        }
    except Exception as e:
        logger.error(f"Failed to materialize features via Analytics: {e}")
        return {"success": False, "error": str(e), "total_processed": 0}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Materialize features via Analytics")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    stats = materialize_features(batch_size=args.batch_size)
    print(f"Feature Materialization Complete: {stats['total_processed']} processed")