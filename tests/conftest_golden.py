"""Golden file generation and comparison for synthetic data generator.

This module creates deterministic generator output with a fixed seed,
stores it as JSON golden files, and provides comparison fixtures for
validating Go implementation parity.
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from synthetic_pipeline.generator import DataGenerator


GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_SEED = 42
GOLDEN_NUM_USERS = 100
GOLDEN_FRAUD_RATE = 0.05


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal and datetime types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def normalize_record(record_dict: dict) -> dict:
    """Normalize a record dict for comparison (remove volatile fields)."""
    # Remove fields that may vary between runs
    normalized = record_dict.copy()
    # Keep all fields but normalize for comparison
    return normalized


def generate_golden_file():
    """Generate and save golden file with seeded generator output.
    
    Run this manually to regenerate golden files:
        python -c "from tests.conftest_golden import generate_golden_file; generate_golden_file()"
    """
    generator = DataGenerator(seed=GOLDEN_SEED)
    result = generator.generate_dataset_with_sequences(
        num_users=GOLDEN_NUM_USERS,
        fraud_rate=GOLDEN_FRAUD_RATE,
    )

    # Convert to JSON-serializable format
    records = [r.model_dump() for r in result.records]
    metadata = [m.model_dump() for m in result.metadata]

    golden_data = {
        "seed": GOLDEN_SEED,
        "num_users": GOLDEN_NUM_USERS,
        "fraud_rate": GOLDEN_FRAUD_RATE,
        "generated_at": datetime.now().isoformat(),
        "record_count": len(records),
        "fraud_count": sum(1 for r in records if r["is_fraudulent"]),
        "records": records,
        "metadata": metadata,
        # Statistics for parity testing
        "stats": {
            "amount_mean": sum(float(r["transaction"]["amount"]) for r in records) / len(records),
            "amount_min": min(float(r["transaction"]["amount"]) for r in records),
            "amount_max": max(float(r["transaction"]["amount"]) for r in records),
            "z_score_mean": sum(r["behavior"]["balance_volatility_z_score"] for r in records) / len(records),
            "fraud_types": list(set(r["fraud_type"] for r in records if r["fraud_type"])),
        },
    }

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / "generator_seed42.json"
    
    with open(golden_path, "w") as f:
        json.dump(golden_data, f, indent=2, cls=DecimalEncoder)
    
    print(f"Golden file written to: {golden_path}")
    print(f"  Records: {golden_data['record_count']}")
    print(f"  Fraud: {golden_data['fraud_count']}")
    return golden_path


@pytest.fixture
def golden_data() -> dict:
    """Load golden file data for comparison."""
    golden_path = GOLDEN_DIR / "generator_seed42.json"
    if not golden_path.exists():
        pytest.skip(f"Golden file not found: {golden_path}. Run generate_golden_file() first.")
    
    with open(golden_path) as f:
        return json.load(f)


@pytest.fixture
def seeded_generator() -> DataGenerator:
    """Create a generator with the golden seed."""
    return DataGenerator(seed=GOLDEN_SEED)


class TestGoldenFileGeneration:
    """Tests that verify generator output matches golden files."""

    def test_generator_is_deterministic(self):
        """Verify same seed produces same output."""
        gen1 = DataGenerator(seed=42)
        gen2 = DataGenerator(seed=42)

        result1 = gen1.generate_legitimate(count=10)
        result2 = gen2.generate_legitimate(count=10)

        for r1, r2 in zip(result1, result2):
            assert r1.transaction.amount == r2.transaction.amount
            assert r1.behavior.balance_volatility_z_score == r2.behavior.balance_volatility_z_score

    def test_golden_record_count(self, golden_data: dict, seeded_generator: DataGenerator):
        """Verify record count matches golden file."""
        result = seeded_generator.generate_dataset_with_sequences(
            num_users=GOLDEN_NUM_USERS,
            fraud_rate=GOLDEN_FRAUD_RATE,
        )
        assert len(result.records) == golden_data["record_count"]

    def test_golden_fraud_count(self, golden_data: dict, seeded_generator: DataGenerator):
        """Verify fraud count matches golden file."""
        result = seeded_generator.generate_dataset_with_sequences(
            num_users=GOLDEN_NUM_USERS,
            fraud_rate=GOLDEN_FRAUD_RATE,
        )
        fraud_count = sum(1 for r in result.records if r.is_fraudulent)
        assert fraud_count == golden_data["fraud_count"]

    def test_golden_fraud_types(self, golden_data: dict, seeded_generator: DataGenerator):
        """Verify fraud types match golden file."""
        result = seeded_generator.generate_dataset_with_sequences(
            num_users=GOLDEN_NUM_USERS,
            fraud_rate=GOLDEN_FRAUD_RATE,
        )
        fraud_types = set(r.fraud_type for r in result.records if r.fraud_type)
        expected_types = set(golden_data["stats"]["fraud_types"])
        assert fraud_types == expected_types


if __name__ == "__main__":
    generate_golden_file()
