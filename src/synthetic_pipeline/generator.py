"""Synthetic data generator wrapper around core simulator."""

from generator.core import (
    BustOutProfile,
    SleeperProfile,
    UserSequenceResult,
    UserSimulator,
)


class DataGenerator:
    """Legacy wrapper for DataGenerator using new core UserSimulator."""

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate_dataset_with_sequences(
        self, num_users: int = 100, fraud_rate: float = 0.05
    ) -> UserSequenceResult:
        """Generate a complete dataset with user sequences using core simulator."""
        all_records = []
        all_metadata = []

        # Simple implementation using UserSimulator directly to match old interface
        # but with improved stateful logic
        import numpy as np

        rng = np.random.default_rng(self.seed)

        num_fraud = int(num_users * fraud_rate)
        num_legit = num_users - num_fraud

        # Legit
        for _ in range(num_legit):
            s = UserSimulator(seed=int(rng.integers(0, 2**31)) if self.seed else None)
            recs, meta = s.generate_full_sequence()
            all_records.extend(recs)
            all_metadata.extend(meta)

        # Fraud (alternating between profiles)
        for i in range(num_fraud):
            profile = BustOutProfile() if i % 2 == 0 else SleeperProfile()
            s = UserSimulator(
                fraud_profile=profile,
                seed=int(rng.integers(0, 2**31)) if self.seed else None,
            )
            recs, meta = s.generate_full_sequence()
            all_records.extend(recs)
            all_metadata.extend(meta)

        return UserSequenceResult(records=all_records, metadata=all_metadata)
