from datetime import UTC, datetime
from hashlib import sha256

from pydantic import BaseModel, Field


class FeatureSetSpec(BaseModel):
    """Specification for a reproducible set of features."""

    id: str = Field(..., description="Unique identifier for this feature set")
    features: list[str] = Field(..., description="Sorted list of feature names")
    hash: str = Field(..., description="SHA256 hash of the feature list")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    created_by: str = Field(default="system")

    @classmethod
    def from_features(
        cls, features: list[str], creator: str = "system"
    ) -> "FeatureSetSpec":
        """Create a spec from a list of features."""
        sorted_features = sorted(features)
        feature_str = ",".join(sorted_features)
        f_hash = sha256(feature_str.encode("utf-8")).hexdigest()
        # ID is hash prefix for now, or could be UUID
        f_id = f"fs-{f_hash[:8]}"
        return cls(
            id=f_id,
            features=sorted_features,
            hash=f_hash,
            created_by=creator,
        )
