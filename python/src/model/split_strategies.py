"""Split strategies for train/test and CV."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from api.schemas import SplitConfig, SplitStrategy


@dataclass
class SplitResult:
    """Result of a split strategy."""

    train_indices: list[int]
    test_indices: list[int]
    fold_assignments: dict[str, dict[str, list[int]]] | None = None


class BaseSplitStrategy(ABC):
    """Abstract base for split strategies."""

    @abstractmethod
    def split(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp | None,
        config: SplitConfig,
    ) -> SplitResult:
        """Split data into train/test and optionally folds."""
        ...


class TemporalSplitStrategy(BaseSplitStrategy):
    """Temporal split: train before cutoff, test after."""

    def split(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp | None,
        config: SplitConfig,
    ) -> SplitResult:
        """Split by timestamp < cutoff (train) vs >= cutoff (test)."""
        n = len(df)
        if cutoff is None or "transaction_timestamp" not in df.columns:
            mid = int(0.8 * n) or max(1, n - 1)
            return SplitResult(
                train_indices=list(range(0, mid)),
                test_indices=list(range(mid, n)),
            )
        ts = pd.to_datetime(df["transaction_timestamp"])
        train_mask = ts < pd.Timestamp(cutoff)
        train_pos = np.where(train_mask)[0].tolist()
        test_pos = np.where(~train_mask)[0].tolist()
        return SplitResult(
            train_indices=train_pos,
            test_indices=test_pos,
        )


class TimeSeriesKFoldStrategy(BaseSplitStrategy):
    """K contiguous temporal folds for CV."""

    def split(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp | None,
        config: SplitConfig,
    ) -> SplitResult:
        """Produce k folds; optionally use last chunk as holdout test."""
        n = len(df)
        k = config.n_folds
        if n < k:
            # Fallback: single train/test
            mid = max(1, n // 2)
            return SplitResult(
                train_indices=list(range(0, mid)),
                test_indices=list(range(mid, n)),
                fold_assignments={},
            )
        fold_size = n // k
        folds: list[list[int]] = []
        start = 0
        for i in range(k):
            end = n if i == k - 1 else start + fold_size
            folds.append(list(range(start, end)))
            start = end
        fold_assignments: dict[str, dict[str, list[int]]] = {}
        for i, val_idx in enumerate(folds):
            train_idx = [x for j, f in enumerate(folds) if j != i for x in f]
            fold_assignments[f"fold_{i}"] = {"train": train_idx, "val": val_idx}
        # Use last fold as "test" for compatibility
        last = folds[-1]
        train_all = [x for f in folds[:-1] for x in f]
        return SplitResult(
            train_indices=train_all,
            test_indices=last,
            fold_assignments=fold_assignments,
        )


class GroupTemporalSplitStrategy(BaseSplitStrategy):
    """Temporal split with no user overlap between train and test."""

    def split(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp | None,
        config: SplitConfig,
    ) -> SplitResult:
        """Split by time; ensure group_column (e.g. user_id) not in both."""
        group_col = config.group_column or "user_id"
        if group_col not in df.columns or cutoff is None:
            return TemporalSplitStrategy().split(df, cutoff, config)
        ts = pd.to_datetime(df["transaction_timestamp"])
        train_mask = ts < pd.Timestamp(cutoff)
        test_mask = ~train_mask
        train_ids = set(df.loc[train_mask, group_col].dropna().unique())
        test_ids = set(df.loc[test_mask, group_col].dropna().unique())
        overlap = train_ids & test_ids
        if overlap:
            test_mask = test_mask & ~df[group_col].isin(overlap)
        train_pos = np.where(train_mask)[0].tolist()
        test_pos = np.where(test_mask)[0].tolist()
        return SplitResult(
            train_indices=train_pos,
            test_indices=test_pos,
        )


class StratifiedTemporalSplitStrategy(BaseSplitStrategy):
    """Temporal split with stratification within time buckets (stub)."""

    def split(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp | None,
        config: SplitConfig,
    ) -> SplitResult:
        """Delegate to temporal; stratification can be refined later."""
        return TemporalSplitStrategy().split(df, cutoff, config)


class ExpandingWindowStrategy(BaseSplitStrategy):
    """Expanding window (stub)."""

    def split(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp | None,
        config: SplitConfig,
    ) -> SplitResult:
        """Delegate to temporal for now."""
        return TemporalSplitStrategy().split(df, cutoff, config)


def get_strategy(strategy: SplitStrategy) -> BaseSplitStrategy:
    """Return split strategy implementation for the given enum value."""
    map_ = {
        SplitStrategy.TEMPORAL: TemporalSplitStrategy(),
        SplitStrategy.TEMPORAL_STRATIFIED: StratifiedTemporalSplitStrategy(),
        SplitStrategy.GROUP_TEMPORAL: GroupTemporalSplitStrategy(),
        SplitStrategy.KFOLD_TEMPORAL: TimeSeriesKFoldStrategy(),
        SplitStrategy.EXPANDING_WINDOW: ExpandingWindowStrategy(),
    }
    return map_[strategy]
