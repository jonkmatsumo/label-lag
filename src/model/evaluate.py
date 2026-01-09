"""Model evaluation with score calibration and impact analysis."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    scores: np.ndarray
    y_true: np.ndarray
    amounts: np.ndarray | None
    impact_analysis: pd.DataFrame

    @property
    def n_samples(self) -> int:
        return len(self.scores)

    @property
    def n_fraud(self) -> int:
        return int(self.y_true.sum())

    @property
    def fraud_rate(self) -> float:
        if len(self.y_true) == 0:
            return 0.0
        return float(self.y_true.mean())


class ScoreCalibrator:
    """Calibrates raw probabilities to 1-99 integer scores with skewed distribution.

    Real fraud scores (like Plaid) have skewed distributions where:
    - Scores 1-10 are very common (low risk)
    - Scores 90+ are very rare (high risk)

    This calibrator uses a combination of:
    1. Isotonic regression for probability calibration
    2. Power transform to skew the distribution
    """

    def __init__(self, power: float = 3.0):
        """Initialize calibrator.

        Args:
            power: Power transform exponent. Higher values = more skewed toward
                low scores. Default 3.0 gives realistic fraud score distribution
                where most transactions get low risk scores.
        """
        self.power = power
        self._isotonic: IsotonicRegression | None = None
        self._is_fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "ScoreCalibrator":
        """Fit the calibrator using isotonic regression.

        Args:
            y_prob: Raw probability predictions (0.0 to 1.0).
            y_true: True binary labels (0 or 1).

        Returns:
            Self for method chaining.
        """
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(y_prob, y_true)
        self._is_fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform probabilities to 1-99 integer scores.

        Args:
            y_prob: Raw probability predictions (0.0 to 1.0).

        Returns:
            Integer scores from 1 to 99.
        """
        # Apply isotonic calibration if fitted
        if self._is_fitted and self._isotonic is not None:
            calibrated = self._isotonic.predict(y_prob)
        else:
            calibrated = y_prob

        # Apply power transform to skew distribution
        # Lower power = more mass at low scores
        transformed = np.power(calibrated, self.power)

        # Scale to 1-99 range
        scores = np.clip(transformed * 98 + 1, 1, 99).astype(int)

        return scores

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            y_prob: Raw probability predictions (0.0 to 1.0).
            y_true: True binary labels (0 or 1).

        Returns:
            Integer scores from 1 to 99.
        """
        self.fit(y_prob, y_true)
        return self.transform(y_prob)


class ModelEvaluator:
    """Evaluates fraud model with impact analysis and visualization."""

    def __init__(self, calibrator: ScoreCalibrator | None = None):
        """Initialize evaluator.

        Args:
            calibrator: Score calibrator instance. Creates default if None.
        """
        self.calibrator = calibrator or ScoreCalibrator()

    def evaluate(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        amounts: np.ndarray | None = None,
        fit_calibrator: bool = True,
    ) -> EvaluationResult:
        """Evaluate model predictions and generate impact analysis.

        Args:
            y_prob: Raw probability predictions (0.0 to 1.0).
            y_true: True binary labels (0 or 1).
            amounts: Optional transaction amounts for dollar impact analysis.
            fit_calibrator: Whether to fit calibrator on this data.

        Returns:
            EvaluationResult with scores and impact analysis.
        """
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)

        if amounts is not None:
            amounts = np.asarray(amounts)

        # Calibrate scores
        if fit_calibrator:
            scores = self.calibrator.fit_transform(y_prob, y_true)
        else:
            scores = self.calibrator.transform(y_prob)

        # Generate impact analysis
        impact_df = self._generate_impact_analysis(scores, y_true, amounts)

        return EvaluationResult(
            scores=scores,
            y_true=y_true,
            amounts=amounts,
            impact_analysis=impact_df,
        )

    def _generate_impact_analysis(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        amounts: np.ndarray | None,
    ) -> pd.DataFrame:
        """Generate impact analysis report with 99 rows.

        For each threshold 1-99:
        - Records with score >= threshold would be blocked/flagged
        - Calculate recall (fraud caught) and FPR (legitimate blocked)

        Args:
            scores: Calibrated integer scores (1-99).
            y_true: True binary labels.
            amounts: Optional transaction amounts.

        Returns:
            DataFrame with 99 rows, one per threshold.
        """
        total_fraud = y_true.sum()
        total_legit = len(y_true) - total_fraud

        rows = []
        for threshold in range(1, 100):
            # Records at or above threshold would be blocked
            blocked_mask = scores >= threshold

            # True positives: fraud that would be caught
            tp = (blocked_mask & (y_true == 1)).sum()

            # False positives: legitimate that would be blocked
            fp = (blocked_mask & (y_true == 0)).sum()

            # Recall: % of fraud caught
            recall = tp / total_fraud if total_fraud > 0 else 0.0

            # False positive rate: % of legitimate blocked
            fpr = fp / total_legit if total_legit > 0 else 0.0

            row = {
                "threshold": threshold,
                "recall": recall,
                "false_positive_rate": fpr,
                "fraud_caught": int(tp),
                "legitimate_blocked": int(fp),
            }

            # Add dollar estimates if amounts provided
            if amounts is not None:
                fraud_dollars_caught = amounts[blocked_mask & (y_true == 1)].sum()
                legit_dollars_blocked = amounts[blocked_mask & (y_true == 0)].sum()

                row["est_dollar_saved"] = float(fraud_dollars_caught)
                row["est_good_dollar_blocked"] = float(legit_dollars_blocked)

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_threshold_tradeoff(
        self,
        result: EvaluationResult,
        title: str = "Fraud Detection Threshold Analysis",
        figsize: tuple[int, int] = (12, 6),
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot dual-axis chart showing recall vs false positive rate.

        Args:
            result: EvaluationResult from evaluate().
            title: Chart title.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.
        """
        df = result.impact_analysis

        fig, ax1 = plt.subplots(figsize=figsize)

        # Left Y-axis: Fraud Capture Rate (Recall)
        color1 = "#2ecc71"  # Green
        ax1.set_xlabel("Risk Score Threshold", fontsize=12)
        ax1.set_ylabel("Fraud Capture Rate (Recall)", color=color1, fontsize=12)
        line1 = ax1.plot(
            df["threshold"],
            df["recall"] * 100,
            color=color1,
            linewidth=2,
            label="Fraud Caught (%)",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(0, 105)
        ax1.set_xlim(1, 99)

        # Right Y-axis: User Friction (False Positive Rate)
        ax2 = ax1.twinx()
        color2 = "#e74c3c"  # Red
        ax2.set_ylabel("User Friction (False Positive Rate)", color=color2, fontsize=12)
        line2 = ax2.plot(
            df["threshold"],
            df["false_positive_rate"] * 100,
            color=color2,
            linewidth=2,
            linestyle="--",
            label="Legitimate Blocked (%)",
        )
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 105)

        # Add grid
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="center right", fontsize=10)

        # Title
        plt.title(title, fontsize=14, fontweight="bold")

        # Add annotation for optimal zone
        self._add_optimal_zone_annotation(ax1, df)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def _add_optimal_zone_annotation(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
    ) -> None:
        """Add shaded region showing typical operational zone."""
        # Typical operational zone: 70-90% recall with <10% FPR
        good_thresholds = df[(df["recall"] >= 0.7) & (df["false_positive_rate"] <= 0.1)]

        if len(good_thresholds) > 0:
            min_thresh = good_thresholds["threshold"].min()
            max_thresh = good_thresholds["threshold"].max()

            ax.axvspan(
                min_thresh,
                max_thresh,
                alpha=0.15,
                color="blue",
                label="Typical Operational Zone",
            )
            ax.text(
                (min_thresh + max_thresh) / 2,
                95,
                "Optimal\nZone",
                ha="center",
                va="top",
                fontsize=9,
                color="blue",
                alpha=0.7,
            )

    def plot_score_distribution(
        self,
        result: EvaluationResult,
        title: str = "Risk Score Distribution",
        figsize: tuple[int, int] = (12, 5),
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot score distribution showing skew toward low scores.

        Args:
            result: EvaluationResult from evaluate().
            title: Chart title.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Separate fraud and legitimate
        fraud_mask = result.y_true == 1
        fraud_scores = result.scores[fraud_mask]
        legit_scores = result.scores[~fraud_mask]

        # Plot histograms
        bins = np.arange(0.5, 100.5, 1)

        ax.hist(
            legit_scores,
            bins=bins,
            alpha=0.7,
            label=f"Legitimate (n={len(legit_scores)})",
            color="#3498db",
            edgecolor="white",
        )
        ax.hist(
            fraud_scores,
            bins=bins,
            alpha=0.7,
            label=f"Fraud (n={len(fraud_scores)})",
            color="#e74c3c",
            edgecolor="white",
        )

        ax.set_xlabel("Risk Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def export_impact_analysis(
        self,
        result: EvaluationResult,
        path: str | Path,
    ) -> None:
        """Export impact analysis to CSV.

        Args:
            result: EvaluationResult from evaluate().
            path: Output file path.
        """
        result.impact_analysis.to_csv(path, index=False)

    def find_optimal_threshold(
        self,
        result: EvaluationResult,
        min_recall: float = 0.8,
        max_fpr: float = 0.05,
    ) -> dict:
        """Find optimal threshold given constraints.

        Args:
            result: EvaluationResult from evaluate().
            min_recall: Minimum acceptable recall (fraud catch rate).
            max_fpr: Maximum acceptable false positive rate.

        Returns:
            Dict with optimal threshold and metrics, or None if no valid threshold.
        """
        df = result.impact_analysis

        # Filter to valid thresholds
        recall_ok = df["recall"] >= min_recall
        fpr_ok = df["false_positive_rate"] <= max_fpr
        valid = df[recall_ok & fpr_ok]

        if len(valid) == 0:
            # Find best compromise
            df["score"] = df["recall"] - df["false_positive_rate"] * 2
            best_idx = df["score"].idxmax()
            best = df.loc[best_idx]

            return {
                "threshold": int(best["threshold"]),
                "recall": float(best["recall"]),
                "false_positive_rate": float(best["false_positive_rate"]),
                "meets_constraints": False,
                "fraud_caught": int(best["fraud_caught"]),
                "legitimate_blocked": int(best["legitimate_blocked"]),
            }

        # Find threshold with highest recall within constraints
        best_idx = valid["recall"].idxmax()
        best = valid.loc[best_idx]

        return {
            "threshold": int(best["threshold"]),
            "recall": float(best["recall"]),
            "false_positive_rate": float(best["false_positive_rate"]),
            "meets_constraints": True,
            "fraud_caught": int(best["fraud_caught"]),
            "legitimate_blocked": int(best["legitimate_blocked"]),
        }
