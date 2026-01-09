"""Tests for model evaluation with score calibration and impact analysis."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.evaluate import EvaluationResult, ModelEvaluator, ScoreCalibrator


class TestScoreCalibrator:
    """Tests for ScoreCalibrator."""

    def test_transform_returns_integers(self):
        calibrator = ScoreCalibrator()
        y_prob = np.array([0.1, 0.5, 0.9])
        scores = calibrator.transform(y_prob)

        assert scores.dtype == np.int64 or scores.dtype == np.int32

    def test_transform_range_1_to_99(self):
        calibrator = ScoreCalibrator()
        y_prob = np.linspace(0, 1, 100)
        scores = calibrator.transform(y_prob)

        assert scores.min() >= 1
        assert scores.max() <= 99

    def test_transform_preserves_order(self):
        calibrator = ScoreCalibrator()
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        scores = calibrator.transform(y_prob)

        # Higher probabilities should give higher or equal scores
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_fit_uses_isotonic_regression(self):
        calibrator = ScoreCalibrator()
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        y_true = np.array([0, 0, 1, 1])

        calibrator.fit(y_prob, y_true)
        assert calibrator._is_fitted
        assert calibrator._isotonic is not None

    def test_fit_transform_combines_operations(self):
        calibrator = ScoreCalibrator()
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        y_true = np.array([0, 0, 1, 1])

        scores = calibrator.fit_transform(y_prob, y_true)
        assert len(scores) == len(y_prob)
        assert calibrator._is_fitted

    def test_power_affects_distribution(self):
        """Higher power should skew more toward low scores."""
        y_prob = np.linspace(0, 1, 1000)

        calibrator_low = ScoreCalibrator(power=1.0)
        calibrator_high = ScoreCalibrator(power=4.0)

        scores_low = calibrator_low.transform(y_prob)
        scores_high = calibrator_high.transform(y_prob)

        # Higher power should have lower mean score (more skew to low)
        assert scores_high.mean() < scores_low.mean()

    def test_skewed_distribution_low_scores_common(self):
        """Default calibrator should produce more low scores than high."""
        calibrator = ScoreCalibrator(power=3.0)
        # Simulate uniform probability distribution
        y_prob = np.random.RandomState(42).uniform(0, 1, 10000)
        scores = calibrator.transform(y_prob)

        low_count = (scores <= 20).sum()
        high_count = (scores >= 80).sum()

        # Low scores should be much more common
        assert low_count > high_count * 2


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_n_samples(self):
        result = EvaluationResult(
            scores=np.array([1, 2, 3, 4, 5]),
            y_true=np.array([0, 0, 1, 1, 0]),
            amounts=None,
            impact_analysis=pd.DataFrame(),
        )
        assert result.n_samples == 5

    def test_n_fraud(self):
        result = EvaluationResult(
            scores=np.array([1, 2, 3, 4, 5]),
            y_true=np.array([0, 0, 1, 1, 0]),
            amounts=None,
            impact_analysis=pd.DataFrame(),
        )
        assert result.n_fraud == 2

    def test_fraud_rate(self):
        result = EvaluationResult(
            scores=np.array([1, 2, 3, 4, 5]),
            y_true=np.array([0, 0, 1, 1, 0]),
            amounts=None,
            impact_analysis=pd.DataFrame(),
        )
        assert result.fraud_rate == 0.4

    def test_fraud_rate_empty(self):
        result = EvaluationResult(
            scores=np.array([]),
            y_true=np.array([]),
            amounts=None,
            impact_analysis=pd.DataFrame(),
        )
        assert result.fraud_rate == 0.0


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample prediction data."""
        np.random.seed(42)
        n = 1000

        # Create probabilities that correlate with fraud
        y_true = np.random.binomial(1, 0.1, n)
        y_prob = np.where(
            y_true == 1,
            np.random.beta(5, 2, n),  # Higher probs for fraud
            np.random.beta(2, 5, n),  # Lower probs for legitimate
        )
        amounts = np.random.exponential(100, n)

        return y_prob, y_true, amounts

    def test_evaluate_returns_result(self, sample_data):
        y_prob, y_true, amounts = sample_data
        evaluator = ModelEvaluator()

        result = evaluator.evaluate(y_prob, y_true)

        assert isinstance(result, EvaluationResult)
        assert len(result.scores) == len(y_prob)

    def test_evaluate_with_amounts(self, sample_data):
        y_prob, y_true, amounts = sample_data
        evaluator = ModelEvaluator()

        result = evaluator.evaluate(y_prob, y_true, amounts=amounts)

        assert result.amounts is not None
        assert "est_dollar_saved" in result.impact_analysis.columns
        assert "est_good_dollar_blocked" in result.impact_analysis.columns

    def test_impact_analysis_has_99_rows(self, sample_data):
        y_prob, y_true, _ = sample_data
        evaluator = ModelEvaluator()

        result = evaluator.evaluate(y_prob, y_true)

        assert len(result.impact_analysis) == 99

    def test_impact_analysis_columns(self, sample_data):
        y_prob, y_true, _ = sample_data
        evaluator = ModelEvaluator()

        result = evaluator.evaluate(y_prob, y_true)
        df = result.impact_analysis

        assert "threshold" in df.columns
        assert "recall" in df.columns
        assert "false_positive_rate" in df.columns
        assert "fraud_caught" in df.columns
        assert "legitimate_blocked" in df.columns

    def test_impact_analysis_thresholds_1_to_99(self, sample_data):
        y_prob, y_true, _ = sample_data
        evaluator = ModelEvaluator()

        result = evaluator.evaluate(y_prob, y_true)
        df = result.impact_analysis

        assert df["threshold"].min() == 1
        assert df["threshold"].max() == 99
        assert list(df["threshold"]) == list(range(1, 100))

    def test_recall_decreases_with_threshold(self, sample_data):
        """Higher threshold = fewer blocked = less fraud caught."""
        y_prob, y_true, _ = sample_data
        evaluator = ModelEvaluator()

        result = evaluator.evaluate(y_prob, y_true)
        df = result.impact_analysis

        # Recall should generally decrease (with some tolerance for ties)
        recalls = df["recall"].values
        assert recalls[0] >= recalls[-1]

    def test_fpr_decreases_with_threshold(self, sample_data):
        """Higher threshold = fewer blocked = fewer false positives."""
        y_prob, y_true, _ = sample_data
        evaluator = ModelEvaluator()

        result = evaluator.evaluate(y_prob, y_true)
        df = result.impact_analysis

        fprs = df["false_positive_rate"].values
        assert fprs[0] >= fprs[-1]

    def test_dollar_amounts_cumulative(self, sample_data):
        """Dollar amounts should decrease with higher thresholds."""
        y_prob, y_true, amounts = sample_data
        evaluator = ModelEvaluator()

        result = evaluator.evaluate(y_prob, y_true, amounts=amounts)
        df = result.impact_analysis

        dollars_saved = df["est_dollar_saved"].values
        assert dollars_saved[0] >= dollars_saved[-1]


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold method."""

    @pytest.fixture
    def evaluator_with_result(self):
        """Create evaluator with sample result."""
        np.random.seed(42)
        n = 1000

        y_true = np.random.binomial(1, 0.1, n)
        y_prob = np.where(
            y_true == 1,
            np.random.beta(5, 2, n),
            np.random.beta(2, 5, n),
        )

        evaluator = ModelEvaluator()
        result = evaluator.evaluate(y_prob, y_true)

        return evaluator, result

    def test_returns_dict(self, evaluator_with_result):
        evaluator, result = evaluator_with_result

        optimal = evaluator.find_optimal_threshold(result)

        assert isinstance(optimal, dict)
        assert "threshold" in optimal
        assert "recall" in optimal
        assert "false_positive_rate" in optimal
        assert "meets_constraints" in optimal

    def test_respects_min_recall(self, evaluator_with_result):
        evaluator, result = evaluator_with_result

        optimal = evaluator.find_optimal_threshold(result, min_recall=0.8)

        if optimal["meets_constraints"]:
            assert optimal["recall"] >= 0.8

    def test_respects_max_fpr(self, evaluator_with_result):
        evaluator, result = evaluator_with_result

        optimal = evaluator.find_optimal_threshold(result, max_fpr=0.1)

        if optimal["meets_constraints"]:
            assert optimal["false_positive_rate"] <= 0.1


class TestVisualization:
    """Tests for visualization methods."""

    @pytest.fixture
    def evaluator_with_result(self):
        np.random.seed(42)
        n = 500

        y_true = np.random.binomial(1, 0.1, n)
        y_prob = np.where(
            y_true == 1,
            np.random.beta(5, 2, n),
            np.random.beta(2, 5, n),
        )

        evaluator = ModelEvaluator()
        result = evaluator.evaluate(y_prob, y_true)

        return evaluator, result

    def test_plot_threshold_tradeoff_returns_figure(self, evaluator_with_result):
        import matplotlib.pyplot as plt

        evaluator, result = evaluator_with_result

        fig = evaluator.plot_threshold_tradeoff(result)

        assert fig is not None
        plt.close(fig)

    def test_plot_threshold_tradeoff_saves_file(self, evaluator_with_result):
        import matplotlib.pyplot as plt

        evaluator, result = evaluator_with_result

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plot.png"
            evaluator.plot_threshold_tradeoff(result, save_path=path)

            assert path.exists()

        plt.close("all")

    def test_plot_score_distribution_returns_figure(self, evaluator_with_result):
        import matplotlib.pyplot as plt

        evaluator, result = evaluator_with_result

        fig = evaluator.plot_score_distribution(result)

        assert fig is not None
        plt.close(fig)

    def test_plot_score_distribution_saves_file(self, evaluator_with_result):
        import matplotlib.pyplot as plt

        evaluator, result = evaluator_with_result

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dist.png"
            evaluator.plot_score_distribution(result, save_path=path)

            assert path.exists()

        plt.close("all")


class TestExportImpactAnalysis:
    """Tests for CSV export."""

    def test_export_creates_file(self):
        np.random.seed(42)
        n = 100

        y_true = np.random.binomial(1, 0.1, n)
        y_prob = np.random.uniform(0, 1, n)

        evaluator = ModelEvaluator()
        result = evaluator.evaluate(y_prob, y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "impact.csv"
            evaluator.export_impact_analysis(result, path)

            assert path.exists()

            df = pd.read_csv(path)
            assert len(df) == 99
            assert "threshold" in df.columns

    def test_export_with_amounts(self):
        np.random.seed(42)
        n = 100

        y_true = np.random.binomial(1, 0.1, n)
        y_prob = np.random.uniform(0, 1, n)
        amounts = np.random.exponential(100, n)

        evaluator = ModelEvaluator()
        result = evaluator.evaluate(y_prob, y_true, amounts=amounts)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "impact.csv"
            evaluator.export_impact_analysis(result, path)

            df = pd.read_csv(path)
            assert "est_dollar_saved" in df.columns
            assert "est_good_dollar_blocked" in df.columns
