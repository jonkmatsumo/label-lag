"""Tests for heuristic rule suggestion engine."""

from unittest.mock import MagicMock, patch

import pytest

from api.rules import RuleStatus
from api.suggestions import (
    ModelAssistedSuggestionEngine,
    RuleSuggestion,
    SuggestionEngine,
    get_suggestion_engine,
    set_suggestion_engine,
)


class TestRuleSuggestion:
    """Tests for RuleSuggestion dataclass."""

    def test_suggestion_to_rule(self):
        """Test converting suggestion to rule."""
        suggestion = RuleSuggestion(
            field="velocity_24h",
            operator=">",
            threshold=10.0,
            action="clamp_min",
            suggested_score=80,
            confidence=0.8,
            evidence={"percentile_90": 10.0, "sample_count": 1000},
        )

        rule = suggestion.to_rule(rule_id="test_rule")

        assert rule.id == "test_rule"
        assert rule.field == "velocity_24h"
        assert rule.op == ">"
        assert rule.value == 10.0
        assert rule.action == "clamp_min"
        assert rule.score == 80
        assert rule.status == RuleStatus.DRAFT.value

    def test_suggestion_to_rule_generates_id(self):
        """Test that suggestion generates rule ID if not provided."""
        suggestion = RuleSuggestion(
            field="velocity_24h",
            operator=">",
            threshold=10.0,
            action="clamp_min",
            suggested_score=80,
            confidence=0.8,
            evidence={},
        )

        rule = suggestion.to_rule()

        assert rule.id.startswith("suggested_velocity_24h")


class TestSuggestionEngine:
    """Tests for SuggestionEngine class."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = MagicMock()
        # Mock query results with sample data
        mock_rows = [
            (10,),  # velocity_24h
            (5,),
            (15,),
            (8,),
            (20,),
            (3,),
            (12,),
            (25,),
            (7,),
            (18,),
        ] * 10  # 100 samples

        session.get_session.return_value.__enter__.return_value.execute.return_value = (
            iter(mock_rows)
        )
        return session

    def test_generate_suggestions_filters_by_confidence(self, mock_db_session):
        """Test that suggestions are filtered by minimum confidence."""
        engine = SuggestionEngine(db_session=mock_db_session, min_confidence=0.9)

        # Mock the distribution analysis to return suggestions with varying confidence
        with patch.object(engine, "_get_feature_distributions") as mock_dist:
            mock_dist.return_value = {
                "velocity_24h": {
                    "values": list(range(100)),
                    "mean": 50.0,
                    "std": 20.0,
                    "min": 0.0,
                    "max": 100.0,
                    "percentile_75": 75.0,
                    "percentile_90": 90.0,
                    "percentile_95": 95.0,
                    "percentile_99": 99.0,
                    "count": 100,
                }
            }

            suggestions = engine.generate_suggestions(field="velocity_24h")

            # Should only include suggestions with confidence >= 0.9
            assert all(s.confidence >= 0.9 for s in suggestions)

    def test_analyze_field_distribution_high_thresholds(self):
        """Test that high thresholds generate clamp_min suggestions."""
        engine = SuggestionEngine()

        stats = {
            "values": list(range(100)),
            "mean": 50.0,
            "std": 20.0,
            "min": 0.0,
            "max": 100.0,
            "percentile_75": 75.0,
            "percentile_90": 90.0,
            "percentile_95": 95.0,
            "percentile_99": 99.0,
            "count": 100,
        }

        suggestions = engine._analyze_field_distribution("velocity_24h", stats)

        # Should generate suggestions for high percentiles
        assert len(suggestions) > 0
        assert all(s.operator == ">" for s in suggestions)
        assert all(s.action == "clamp_min" for s in suggestions)

    def test_analyze_field_distribution_low_thresholds(self):
        """Test that low balance volatility thresholds generate suggestions."""
        engine = SuggestionEngine()

        stats = {
            "values": [-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 2.0],
            "mean": -1.0,
            "std": 1.5,
            "min": -3.0,
            "max": 2.0,
            "percentile_75": 0.5,
            "percentile_90": 1.5,
            "percentile_95": 2.0,
            "percentile_99": 2.0,
            "count": 8,
        }

        suggestions = engine._analyze_field_distribution(
            "balance_volatility_z_score", stats
        )

        # Should generate suggestions for low thresholds
        low_suggestions = [s for s in suggestions if s.operator == "<"]
        assert len(low_suggestions) > 0

    def test_create_ruleset_from_suggestions(self):
        """Test creating a ruleset from suggestions."""
        engine = SuggestionEngine()

        suggestions = [
            RuleSuggestion(
                field="velocity_24h",
                operator=">",
                threshold=10.0,
                action="clamp_min",
                suggested_score=80,
                confidence=0.8,
                evidence={},
            ),
            RuleSuggestion(
                field="amount_to_avg_ratio_30d",
                operator=">",
                threshold=3.0,
                action="clamp_min",
                suggested_score=75,
                confidence=0.75,
                evidence={},
            ),
        ]

        ruleset = engine.create_ruleset_from_suggestions(suggestions, version="test_v1")

        assert ruleset.version == "test_v1"
        assert len(ruleset.rules) == 2
        assert all(r.status == RuleStatus.DRAFT.value for r in ruleset.rules)

    def test_suggestions_include_evidence(self):
        """Test that suggestions include supporting evidence."""
        engine = SuggestionEngine()

        stats = {
            "values": list(range(100)),
            "mean": 50.0,
            "std": 20.0,
            "min": 0.0,
            "max": 100.0,
            "percentile_75": 75.0,
            "percentile_90": 90.0,
            "percentile_95": 95.0,
            "percentile_99": 99.0,
            "count": 100,
        }

        suggestions = engine._analyze_field_distribution("velocity_24h", stats)

        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert "statistic" in suggestion.evidence
            assert "value" in suggestion.evidence
            assert "sample_count" in suggestion.evidence


class TestGlobalSuggestionEngine:
    """Tests for global suggestion engine."""

    def test_get_suggestion_engine_returns_singleton(self):
        """Test that get_suggestion_engine returns a singleton."""
        engine1 = get_suggestion_engine()
        engine2 = get_suggestion_engine()

        assert engine1 is engine2

    def test_set_suggestion_engine_for_testing(self):
        """Test that set_suggestion_engine allows replacing for testing."""
        original = get_suggestion_engine()
        test_engine = SuggestionEngine()

        set_suggestion_engine(test_engine)
        assert get_suggestion_engine() is test_engine

        # Restore original
        set_suggestion_engine(original)
        assert get_suggestion_engine() is original


class TestModelAssistedSuggestionEngine:
    """Tests for ModelAssistedSuggestionEngine class."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager with feature importance."""
        manager = MagicMock()
        manager.model_loaded = True
        manager.get_feature_importance.return_value = {
            "velocity_24h": 0.3,
            "amount_to_avg_ratio_30d": 0.25,
            "balance_volatility_z_score": 0.2,
            "bank_connections_24h": 0.15,
        }
        manager.required_features = [
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
        ]
        return manager

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = MagicMock()
        mock_rows = [(10.0,), (5.0,), (15.0,), (8.0,), (20.0,)] * 20  # 100 samples
        session.get_session.return_value.__enter__.return_value.execute.return_value = (
            iter(mock_rows)
        )
        return session

    @patch("api.suggestions.get_model_manager")
    def test_generate_suggestions_from_model(
        self, mock_get_manager, mock_model_manager, mock_db_session
    ):
        """Test generating suggestions from model feature importance."""
        mock_get_manager.return_value = mock_model_manager

        engine = ModelAssistedSuggestionEngine(db_session=mock_db_session)

        suggestions = engine.generate_suggestions_from_model(sample_size=100)

        # Should generate suggestions for high-importance features
        assert len(suggestions) > 0
        assert all("feature_importance" in s.evidence for s in suggestions)

    @patch("api.suggestions.get_model_manager")
    def test_suggestions_filtered_by_importance(
        self, mock_get_manager, mock_model_manager, mock_db_session
    ):
        """Test that suggestions are filtered by minimum importance."""
        mock_get_manager.return_value = mock_model_manager

        engine = ModelAssistedSuggestionEngine(
            db_session=mock_db_session, min_importance=0.3
        )

        suggestions = engine.generate_suggestions_from_model()

        # Should only include features with importance >= 0.3
        assert all(s.evidence.get("feature_importance", 0) >= 0.3 for s in suggestions)

    @patch("api.suggestions.get_model_manager")
    def test_no_suggestions_when_model_not_loaded(
        self, mock_get_manager, mock_db_session
    ):
        """Test that no suggestions are generated when model is not loaded."""
        mock_manager = MagicMock()
        mock_manager.model_loaded = False
        mock_get_manager.return_value = mock_manager

        engine = ModelAssistedSuggestionEngine(db_session=mock_db_session)

        suggestions = engine.generate_suggestions_from_model()

        assert len(suggestions) == 0

    def test_create_ruleset_from_model_suggestions(self, mock_db_session):
        """Test creating a ruleset from model-assisted suggestions."""
        engine = ModelAssistedSuggestionEngine(db_session=mock_db_session)

        suggestions = [
            RuleSuggestion(
                field="velocity_24h",
                operator=">",
                threshold=10.0,
                action="clamp_min",
                suggested_score=85,
                confidence=0.8,
                evidence={"feature_importance": 0.3},
            ),
        ]

        ruleset = engine.create_ruleset_from_suggestions(
            suggestions, version="model_v1"
        )

        assert ruleset.version == "model_v1"
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].status == RuleStatus.DRAFT.value
