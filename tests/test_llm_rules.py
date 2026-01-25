"""Tests for LLM-assisted rule authoring."""

import json
from unittest.mock import MagicMock, patch

import pytest

from api.llm_rules import LLMGenerationResult, LLMRuleGenerator, get_llm_generator, set_llm_generator
from api.rules import Rule, RuleSet


class TestLLMRuleGenerator:
    """Tests for LLMRuleGenerator class."""

    def test_generate_rule_mock_provider(self):
        """Test generating a rule with mock provider."""
        generator = LLMRuleGenerator(provider="mock")

        result = generator.generate_rule("Block transactions with velocity > 10")

        assert result.rule is not None
        assert result.rule.field == "velocity_24h"
        assert result.error is None

    def test_generate_rule_parses_json(self):
        """Test that rule generation parses JSON correctly."""
        generator = LLMRuleGenerator(provider="mock")

        result = generator.generate_rule("High velocity rule")

        assert result.rule is not None
        assert isinstance(result.rule, Rule)
        assert result.rule.id == "llm_generated_rule"

    def test_generate_rule_includes_explanation(self):
        """Test that generated rules include explanations."""
        generator = LLMRuleGenerator(provider="mock")

        result = generator.generate_rule("Test description")

        assert result.explanation is not None
        assert len(result.explanation) > 0

    def test_generate_rule_validates_against_existing_rules(self):
        """Test that generated rules are validated against existing rules."""
        generator = LLMRuleGenerator(provider="mock")

        # Create existing ruleset with conflicting rule
        existing_rule = Rule(
            id="existing",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        existing_ruleset = RuleSet(version="v1", rules=[existing_rule])

        # Mock generator to create conflicting rule
        with patch.object(generator, "_call_mock") as mock_call:
            mock_call.return_value = json.dumps({
                "id": "conflicting",
                "field": "velocity_24h",
                "op": ">",
                "value": 3,  # Overlaps with existing rule
                "action": "reject",  # Different action = conflict
                "severity": "high",
                "reason": "Conflicting rule",
            })

            result = generator.generate_rule("Test", existing_rules=existing_ruleset)

            # Should detect conflict
            assert result.error is not None
            assert "conflict" in result.error.lower()

    def test_generate_rule_handles_invalid_json(self):
        """Test that invalid JSON responses are handled gracefully."""
        generator = LLMRuleGenerator(provider="mock")

        with patch.object(generator, "_call_mock") as mock_call:
            mock_call.return_value = "This is not JSON"

            result = generator.generate_rule("Test")

            assert result.rule is None
            assert result.error is not None

    def test_generate_rule_handles_invalid_rule_structure(self):
        """Test that invalid rule structures are handled."""
        generator = LLMRuleGenerator(provider="mock")

        with patch.object(generator, "_call_mock") as mock_call:
            mock_call.return_value = json.dumps({
                "id": "invalid",
                "field": "velocity_24h",
                # Missing required fields
            })

            result = generator.generate_rule("Test")

            assert result.rule is None
            assert result.error is not None

    def test_build_prompt_includes_existing_rules(self):
        """Test that prompt includes existing rules for context."""
        generator = LLMRuleGenerator(provider="mock")

        existing_rule = Rule(
            id="existing",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        existing_ruleset = RuleSet(version="v1", rules=[existing_rule])

        prompt = generator._build_prompt("Test description", existing_ruleset)

        assert "existing" in prompt
        assert "velocity_24h" in prompt

    def test_cost_tracking(self):
        """Test that cost tracking works."""
        generator = LLMRuleGenerator(provider="mock")

        generator.generate_rule("Test 1")
        generator.generate_rule("Test 2")

        stats = generator.get_cost_stats()
        assert stats["requests"] == 2

    def test_validate_rule_detects_conflicts(self):
        """Test that rule validation detects conflicts."""
        generator = LLMRuleGenerator(provider="mock")

        existing_rule = Rule(
            id="existing",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
        )
        existing_ruleset = RuleSet(version="v1", rules=[existing_rule])

        # Create conflicting rule
        conflicting_rule = Rule(
            id="conflicting",
            field="velocity_24h",
            op=">=",
            value=3,
            action="reject",
        )

        error = generator._validate_rule(conflicting_rule, existing_ruleset)

        assert error is not None
        assert "conflict" in error.lower()


class TestGlobalLLMGenerator:
    """Tests for global LLM generator."""

    def test_get_llm_generator_returns_singleton(self):
        """Test that get_llm_generator returns a singleton."""
        generator1 = get_llm_generator()
        generator2 = get_llm_generator()

        assert generator1 is generator2

    def test_set_llm_generator_for_testing(self):
        """Test that set_llm_generator allows replacing for testing."""
        original = get_llm_generator()
        test_generator = LLMRuleGenerator(provider="mock")

        set_llm_generator(test_generator)
        assert get_llm_generator() is test_generator

        # Restore original
        set_llm_generator(original)
        assert get_llm_generator() is original
