"""LLM-assisted rule authoring from natural language."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from api.rules import Rule, RuleSet
from api.validation import validate_ruleset

logger = logging.getLogger(__name__)


@dataclass
class LLMGenerationResult:
    """Result of LLM rule generation."""

    rule: Rule | None
    explanation: str
    error: str | None = None
    raw_output: str | None = None


class LLMRuleGenerator:
    """Generates rules from natural language using LLM."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        """Initialize LLM rule generator.

        Args:
            provider: LLM provider ("openai", "anthropic", or "mock" for testing).
            model: Model name to use.
            api_key: API key for the provider. If None, reads from environment.
            max_retries: Maximum number of retries for invalid outputs.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self._cost_tracker: dict[str, int] = {"requests": 0, "tokens": 0}

    def generate_rule(self, description: str, existing_rules: RuleSet | None = None) -> LLMGenerationResult:
        """Generate a rule from natural language description.

        Args:
            description: Natural language description of the rule.
            existing_rules: Optional existing ruleset for context and conflict checking.

        Returns:
            LLMGenerationResult with generated rule or error.
        """
        try:
            # Build prompt
            prompt = self._build_prompt(description, existing_rules)

            # Call LLM
            response = self._call_llm(prompt)

            # Parse response
            rule, explanation = self._parse_llm_response(response, description)

            # If parsing failed, explanation contains the error
            if rule is None:
                return LLMGenerationResult(
                    rule=None,
                    explanation=explanation,
                    error=explanation,  # Use explanation as error when parsing fails
                    raw_output=response,
                )

            # Validate rule
            validation_error = self._validate_rule(rule, existing_rules)
            if validation_error:
                return LLMGenerationResult(
                    rule=None,
                    explanation="",
                    error=f"Generated rule failed validation: {validation_error}",
                    raw_output=response,
                )

            return LLMGenerationResult(
                rule=rule,
                explanation=explanation or f"Generated from: {description}",
                raw_output=response,
            )

        except Exception as e:
            logger.error(f"LLM rule generation failed: {e}", exc_info=True)
            return LLMGenerationResult(
                rule=None,
                explanation="",
                error=f"LLM generation error: {str(e)}",
            )

    def _build_prompt(self, description: str, existing_rules: RuleSet | None) -> str:
        """Build the prompt for LLM.

        Args:
            description: Natural language description.
            existing_rules: Optional existing rules for context.

        Returns:
            Prompt string.
        """
        prompt = f"""You are a rule engine assistant. Convert the following natural language description into a JSON rule definition.

Available fields: velocity_24h, amount_to_avg_ratio_30d, balance_volatility_z_score, bank_connections_24h, merchant_risk_score, has_history, transaction_amount

Available operators: >, >=, <, <=, ==, in, not_in

Available actions: override_score, clamp_min, clamp_max, reject

Return a JSON object with this exact structure:
{{
    "id": "rule_id_here",
    "field": "field_name",
    "op": "operator",
    "value": value_or_list,
    "action": "action_type",
    "score": integer_if_needed,
    "severity": "low|medium|high",
    "reason": "human readable explanation"
}}

Description: {description}
"""

        if existing_rules and existing_rules.rules:
            prompt += "\n\nExisting rules for context (avoid conflicts):\n"
            for rule in existing_rules.rules[:5]:  # Limit to 5 for context
                prompt += f"- {rule.id}: {rule.field} {rule.op} {rule.value} -> {rule.action}\n"

        prompt += "\n\nReturn ONLY valid JSON, no additional text."

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API.

        Args:
            prompt: Prompt to send.

        Returns:
            LLM response text.

        Raises:
            NotImplementedError: If provider is not implemented.
        """
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "mock":
            return self._call_mock(prompt)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API.

        Args:
            prompt: Prompt to send.

        Returns:
            Response text.
        """
        try:
            import os

            import openai

            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")

            client = openai.OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates JSON rule definitions."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more deterministic output
                response_format={"type": "json_object"},
            )

            self._cost_tracker["requests"] += 1
            # Approximate token count (rough estimate)
            self._cost_tracker["tokens"] += len(prompt.split()) + len(response.choices[0].message.content.split())

            return response.choices[0].message.content

        except ImportError:
            raise ValueError("openai package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API.

        Args:
            prompt: Prompt to send.

        Returns:
            Response text.
        """
        try:
            import os

            from anthropic import Anthropic

            api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided")

            client = Anthropic(api_key=api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            self._cost_tracker["requests"] += 1
            self._cost_tracker["tokens"] += response.usage.input_tokens + response.usage.output_tokens

            return response.content[0].text

        except ImportError:
            raise ValueError("anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

    def _call_mock(self, prompt: str) -> str:
        """Mock LLM call for testing.

        Args:
            prompt: Prompt (ignored).

        Returns:
            Mock JSON response.
        """
        # Track cost even for mock
        self._cost_tracker["requests"] += 1
        self._cost_tracker["tokens"] += 100  # Mock token count

        # Simple mock that generates a basic rule
        return json.dumps({
            "id": "llm_generated_rule",
            "field": "velocity_24h",
            "op": ">",
            "value": 10,
            "action": "clamp_min",
            "score": 80,
            "severity": "medium",
            "reason": "High transaction velocity detected",
        })

    def _parse_llm_response(self, response: str, original_description: str) -> tuple[Rule | None, str]:
        """Parse LLM response and extract rule.

        Args:
            response: LLM response text.
            original_description: Original description for context.

        Returns:
            Tuple of (Rule or None, explanation string).
        """
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response

            # Parse JSON
            data = json.loads(response)

            # Create rule
            rule = Rule(**data)
            explanation = data.get("reason", f"Generated from: {original_description}")

            return rule, explanation

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None, f"Failed to parse LLM response: {str(e)}"
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to create rule from LLM response: {e}")
            return None, f"Invalid rule structure: {str(e)}"

    def _validate_rule(self, rule: Rule, existing_rules: RuleSet | None) -> str | None:
        """Validate a generated rule.

        Args:
            rule: Rule to validate.
            existing_rules: Optional existing rules for conflict checking.

        Returns:
            Error message if validation fails, None otherwise.
        """
        # Check for conflicts if existing rules provided
        if existing_rules:
            test_ruleset = RuleSet(version=existing_rules.version, rules=existing_rules.rules + [rule])
            conflicts, redundancies = validate_ruleset(test_ruleset, strict=False)

            if conflicts:
                conflict_msgs = [c.description for c in conflicts]
                return f"Rule conflicts detected: {'; '.join(conflict_msgs)}"

        return None

    def get_cost_stats(self) -> dict[str, int]:
        """Get cost tracking statistics.

        Returns:
            Dictionary with request and token counts.
        """
        return self._cost_tracker.copy()


# Global LLM generator instance
_global_llm_generator: LLMRuleGenerator | None = None


def get_llm_generator() -> LLMRuleGenerator:
    """Get the global LLM generator instance.

    Returns:
        Global LLMRuleGenerator instance.
    """
    global _global_llm_generator
    if _global_llm_generator is None:
        _global_llm_generator = LLMRuleGenerator(provider="mock")  # Default to mock for safety
    return _global_llm_generator


def set_llm_generator(generator: LLMRuleGenerator) -> None:
    """Set the global LLM generator instance (for testing).

    Args:
        generator: LLMRuleGenerator instance to use.
    """
    global _global_llm_generator
    _global_llm_generator = generator
