package rules

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEvaluateRules_OverridePrecedence(t *testing.T) {
	score := 50
	override := 10
	clamp := 80
	ruleset := RuleSet{
		Version: "v1",
		Rules: []Rule{
			{
				ID:     "override",
				Field:  "velocity_24h",
				Op:     ">",
				Value:  1,
				Action: "override_score",
				Score:  &override,
				Status: RuleStatusActive,
			},
			{
				ID:     "clamp",
				Field:  "velocity_24h",
				Op:     ">",
				Value:  1,
				Action: "clamp_min",
				Score:  &clamp,
				Status: RuleStatusActive,
			},
		},
	}

	features := map[string]any{"velocity_24h": 10}
	result, err := EvaluateRules(features, score, ruleset)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinalScore != override {
		t.Fatalf("expected override score %d, got %d", override, result.FinalScore)
	}
}

func TestEvaluateRules_ShadowRulesDoNotAffectScore(t *testing.T) {
	score := 50
	ruleset := RuleSet{
		Version: "v1",
		Rules: []Rule{
			{
				ID:     "shadow",
				Field:  "merchant_risk_score",
				Op:     ">",
				Value:  70,
				Action: "override_score",
				Score:  intPtr(99),
				Status: RuleStatusShadow,
			},
		},
	}

	features := map[string]any{"merchant_risk_score": 90}
	result, err := EvaluateRules(features, score, ruleset)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinalScore != score {
		t.Fatalf("expected score to remain %d, got %d", score, result.FinalScore)
	}
	if len(result.ShadowMatchedRules) != 1 {
		t.Fatalf("expected shadow rule match, got %v", result.ShadowMatchedRules)
	}
}

func TestEvaluateRules_Parity(t *testing.T) {
	t.Run("clamp_min", func(t *testing.T) {
		ruleset := RuleSet{
			Version: "v1",
			Rules: []Rule{
				{ID: "min", Field: "velocity_24h", Op: ">", Value: 5.0, Action: "clamp_min", Score: intPtr(80), Status: RuleStatusActive},
			},
		}
		res, _ := EvaluateRules(map[string]any{"velocity_24h": 10}, 50, ruleset)
		assert.Equal(t, 80, res.FinalScore)
	})

	t.Run("clamp_max", func(t *testing.T) {
		ruleset := RuleSet{
			Version: "v1",
			Rules: []Rule{
				{ID: "max", Field: "velocity_24h", Op: "<", Value: 2.0, Action: "clamp_max", Score: intPtr(20), Status: RuleStatusActive},
			},
		}
		res, _ := EvaluateRules(map[string]any{"velocity_24h": 1}, 50, ruleset)
		assert.Equal(t, 20, res.FinalScore)
	})

	t.Run("reject", func(t *testing.T) {
		ruleset := RuleSet{
			Version: "v1",
			Rules: []Rule{
				{ID: "rej", Field: "velocity_24h", Op: ">", Value: 20.0, Action: "reject", Status: RuleStatusActive},
			},
		}
		res, _ := EvaluateRules(map[string]any{"velocity_24h": 25}, 50, ruleset)
		assert.Equal(t, 99, res.FinalScore)
		assert.True(t, res.Rejected)
	})

	t.Run("in_list", func(t *testing.T) {
		ruleset := RuleSet{
			Version: "v1",
			Rules: []Rule{
				{ID: "in", Field: "merchant", Op: "in", Value: []any{"risky", "very_risky"}, Action: "clamp_min", Score: intPtr(90), Status: RuleStatusActive},
			},
		}
		res, _ := EvaluateRules(map[string]any{"merchant": "risky"}, 50, ruleset)
		assert.Equal(t, 90, res.FinalScore)
	})

	t.Run("not_in_list", func(t *testing.T) {
		ruleset := RuleSet{
			Version: "v1",
			Rules: []Rule{
				{ID: "nin", Field: "country", Op: "not_in", Value: []any{"US", "CA"}, Action: "clamp_min", Score: intPtr(70), Status: RuleStatusActive},
			},
		}
		res, _ := EvaluateRules(map[string]any{"country": "UK"}, 50, ruleset)
		assert.Equal(t, 70, res.FinalScore)
	})

	t.Run("score_clamping", func(t *testing.T) {
		ruleset := RuleSet{
			Version: "v1",
			Rules: []Rule{
				{ID: "high", Field: "v", Op: ">", Value: 0.0, Action: "override_score", Score: intPtr(150), Status: RuleStatusActive},
			},
		}
		res, _ := EvaluateRules(map[string]any{"v": 1}, 50, ruleset)
		assert.Equal(t, 99, res.FinalScore)
	})
}

func intPtr(v int) *int {
	return &v
}
