package rules

import "testing"

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

func intPtr(v int) *int {
	return &v
}
