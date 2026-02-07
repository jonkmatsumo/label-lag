package rules

import (
	"testing"
)

func BenchmarkEvaluateRules(b *testing.B) {
	// Setup a rule set with mixed types
	score20 := 20
	score10 := 10
	ruleSet := RuleSet{
		Version: "v1",
		Rules: []Rule{
			{ID: "r1", Field: "amount", Op: ">", Value: 100.0, Action: "reject", Severity: "high", Reason: "too high", Status: RuleStatusActive},
			{ID: "r2", Field: "user_score", Op: "<", Value: 50, Action: "clamp_max", Score: &score20, Status: RuleStatusActive},
			{ID: "r3", Field: "status", Op: "==", Value: "active", Action: "override_score", Score: &score10, Status: RuleStatusActive},
			{ID: "r4", Field: "tags", Op: "in", Value: []any{"fraud", "suspicious"}, Action: "reject", Status: RuleStatusActive},
		},
	}
	features := map[string]any{
		"amount":     150.0,
		"user_score": 30,
		"status":     "active",
		"tags":       "suspicious",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateRules(features, 50, &ruleSet, EvalOptions{})
	}
}
