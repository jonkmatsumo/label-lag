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
        result, err := EvaluateRules(features, score, &ruleset, EvalOptions{})
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
        result, err := EvaluateRules(features, score, &ruleset, EvalOptions{})
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

func TestFilterValidRulesSkipsInvalid(t *testing.T) {
        score := 10
        rules := []Rule{
                {
                        ID:     "valid",
                        Field:  "velocity_24h",
                        Op:     ">",
                        Value:  1,
                        Action: "clamp_min",
                        Score:  &score,
                        Status: RuleStatusActive,
                },
                {
                        ID:     "missing-field",
                        Field:  "",
                        Op:     ">",
                        Value:  1,
                        Action: "clamp_min",
                        Score:  &score,
                        Status: RuleStatusActive,
                },
                {
                        ID:     "bad-op",
                        Field:  "velocity_24h",
                        Op:     "???",
                        Value:  1,
                        Action: "reject",
                        Status: RuleStatusActive,
                },
        }

        valid, errs := FilterValidRules(rules)
        if len(valid) != 1 {
                t.Fatalf("expected 1 valid rule, got %d", len(valid))
        }
        if len(errs) != 2 {
                t.Fatalf("expected 2 validation errors, got %d", len(errs))
        }
}

func intPtr(v int) *int {
        return &v
}

func TestEvaluateRules_RejectAndClamp(t *testing.T) {
        // In Python, reject sets score to 99 and rejected=true,
        // but subsequent clamp_max can still reduce the score.
        score := 50
        ruleset := RuleSet{
                Version: "v1",
                Rules: []Rule{
                        {
                                ID:     "reject",
                                Field:  "risk_score",
                                Op:     ">",
                                Value:  80,
                                Action: "reject",
                                Status: RuleStatusActive,
                        },
                        {
                                ID:     "clamp_max",
                                Field:  "risk_score",
                                Op:     ">",
                                Value:  80,
                                Action: "clamp_max",
                                Score:  intPtr(30),
                                Status: RuleStatusActive,
                        },
                },
        }

        features := map[string]any{"risk_score": 90}
        result, err := EvaluateRules(features, score, &ruleset, EvalOptions{})
        if err != nil {
                t.Fatalf("unexpected error: %v", err)
        }

        // Current Go implementation will result in 99 because it blocks clamp after reject.
        // We want it to be 30 to match Python.
        if result.FinalScore != 30 {
                t.Errorf("expected score 30 (clamp after reject), got %d", result.FinalScore)
        }
        if !result.Rejected {
                t.Errorf("expected rejected=true")
        }
}

func TestEvaluateRules_RejectAndOverride(t *testing.T) {
        score := 50
        ruleset := RuleSet{
                Version: "v1",
                Rules: []Rule{
                        {
                                ID:     "reject",
                                Field:  "risk_score",
                                Op:     ">",
                                Value:  80,
                                Action: "reject",
                                Status: RuleStatusActive,
                        },
                        {
                                ID:     "override",
                                Field:  "risk_score",
                                Op:     ">",
                                Value:  80,
                                Action: "override_score",
                                Score:  intPtr(10),
                                Status: RuleStatusActive,
                        },
                },
        }

        features := map[string]any{"risk_score": 90}
        result, err := EvaluateRules(features, score, &ruleset, EvalOptions{})
        if err != nil {
                t.Fatalf("unexpected error: %v", err)
        }

        // Python: override applies after reject if override_applied is false.
        if result.FinalScore != 10 {
                t.Errorf("expected score 10 (override after reject), got %d", result.FinalScore)
        }
        if !result.Rejected {
                t.Errorf("expected rejected=true")
        }
}

func TestEvaluateRules_NumericCoercion(t *testing.T) {
        score := 50
        ruleset := RuleSet{
                Version: "v1",
                Rules: []Rule{
                        {
                                ID:     "numeric-match",
                                Field:  "val",
                                Op:     "==",
                                Value:  10.0,
                                Action: "override_score",
                                Score:  intPtr(99),
                                Status: RuleStatusActive,
                        },
                },
        }

        // Test int feature vs float rule
        features := map[string]any{"val": 10}
        result, _ := EvaluateRules(features, score, &ruleset, EvalOptions{})
        if result.FinalScore != 99 {
                t.Errorf("expected match for int 10 == float 10.0")
        }

        // Test string feature (no coercion)
        features = map[string]any{"val": "10"}
        result, _ = EvaluateRules(features, score, &ruleset, EvalOptions{})
        if result.FinalScore != 50 {
                t.Errorf("expected NO match for string \"10\" == float 10.0")
        }
}

func TestEvaluateRules_MissingFeature(t *testing.T) {
        score := 50
        ruleset := RuleSet{
                Version: "v1",
                Rules: []Rule{
                        {
                                ID:     "missing",
                                Field:  "non_existent",
                                Op:     "==",
                                Value:  1,
                                Action: "reject",
                                Status: RuleStatusActive,
                        },
                },
        }

        features := map[string]any{"other": 1}
        result, _ := EvaluateRules(features, score, &ruleset, EvalOptions{})
        if result.FinalScore != 50 {
                t.Errorf("expected score to remain 50 when feature is missing")
        }
                if len(result.MatchedRules) != 0 {
                        t.Errorf("expected no matched rules")
                }
        }

        func TestFilterValidRules_MaxLimit(t *testing.T) {
                rules := make([]Rule, DefaultMaxRules+1)
                for i := range rules {
                        rules[i] = Rule{ID: "r", Field: "f", Op: "==", Value: 1, Action: "reject", Status: RuleStatusActive}
                }

                valid, errs := FilterValidRules(rules)
                if valid != nil {
                        t.Error("expected nil valid rules when limit exceeded")
                }
                if len(errs) != 1 {
                        t.Errorf("expected 1 error, got %d", len(errs))
                }
        }
