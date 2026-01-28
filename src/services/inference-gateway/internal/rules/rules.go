package rules

import (
	"errors"
	"fmt"
	"reflect"
)

type RuleStatus string

const (
	RuleStatusDraft         RuleStatus = "draft"
	RuleStatusPendingReview RuleStatus = "pending_review"
	RuleStatusApproved      RuleStatus = "approved"
	RuleStatusActive        RuleStatus = "active"
	RuleStatusShadow        RuleStatus = "shadow"
	RuleStatusDisabled      RuleStatus = "disabled"
	RuleStatusArchived      RuleStatus = "archived"
)

type Rule struct {
	ID       string
	Field    string
	Op       string
	Value    any
	Action   string
	Score    *int
	Severity string
	Reason   string
	Status   RuleStatus
}

type RuleSet struct {
	Version string
	Rules   []Rule
}

type Explanation struct {
	RuleID   string
	Severity string
	Reason   string
}

type RuleResult struct {
	FinalScore         int
	MatchedRules       []string
	Explanations       []Explanation
	Rejected           bool
	ShadowMatchedRules []string
	ShadowExplanations []Explanation
}

func EvaluateRules(features map[string]any, currentScore int, ruleset RuleSet) (RuleResult, error) {
	if len(ruleset.Rules) == 0 {
		return RuleResult{
			FinalScore:         currentScore,
			MatchedRules:       []string{},
			Explanations:       []Explanation{},
			ShadowMatchedRules: []string{},
			ShadowExplanations: []Explanation{},
		}, nil
	}

	score := currentScore
	matched := []string{}
	explanations := []Explanation{}
	shadowMatched := []string{}
	shadowExplanations := []Explanation{}
	rejected := false
	overrideApplied := false

	activeRules, shadowRules := splitRules(ruleset.Rules)

	for _, rule := range activeRules {
		featureValue, ok := features[rule.Field]
		if !ok {
			continue
		}

		matches, err := evaluateCondition(rule.Op, featureValue, rule.Value)
		if err != nil || !matches {
			continue
		}

		matched = append(matched, rule.ID)
		explanations = append(explanations, Explanation{
			RuleID:   rule.ID,
			Severity: defaultSeverity(rule.Severity),
			Reason:   defaultReason(rule.Reason, fmt.Sprintf("rule_matched:%s", rule.ID)),
		})

		switch rule.Action {
		case "reject":
			rejected = true
			score = 99
		case "override_score":
			if !overrideApplied {
				if rule.Score == nil {
					return RuleResult{}, errors.New("override_score requires score")
				}
				score = *rule.Score
				overrideApplied = true
			}
		case "clamp_min":
			if !overrideApplied {
				if rule.Score == nil {
					return RuleResult{}, errors.New("clamp_min requires score")
				}
				if score < *rule.Score {
					score = *rule.Score
				}
			}
		case "clamp_max":
			if !overrideApplied {
				if rule.Score == nil {
					return RuleResult{}, errors.New("clamp_max requires score")
				}
				if score > *rule.Score {
					score = *rule.Score
				}
			}
		}
	}

	for _, rule := range shadowRules {
		featureValue, ok := features[rule.Field]
		if !ok {
			continue
		}

		matches, err := evaluateCondition(rule.Op, featureValue, rule.Value)
		if err != nil || !matches {
			continue
		}

		shadowMatched = append(shadowMatched, rule.ID)
		shadowExplanations = append(shadowExplanations, Explanation{
			RuleID:   rule.ID,
			Severity: defaultSeverity(rule.Severity),
			Reason:   defaultReason(rule.Reason, fmt.Sprintf("shadow_rule_matched:%s", rule.ID)),
		})
	}

	score = clampScore(score)

	return RuleResult{
		FinalScore:         score,
		MatchedRules:       matched,
		Explanations:       explanations,
		Rejected:           rejected,
		ShadowMatchedRules: shadowMatched,
		ShadowExplanations: shadowExplanations,
	}, nil
}

func splitRules(rules []Rule) (active []Rule, shadow []Rule) {
	for _, rule := range rules {
		switch rule.Status {
		case RuleStatusActive:
			active = append(active, rule)
		case RuleStatusShadow:
			shadow = append(shadow, rule)
		}
	}
	return active, shadow
}

func clampScore(score int) int {
	if score < 1 {
		return 1
	}
	if score > 99 {
		return 99
	}
	return score
}

func defaultSeverity(severity string) string {
	if severity == "" {
		return "medium"
	}
	return severity
}

func defaultReason(reason, fallback string) string {
	if reason == "" {
		return fallback
	}
	return reason
}

func evaluateCondition(op string, featureValue any, ruleValue any) (bool, error) {
	switch op {
	case ">", ">=", "<", "<=":
		fVal, ok1 := toFloat(featureValue)
		rVal, ok2 := toFloat(ruleValue)
		if !ok1 || !ok2 {
			return false, fmt.Errorf("non-numeric comparison")
		}
		switch op {
		case ">":
			return fVal > rVal, nil
		case ">=":
			return fVal >= rVal, nil
		case "<":
			return fVal < rVal, nil
		case "<=":
			return fVal <= rVal, nil
		}
	case "==":
		if isNumber(featureValue) && isNumber(ruleValue) {
			fVal, _ := toFloat(featureValue)
			rVal, _ := toFloat(ruleValue)
			return fVal == rVal, nil
		}
		return reflect.DeepEqual(featureValue, ruleValue), nil
	case "in", "not_in":
		in, err := containsValue(ruleValue, featureValue)
		if err != nil {
			return false, err
		}
		if op == "in" {
			return in, nil
		}
		return !in, nil
	default:
		return false, fmt.Errorf("unknown operator: %s", op)
	}
	return false, fmt.Errorf("unsupported operator: %s", op)
}

func containsValue(ruleValue any, featureValue any) (bool, error) {
	rv := reflect.ValueOf(ruleValue)
	if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
		return false, fmt.Errorf("rule value is not a list")
	}
	for i := 0; i < rv.Len(); i++ {
		item := rv.Index(i).Interface()
		if isNumber(item) && isNumber(featureValue) {
			fItem, _ := toFloat(item)
			fFeature, _ := toFloat(featureValue)
			if fItem == fFeature {
				return true, nil
			}
			continue
		}
		if reflect.DeepEqual(item, featureValue) {
			return true, nil
		}
	}
	return false, nil
}

func toFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint8:
		return float64(v), true
	case uint16:
		return float64(v), true
	case uint32:
		return float64(v), true
	case uint64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
	}
}

func isNumber(value any) bool {
	_, ok := toFloat(value)
	return ok
}
