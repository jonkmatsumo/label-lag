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
	ID       string     `json:"id"`
	Field    string     `json:"field"`
	Op       string     `json:"op"`
	Value    any        `json:"value"`
	Action   string     `json:"action"`
	Score    *int       `json:"score,omitempty"`
	Severity string     `json:"severity"`
	Reason   string     `json:"reason"`
	Status   RuleStatus `json:"status"`
}

type RuleSet struct {
	Version string `json:"version"`
	Rules   []Rule `json:"rules"`
}

type Explanation struct {
	RuleID      string `json:"rule_id"`
	Severity    string `json:"severity"`
	Reason      string `json:"reason"`
	Explanation string `json:"explanation"`
	Action      string `json:"action"`
	Score       *int   `json:"score,omitempty"`
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
			RuleID:      rule.ID,
			Severity:    defaultSeverity(rule.Severity),
			Reason:      defaultReason(rule.Reason, fmt.Sprintf("rule_matched:%s", rule.ID)),
			Explanation: rule.Reason,
			Action:      rule.Action,
			Score:       rule.Score,
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
			RuleID:      rule.ID,
			Severity:    defaultSeverity(rule.Severity),
			Reason:      defaultReason(rule.Reason, fmt.Sprintf("shadow_rule_matched:%s", rule.ID)),
			Explanation: rule.Reason,
			Action:      rule.Action,
			Score:       rule.Score,
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

func ValidateRule(rule Rule) error {
	if rule.ID == "" {
		return errors.New("rule id is required")
	}
	if rule.Field == "" {
		return errors.New("rule field is required")
	}
	if rule.Value == nil {
		return errors.New("rule value is required")
	}
	switch rule.Op {
	case ">", ">=", "<", "<=":
		if _, ok := toFloat(rule.Value); !ok {
			return fmt.Errorf("rule value must be numeric for op %s", rule.Op)
		}
	case "==":
		// Any value is allowed.
	case "in", "not_in":
		val := reflect.ValueOf(rule.Value)
		if val.Kind() != reflect.Slice && val.Kind() != reflect.Array {
			return fmt.Errorf("rule value must be a list for op %s", rule.Op)
		}
	default:
		return fmt.Errorf("invalid rule op %s", rule.Op)
	}

	switch rule.Action {
	case "reject":
	case "override_score", "clamp_min", "clamp_max":
		if rule.Score == nil {
			return fmt.Errorf("rule action %s requires score", rule.Action)
		}
		if *rule.Score < 1 || *rule.Score > 99 {
			return fmt.Errorf("rule score must be between 1 and 99")
		}
	default:
		return fmt.Errorf("invalid rule action %s", rule.Action)
	}

	switch rule.Status {
	case RuleStatusActive, RuleStatusShadow, RuleStatusDraft, RuleStatusPendingReview, RuleStatusApproved,
		RuleStatusDisabled, RuleStatusArchived:
	default:
		return fmt.Errorf("invalid rule status %s", rule.Status)
	}

	return nil
}

func FilterValidRules(rules []Rule) ([]Rule, []error) {
	valid := make([]Rule, 0, len(rules))
	var errs []error
	for _, rule := range rules {
		if err := ValidateRule(rule); err != nil {
			errs = append(errs, fmt.Errorf("rule %s: %w", rule.ID, err))
			continue
		}
		valid = append(valid, rule)
	}
	return valid, errs
}
