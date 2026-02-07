package rules

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"time"
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

	// DefaultMaxRules is the maximum number of rules allowed in a ruleset (R4).
	DefaultMaxRules = 500
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

	// versionOnce ensures version is only computed once (R1).
	versionOnce sync.Once `json:"-"`
	// computedVersion caches the computed version string (R1).
	computedVersion string `json:"-"`
}

func (rs *RuleSet) ComputeVersion() string {
	if len(rs.Rules) == 0 {
		return rs.Version
	}
	rs.versionOnce.Do(func() {
		// Canonical JSON serialization
		data, err := json.Marshal(rs.Rules)
		if err != nil {
			rs.computedVersion = rs.Version // Fallback
			return
		}

		h := sha256.New()
		// Include explicit version in hash if present
		if rs.Version != "" {
			h.Write([]byte(rs.Version))
		}
		h.Write(data)
		rs.computedVersion = fmt.Sprintf("sha256:%x", h.Sum(nil))[:16]
	})
	return rs.computedVersion
}

type Explanation struct {
	RuleID      string `json:"rule_id"`
	Severity    string `json:"severity"`
	Reason      string `json:"reason"`
	Explanation string `json:"explanation"`
	Action      string `json:"action"`
	Score       *int   `json:"score,omitempty"`
	ScoreDelta  int    `json:"score_delta"`
}

type RuleResult struct {
	FinalScore         int
	MatchedRules       []string
	Explanations       []Explanation
	Rejected           bool
	ShadowMatchedRules []string
	ShadowExplanations []Explanation
	RulesVersion       string
	EvaluationTimeMS   float64            `json:"evaluation_time_ms"`
	PerRuleTimingsMS   map[string]float64 `json:"per_rule_timings_ms,omitempty"`
}

type EvalOptions struct {
	Debug bool
}

func EvaluateRules(features map[string]any, currentScore int, ruleset *RuleSet, opts EvalOptions) (RuleResult, error) {
	startTotal := time.Now()

	if ruleset == nil || len(ruleset.Rules) == 0 {
		version := ""
		if ruleset != nil {
			version = ruleset.ComputeVersion()
		}
		return RuleResult{
			FinalScore:         currentScore,
			MatchedRules:       []string{},
			Explanations:       []Explanation{},
			ShadowMatchedRules: []string{},
			ShadowExplanations: []Explanation{},
			RulesVersion:       version,
			EvaluationTimeMS:   float64(time.Since(startTotal).Nanoseconds()) / 1e6,
		}, nil
	}

	score := currentScore
	matched := []string{}
	explanations := []Explanation{}
	shadowMatched := []string{}
	shadowExplanations := []Explanation{}
	rejected := false
	overrideApplied := false

	var perRuleTimings map[string]float64
	if opts.Debug {
		perRuleTimings = make(map[string]float64)
	}

	activeRules, shadowRules := splitRules(ruleset.Rules)

	for _, rule := range activeRules {
		var startRule time.Time
		if opts.Debug {
			startRule = time.Now()
		}

		featureValue, ok := features[rule.Field]
		if !ok {
			if opts.Debug {
				perRuleTimings[rule.ID] = float64(time.Since(startRule).Nanoseconds()) / 1e6
			}
			continue
		}

		matches, err := evaluateCondition(rule.Op, featureValue, rule.Value)
		if err != nil || !matches {
			if opts.Debug {
				perRuleTimings[rule.ID] = float64(time.Since(startRule).Nanoseconds()) / 1e6
			}
			continue
		}

		matched = append(matched, rule.ID)
		beforeScore := score

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

		explanations = append(explanations, Explanation{
			RuleID:      rule.ID,
			Severity:    defaultSeverity(rule.Severity),
			Reason:      defaultReason(rule.Reason, fmt.Sprintf("rule_matched:%s", rule.ID)),
			Explanation: rule.Reason,
			Action:      rule.Action,
			Score:       rule.Score,
			ScoreDelta:  score - beforeScore,
		})

		if opts.Debug {
			perRuleTimings[rule.ID] = float64(time.Since(startRule).Nanoseconds()) / 1e6
		}
	}

	for _, rule := range shadowRules {
		var startRule time.Time
		if opts.Debug {
			startRule = time.Now()
		}

		featureValue, ok := features[rule.Field]
		if !ok {
			if opts.Debug {
				perRuleTimings[rule.ID] = float64(time.Since(startRule).Nanoseconds()) / 1e6
			}
			continue
		}

		matches, err := evaluateCondition(rule.Op, featureValue, rule.Value)
		if err != nil || !matches {
			if opts.Debug {
				perRuleTimings[rule.ID] = float64(time.Since(startRule).Nanoseconds()) / 1e6
			}
			continue
		}

		shadowMatched = append(shadowMatched, rule.ID)

		// Compute shadow delta without modifying final score
		shadowDelta := 0
		switch rule.Action {
		case "reject":
			shadowDelta = 99 - score
		case "override_score":
			if rule.Score != nil {
				shadowDelta = *rule.Score - score
			}
		case "clamp_min":
			if rule.Score != nil && score < *rule.Score {
				shadowDelta = *rule.Score - score
			}
		case "clamp_max":
			if rule.Score != nil && score > *rule.Score {
				shadowDelta = *rule.Score - score
			}
		}

		shadowExplanations = append(shadowExplanations, Explanation{
			RuleID:      rule.ID,
			Severity:    defaultSeverity(rule.Severity),
			Reason:      defaultReason(rule.Reason, fmt.Sprintf("shadow_rule_matched:%s", rule.ID)),
			Explanation: rule.Reason,
			Action:      rule.Action,
			Score:       rule.Score,
			ScoreDelta:  shadowDelta,
		})

		if opts.Debug {
			perRuleTimings[rule.ID] = float64(time.Since(startRule).Nanoseconds()) / 1e6
		}
	}

	score = clampScore(score)

	return RuleResult{
		FinalScore:         score,
		MatchedRules:       matched,
		Explanations:       explanations,
		Rejected:           rejected,
		ShadowMatchedRules: shadowMatched,
		ShadowExplanations: shadowExplanations,
		RulesVersion:       ruleset.ComputeVersion(),
		EvaluationTimeMS:   float64(time.Since(startTotal).Nanoseconds()) / 1e6,
		PerRuleTimingsMS:   perRuleTimings,
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
	if len(rules) > DefaultMaxRules {
		return nil, []error{fmt.Errorf("ruleset exceeds maximum rules limit of %d", DefaultMaxRules)}
	}

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
