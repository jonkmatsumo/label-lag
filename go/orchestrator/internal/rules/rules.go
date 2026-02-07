package rules

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
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
	// splitOnce ensures rules are split only once (R1).
	splitOnce   sync.Once `json:"-"`
	activeRules []Rule    `json:"-"`
	shadowRules []Rule    `json:"-"`
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
	RuleTraces         []RuleTrace        `json:"rule_traces,omitempty"`
}
type EvalOptions struct {
	Debug   bool
	Explain bool
}

type RuleTrace struct {
	RuleID     string           `json:"rule_id"`
	Matched    bool             `json:"matched"`
	EvalTimeMS float64          `json:"eval_time_ms"`
	Conditions []ConditionTrace `json:"conditions"`
}

type ConditionTrace struct {
	Field               string `json:"field"`
	Operator            string `json:"operator"`
	ExpectedType        string `json:"expected_type"`
	ActualType          string `json:"actual_type"`
	Result              bool   `json:"result"`
	ActualValueRedacted string `json:"actual_value_redacted"`
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
	// Pre-allocate assuming worst case (all match) to avoid resizes
	matched := make([]string, 0, len(ruleset.Rules))
	explanations := make([]Explanation, 0, len(ruleset.Rules))
	shadowMatched := make([]string, 0, 5) // Shadow rules are typically fewer
	shadowExplanations := make([]Explanation, 0, 5)
	var traces []RuleTrace
	if opts.Explain {
		traces = make([]RuleTrace, 0, len(ruleset.Rules))
	}
	rejected := false
	overrideApplied := false
	var perRuleTimings map[string]float64
	if opts.Debug {
		perRuleTimings = make(map[string]float64)
	}
	activeRules, shadowRules := ruleset.Split()
	for _, rule := range activeRules {
		var startRule time.Time
		if opts.Debug || opts.Explain {
			startRule = time.Now()
		}
		featureValue, ok := features[rule.Field]
		if !ok {
			if opts.Debug {
				perRuleTimings[rule.ID] = float64(time.Since(startRule).Nanoseconds()) / 1e6
			}
			if opts.Explain {
				traces = append(traces, RuleTrace{
					RuleID:     rule.ID,
					Matched:    false,
					EvalTimeMS: float64(time.Since(startRule).Nanoseconds()) / 1e6,
					Conditions: []ConditionTrace{{
						Field:        rule.Field,
						Result:       false,
						ActualType:   "missing",
						ExpectedType: fmt.Sprintf("%T", rule.Value),
					}},
				})
			}
			continue
		}
		matches, err := evaluateCondition(rule.Op, featureValue, rule.Value)

		if opts.Explain {
			traces = append(traces, RuleTrace{
				RuleID:     rule.ID,
				Matched:    matches,
				EvalTimeMS: float64(time.Since(startRule).Nanoseconds()) / 1e6,
				Conditions: []ConditionTrace{{
					Field:        rule.Field,
					Operator:     rule.Op,
					ExpectedType: fmt.Sprintf("%T", rule.Value),
					ActualType:   fmt.Sprintf("%T", featureValue),
					Result:       matches,
				}},
			})
		}

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
		// Shadow rules processing (simplified for explain, or should we explain shadow too?
		// Explain often wants "why did I get this result", which implies active rules.
		// But debug tool might want all. Let's include shadow in traces if Explain is on.
		var startRule time.Time
		if opts.Debug || opts.Explain {
			startRule = time.Now()
		}
		featureValue, ok := features[rule.Field]
		if !ok {
			if opts.Debug {
				perRuleTimings[rule.ID] = float64(time.Since(startRule).Nanoseconds()) / 1e6
			}
			if opts.Explain {
				traces = append(traces, RuleTrace{
					RuleID:     rule.ID,
					Matched:    false,
					EvalTimeMS: float64(time.Since(startRule).Nanoseconds()) / 1e6,
					Conditions: []ConditionTrace{{
						Field:        rule.Field,
						Result:       false,
						ActualType:   "missing",
						ExpectedType: fmt.Sprintf("%T", rule.Value),
					}},
				})
			}
			continue
		}
		matches, err := evaluateCondition(rule.Op, featureValue, rule.Value)

		if opts.Explain {
			traces = append(traces, RuleTrace{
				RuleID:     rule.ID,
				Matched:    matches,
				EvalTimeMS: float64(time.Since(startRule).Nanoseconds()) / 1e6,
				Conditions: []ConditionTrace{{
					Field:        rule.Field,
					Operator:     rule.Op,
					ExpectedType: fmt.Sprintf("%T", rule.Value),
					ActualType:   fmt.Sprintf("%T", featureValue),
					Result:       matches,
				}},
			})
		}

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
		RuleTraces:         traces,
	}, nil
}
func (rs *RuleSet) Split() (active []Rule, shadow []Rule) {
	if rs == nil {
		return nil, nil
	}
	rs.splitOnce.Do(func() {
		for _, rule := range rs.Rules {
			switch rule.Status {
			case RuleStatusActive:
				rs.activeRules = append(rs.activeRules, rule)
			case RuleStatusShadow:
				rs.shadowRules = append(rs.shadowRules, rule)
			}
		}
	})
	return rs.activeRules, rs.shadowRules
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
			return false, fmt.Errorf("non-numeric comparison between %T and %T", featureValue, ruleValue)
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

			// Float equality check with epsilon
			return math.Abs(fVal-rVal) < 1e-9, nil
		}
		// String comparison optimization
		if s1, ok1 := featureValue.(string); ok1 {
			if s2, ok2 := ruleValue.(string); ok2 {
				return s1 == s2, nil
			}
		}
		// Bool comparison optimization
		if b1, ok1 := featureValue.(bool); ok1 {
			if b2, ok2 := ruleValue.(bool); ok2 {
				return b1 == b2, nil
			}
		}
		// Fallback only if types match perfectly
		return featureValue == ruleValue, nil
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
	// Fast path for string slice
	if sSlice, ok := ruleValue.([]string); ok {
		sVal, ok := featureValue.(string)
		if !ok {
			// Try converting featureValue to string?
			// For strictness, if rule is []string, feature must be string
			return false, nil
		}
		for _, s := range sSlice {
			if s == sVal {
				return true, nil
			}
		}
		return false, nil
	}
	// Fast path for []any
	if aSlice, ok := ruleValue.([]any); ok {
		// If feature is number, check numeric equality
		if isNumber(featureValue) {
			fFeature, _ := toFloat(featureValue)
			for _, item := range aSlice {
				if isNumber(item) {
					fItem, _ := toFloat(item)
					if math.Abs(fItem-fFeature) < 1e-9 {
						return true, nil
					}
				}
			}
			return false, nil
		}
		// Generic equality
		for _, item := range aSlice {
			if item == featureValue {
				return true, nil
			}
		}
		return false, nil
	}
	// Fallback to error if not a slice we handle
	// For JSON unmarshaling, []any is common.
	return false, fmt.Errorf("rule value must be a list (got %T)", ruleValue)
}
func toFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case int32:
		return float64(v), true
	case float32:
		return float64(v), true
	case int16:
		return float64(v), true
	case int8:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint64:
		return float64(v), true
	case uint32:
		return float64(v), true
	case uint16:
		return float64(v), true
	case uint8:
		return float64(v), true
	default:
		return 0, false
	}
}
func isNumber(value any) bool {
	switch value.(type) {
	case float64, int, int64, int32, float32, int16, int8, uint, uint64, uint32, uint16, uint8:
		return true
	default:
		return false
	}
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
		switch rule.Value.(type) {
		case []any, []string, []int, []float64:
			// ok
		default:
			return fmt.Errorf("rule value must be a list for op %s (got %T)", rule.Op, rule.Value)
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
