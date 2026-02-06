package rules

import (
	"fmt"
	"sort"
	"strings"
)

type ConflictKind string

const (
	ConflictKindSameOp       ConflictKind = "same_op"
	ConflictKindRangeOverlap ConflictKind = "range_overlap"
	ConflictKindActionType   ConflictKind = "action_type_clash"
	ConflictKindShadowActive ConflictKind = "shadow_active_overlap"
)

type ConflictSeverity string

const (
	ConflictSeverityBreaking ConflictSeverity = "breaking"
	ConflictSeverityWarning  ConflictSeverity = "warning"
	ConflictSeverityInfo     ConflictSeverity = "info"
)

type Conflict struct {
	Kind     ConflictKind     `json:"kind"`
	Field    string           `json:"field"`
	RuleIDs  []string         `json:"rule_ids"`
	Details  string           `json:"details"`
	Severity ConflictSeverity `json:"severity"`
}

func ValidateRuleset(rs RuleSet) []Conflict {
	var conflicts []Conflict

	// Group rules by field
	rulesByField := make(map[string][]Rule)
	for _, rule := range rs.Rules {
		// Only consider rules that are potentially "live" (Active or Shadow)
		if rule.Status == RuleStatusActive || rule.Status == RuleStatusShadow {
			rulesByField[rule.Field] = append(rulesByField[rule.Field], rule)
		}
	}

	for field, rules := range rulesByField {
		for i := 0; i < len(rules); i++ {
			for j := i + 1; j < len(rules); j++ {
				r1, r2 := rules[i], rules[j]
				if r1.ID == r2.ID {
					continue
				}

				if c, ok := detectConflict(field, r1, r2); ok {
					conflicts = append(conflicts, c)
				}
			}
		}
	}

	// Sort conflicts by field, kind, then joined rule_ids
	sort.Slice(conflicts, func(i, j int) bool {
		if conflicts[i].Field != conflicts[j].Field {
			return conflicts[i].Field < conflicts[j].Field
		}
		if conflicts[i].Kind != conflicts[j].Kind {
			return conflicts[i].Kind < conflicts[j].Kind
		}
		return strings.Join(conflicts[i].RuleIDs, ",") < strings.Join(conflicts[j].RuleIDs, ",")
	})

	return conflicts
}

func detectConflict(field string, r1, r2 Rule) (Conflict, bool) {
	ruleIDs := []string{r1.ID, r2.ID}
	sort.Strings(ruleIDs)

	// 1. Same-field same-op conflicts
	if r1.Op == r2.Op {
		severity := ConflictSeverityWarning
		if (r1.Action == "reject" && r2.Action != "reject") || (r2.Action == "reject" && r1.Action != "reject") {
			severity = ConflictSeverityBreaking
		}
		return Conflict{
			Kind:     ConflictKindSameOp,
			Field:    field,
			RuleIDs:  ruleIDs,
			Details:  fmt.Sprintf("Rules share same field and operator (%s)", r1.Op),
			Severity: severity,
		}, true
	}

	// 2. Range overlap conflicts (Simplified numeric range check)
	overlap := false
	v1, ok1 := toFloat(r1.Value)
	v2, ok2 := toFloat(r2.Value)

	if ok1 && ok2 {
		overlap = checkNumericOverlap(r1.Op, v1, r2.Op, v2)
	}

	if overlap {
		kind := ConflictKindRangeOverlap
		severity := ConflictSeverityWarning

		// 3. Reject vs override/clamp overlap (breaking)
		if (r1.Action == "reject" && r2.Action != "reject") || (r2.Action == "reject" && r1.Action != "reject") {
			severity = ConflictSeverityBreaking
			kind = ConflictKindActionType
		}

		// 4. Shadow vs active overlaps (lower severity)
		if (r1.Status == RuleStatusShadow && r2.Status == RuleStatusActive) || (r2.Status == RuleStatusShadow && r1.Status == RuleStatusActive) {
			if severity != ConflictSeverityBreaking {
				severity = ConflictSeverityInfo
				kind = ConflictKindShadowActive
			}
		}

		return Conflict{
			Kind:     kind,
			Field:    field,
			RuleIDs:  ruleIDs,
			Details:  fmt.Sprintf("Rules have overlapping ranges on field %s", field),
			Severity: severity,
		}, true
	}

	return Conflict{}, false
}

func checkNumericOverlap(op1 string, v1 float64, op2 string, v2 float64) bool {
	// This is a simplified check.
	// e.g. > 10 and < 5 -> no overlap
	// > 10 and < 15 -> overlap (10, 15)
	
	// Normalize so op1 is one of {>, >=} and op2 is one of {<, <=} if possible
	if (op1 == "<" || op1 == "<=") && (op2 == ">" || op2 == ">=") {
		op1, op2 = op2, op1
		v1, v2 = v2, v1
	}

	if (op1 == ">" || op1 == ">=") && (op2 == "<" || op2 == "<=") {
		if op1 == ">" && op2 == "<" {
			return v1 < v2
		}
		return v1 <= v2
	}

	// Same direction: always some overlap if they are both > or both <
	if (op1 == ">" || op1 == ">=") && (op2 == ">" || op2 == ">=") {
		return true
	}
	if (op1 == "<" || op1 == "<=") && (op2 == "<" || op2 == "<=") {
		return true
	}

	// == checks
	if op1 == "==" {
		return matches(op2, v1, v2)
	}
	if op2 == "==" {
		return matches(op1, v2, v1)
	}

	return false
}

func matches(op string, val float64, threshold float64) bool {
	switch op {
	case ">":
		return val > threshold
	case ">=":
		return val >= threshold
	case "<":
		return val < threshold
	case "<=":
		return val <= threshold
	case "==":
		return val == threshold
	}
	return false
}
