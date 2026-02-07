package rules

import "sort"

type RulesDiffSummary struct {
	Severity            string   `json:"severity"`
	ScoreDelta          int      `json:"score_delta"`
	MatchedRulesAdded   []string `json:"matched_rules_added"`
	MatchedRulesRemoved []string `json:"matched_rules_removed"`
}

// ComputeDiff calculates the difference between two rule evaluation results.
// It populates ScoreDelta, MatchedRulesAdded/Removed, and Severity.
func ComputeDiff(resA, resB RuleResult, shadowMode bool) RulesDiffSummary {
	diff := RulesDiffSummary{}

	// Calculate ScoreDelta
	scoreA := resA.FinalScore
	scoreB := resB.FinalScore
	// In shadow mode, RuleResult.FinalScore is usually the *real* score (impacted by rules)
	// but the HTTP handler logic for shadow mode used:
	//   resp.ShadowScore = result.FinalScore
	//   resp.FinalScore = req.BaseScore
	//   diff.ScoreDelta = respB.ShadowScore - respA.ShadowScore
	// So we should compare the FinalScore from RuleResult as it represents the evaluated score.

	diff.ScoreDelta = scoreB - scoreA

	// Matched rules diff
	mapA := make(map[string]bool)
	for _, r := range resA.MatchedRules {
		mapA[r] = true
	}
	mapB := make(map[string]bool)
	for _, r := range resB.MatchedRules {
		mapB[r] = true
	}

	for r := range mapB {
		if !mapA[r] {
			diff.MatchedRulesAdded = append(diff.MatchedRulesAdded, r)
		}
	}
	for r := range mapA {
		if !mapB[r] {
			diff.MatchedRulesRemoved = append(diff.MatchedRulesRemoved, r)
		}
	}

	sort.Strings(diff.MatchedRulesAdded)
	sort.Strings(diff.MatchedRulesRemoved)

	diff.Severity = ComputeSeverity(resA, resB, diff)
	return diff
}

func ComputeSeverity(resA, resB RuleResult, diff RulesDiffSummary) string {
	// Breaking rules:
	// 1. reject added/removed
	if resA.Rejected != resB.Rejected {
		return "breaking"
	}

	// 2. action type change for same rule_id
	actionMapA := make(map[string]string)
	for _, exp := range resA.Explanations {
		actionMapA[exp.RuleID] = exp.Action
	}
	for _, exp := range resB.Explanations {
		if oldAction, ok := actionMapA[exp.RuleID]; ok && oldAction != exp.Action {
			return "breaking"
		}
	}

	// 3. |score_delta| > 20
	if abs(diff.ScoreDelta) > 20 {
		return "breaking"
	}

	// Behavioral rules:
	// 1. any score change
	if diff.ScoreDelta != 0 {
		return "behavioral"
	}
	// 2. matched rules added/removed
	if len(diff.MatchedRulesAdded) > 0 || len(diff.MatchedRulesRemoved) > 0 {
		return "behavioral"
	}

	// Cosmetic: no effective change
	return "cosmetic"
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
