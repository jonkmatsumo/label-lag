package httpserver

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type EvaluateRulesRequest struct {
	Features   map[string]any `json:"features"`
	BaseScore  int            `json:"base_score"`
	RuleSet    *rules.RuleSet `json:"ruleset,omitempty"`
	ShadowMode bool           `json:"shadow_mode"`
}

type EvaluateRulesResponse struct {
	FinalScore         int                 `json:"final_score"`
	BaselineScore      int                 `json:"baseline_score"`
	ShadowScore        int                 `json:"shadow_score,omitempty"`
	MatchedRules       []string            `json:"matched_rules"`
	Explanations       []rules.Explanation `json:"explanations"`
	ShadowMatchedRules []string            `json:"shadow_matched_rules"`
	ShadowExplanations []rules.Explanation `json:"shadow_explanations"`
	Rejected           bool                `json:"rejected"`
	RuleSetVersion     string              `json:"ruleset_version"`
	Warnings           []rules.Conflict    `json:"warnings,omitempty"`
	EvaluationTimeMS   float64             `json:"evaluation_time_ms"`
	PerRuleTimingsMS   map[string]float64  `json:"per_rule_timings_ms,omitempty"`
}

type EvaluateRulesDiffRequest struct {
	Features   map[string]any `json:"features"`
	BaseScore  int            `json:"base_score"`
	RuleSetA   *rules.RuleSet `json:"ruleset_a,omitempty"`
	RuleSetB   *rules.RuleSet `json:"ruleset_b,omitempty"`
	ShadowMode bool           `json:"shadow_mode"`
}

type RulesDiffSummary struct {
	Severity            string   `json:"severity"`
	ScoreDelta          int      `json:"score_delta"`
	MatchedRulesAdded   []string `json:"matched_rules_added"`
	MatchedRulesRemoved []string `json:"matched_rules_removed"`
}

type EvaluateRulesDiffResponse struct {
	A                     EvaluateRulesResponse `json:"a"`
	B                     EvaluateRulesResponse `json:"b"`
	Diff                  RulesDiffSummary      `json:"diff"`
	TotalEvaluationTimeMS float64               `json:"total_evaluation_time_ms"`
}

type SandboxMatchedRule struct {
	RuleID   string `json:"rule_id"`
	Severity string `json:"severity"`
	Reason   string `json:"reason"`
	Action   string `json:"action"`
	Score    *int   `json:"score"`
}

type SandboxEvaluateResponse struct {
	FinalScore         int                  `json:"final_score"`
	BaselineScore      int                  `json:"baseline_score"`
	ShadowScore        *int                 `json:"shadow_score"`
	MatchedRules       []SandboxMatchedRule `json:"matched_rules"`
	Explanations       []rules.Explanation  `json:"explanations"`
	ShadowMatchedRules []SandboxMatchedRule `json:"shadow_matched_rules"`
	Rejected           bool                 `json:"rejected"`
	RuleSetVersion     string               `json:"ruleset_version"`
}

type SandboxDiffResponse struct {
	A    SandboxEvaluateResponse `json:"a"`
	B    SandboxEvaluateResponse `json:"b"`
	Diff RulesDiffSummary        `json:"diff"`
}

func (h *Handler) handleSandboxEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	debug := r.URL.Query().Get("debug") == "true"

	var req EvaluateRulesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	ruleset := rules.RuleSet{}
	if req.RuleSet != nil {
		validRules, errs := rules.FilterValidRules(req.RuleSet.Rules)
		if len(errs) > 0 {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid rules: %v", errs))
			return
		}
		ruleset = *req.RuleSet
		ruleset.Rules = validRules
	} else {
		var err error
		ruleset, err = h.rulesProvider.GetRules(r.Context())
		if err != nil {
			h.logger.Warn("failed to load default ruleset", "error", err)
		}
	}

	result, err := rules.EvaluateRules(req.Features, req.BaseScore, &ruleset, rules.EvalOptions{Debug: debug})
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "rule evaluation failed")
		return
	}

	resp := mapToSandboxEvaluateResponse(result, req.BaseScore, req.ShadowMode)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *Handler) handleSandboxDiff(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	debug := r.URL.Query().Get("debug") == "true"

	var req EvaluateRulesDiffRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	// Helper to evaluate a single ruleset
	eval := func(rs *rules.RuleSet) (rules.RuleResult, error) {
		ruleset := rules.RuleSet{}
		if rs != nil {
			validRules, errs := rules.FilterValidRules(rs.Rules)
			if len(errs) > 0 {
				return rules.RuleResult{}, fmt.Errorf("invalid rules: %v", errs)
			}
			ruleset = *rs
			ruleset.Rules = validRules
		} else {
			var err error
			ruleset, err = h.rulesProvider.GetRules(r.Context())
			if err != nil {
				h.logger.Warn("failed to load default ruleset", "error", err)
			}
		}

		return rules.EvaluateRules(req.Features, req.BaseScore, &ruleset, rules.EvalOptions{Debug: debug})
	}

	resA, err := eval(req.RuleSetA)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	resB, err := eval(req.RuleSetB)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Compute diff summary (reusing existing logic from EvaluateRulesDiff but returning Sandbox version)
	diff := RulesDiffSummary{
		ScoreDelta: resB.FinalScore - resA.FinalScore,
	}
	if req.ShadowMode {
		// Use internal FinalScore from results which is the adjusted score even in shadow mode
		diff.ScoreDelta = resB.FinalScore - resA.FinalScore
	}

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

	// Use temporary EvaluateRulesResponse for severity computation
	tempRespA := EvaluateRulesResponse{Rejected: resA.Rejected, Explanations: resA.Explanations}
	tempRespB := EvaluateRulesResponse{Rejected: resB.Rejected, Explanations: resB.Explanations}
	diff.Severity = computeDiffSeverity(tempRespA, tempRespB, diff)

	resp := SandboxDiffResponse{
		A:    mapToSandboxEvaluateResponse(resA, req.BaseScore, req.ShadowMode),
		B:    mapToSandboxEvaluateResponse(resB, req.BaseScore, req.ShadowMode),
		Diff: diff,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

func mapToSandboxEvaluateResponse(result rules.RuleResult, baseScore int, shadowMode bool) SandboxEvaluateResponse {
	matchedRules := make([]SandboxMatchedRule, 0, len(result.Explanations))
	for _, exp := range result.Explanations {
		matchedRules = append(matchedRules, SandboxMatchedRule{
			RuleID:   exp.RuleID,
			Severity: exp.Severity,
			Reason:   exp.Reason,
			Action:   exp.Action,
			Score:    exp.Score,
		})
	}

	shadowMatchedRules := make([]SandboxMatchedRule, 0, len(result.ShadowExplanations))
	for _, exp := range result.ShadowExplanations {
		shadowMatchedRules = append(shadowMatchedRules, SandboxMatchedRule{
			RuleID:   exp.RuleID,
			Severity: exp.Severity,
			Reason:   exp.Reason,
			Action:   exp.Action,
			Score:    exp.Score,
		})
	}

	finalScore := result.FinalScore
	var shadowScore *int
	if shadowMode {
		sScore := result.FinalScore
		shadowScore = &sScore
		finalScore = baseScore
	}

	return SandboxEvaluateResponse{
		FinalScore:         finalScore,
		BaselineScore:      baseScore,
		ShadowScore:        shadowScore,
		MatchedRules:       matchedRules,
		Explanations:       result.Explanations,
		ShadowMatchedRules: shadowMatchedRules,
		Rejected:           result.Rejected && !shadowMode,
		RuleSetVersion:     result.RulesVersion,
	}
}

func (h *Handler) handleEvaluateRulesDiff(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	debug := r.URL.Query().Get("debug") == "true"

	var req EvaluateRulesDiffRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	// Helper to evaluate a single ruleset
	eval := func(rs *rules.RuleSet) (EvaluateRulesResponse, error) {
		ruleset := rules.RuleSet{}
		var warnings []rules.Conflict
		if rs != nil {
			validRules, errs := rules.FilterValidRules(rs.Rules)
			if len(errs) > 0 {
				return EvaluateRulesResponse{}, fmt.Errorf("invalid rules: %v", errs)
			}
			ruleset = *rs
			ruleset.Rules = validRules
			warnings = rules.ValidateRuleset(ruleset)
		} else {
			var err error
			ruleset, err = h.rulesProvider.GetRules(r.Context())
			if err != nil {
				h.logger.Warn("failed to load default ruleset", "error", err)
			}
		}

		result, err := rules.EvaluateRules(req.Features, req.BaseScore, &ruleset, rules.EvalOptions{Debug: debug})
		if err != nil {
			return EvaluateRulesResponse{}, err
		}

		resp := EvaluateRulesResponse{
			FinalScore:         result.FinalScore,
			BaselineScore:      req.BaseScore,
			MatchedRules:       result.MatchedRules,
			Explanations:       result.Explanations,
			ShadowMatchedRules: result.ShadowMatchedRules,
			ShadowExplanations: result.ShadowExplanations,
			Rejected:           result.Rejected,
			RuleSetVersion:     result.RulesVersion,
			Warnings:           warnings,
			EvaluationTimeMS:   result.EvaluationTimeMS,
			PerRuleTimingsMS:   result.PerRuleTimingsMS,
		}

		if req.ShadowMode {
			resp.ShadowScore = result.FinalScore
			resp.FinalScore = req.BaseScore
			resp.Rejected = false
		}
		return resp, nil
	}

	respA, err := eval(req.RuleSetA)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	respB, err := eval(req.RuleSetB)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	requestID := requestid.FromContext(r.Context())
	span := trace.SpanFromContext(r.Context())

	// Compute diff
	diff := RulesDiffSummary{
		ScoreDelta: respB.FinalScore - respA.FinalScore,
	}
	if req.ShadowMode {
		diff.ScoreDelta = respB.ShadowScore - respA.ShadowScore
	}

	// Matched rules diff
	mapA := make(map[string]bool)
	for _, r := range respA.MatchedRules {
		mapA[r] = true
	}
	mapB := make(map[string]bool)
	for _, r := range respB.MatchedRules {
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

	// Compute severity
	diff.Severity = computeDiffSeverity(respA, respB, diff)

	totalEvalTime := respA.EvaluationTimeMS + respB.EvaluationTimeMS

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	// Logging and Tracing
	h.logger.Info("RulesDiffEvent",
		"request_id", requestID,
		"version_a", respA.RuleSetVersion,
		"version_b", respB.RuleSetVersion,
		"score_delta", diff.ScoreDelta,
		"added_count", len(diff.MatchedRulesAdded),
		"removed_count", len(diff.MatchedRulesRemoved),
		"shadow_mode", req.ShadowMode,
		"total_evaluation_time_ms", totalEvalTime,
	)

	span.SetAttributes(
		attribute.String("rules.request_id", requestID),
		attribute.String("rules.version.a", respA.RuleSetVersion),
		attribute.String("rules.version.b", respB.RuleSetVersion),
		attribute.Int("rules.score_delta", diff.ScoreDelta),
		attribute.Bool("rules.diff", true),
		attribute.Bool("rules.shadow_mode", req.ShadowMode),
		attribute.Float64("rules.evaluation_time_ms.total", totalEvalTime),
		attribute.Float64("rules.evaluation_time_ms.a", respA.EvaluationTimeMS),
		attribute.Float64("rules.evaluation_time_ms.b", respB.EvaluationTimeMS),
	)

	_ = json.NewEncoder(w).Encode(EvaluateRulesDiffResponse{
		A:                     respA,
		B:                     respB,
		Diff:                  diff,
		TotalEvaluationTimeMS: totalEvalTime,
	})
}


func (h *Handler) handleEvaluateRules(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	debug := r.URL.Query().Get("debug") == "true"

	var req EvaluateRulesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	ruleset := rules.RuleSet{}
	var warnings []rules.Conflict
	if req.RuleSet != nil {
		validRules, errs := rules.FilterValidRules(req.RuleSet.Rules)
		if len(errs) > 0 {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid rules: %v", errs))
			return
		}
		ruleset = *req.RuleSet
		ruleset.Rules = validRules
		warnings = rules.ValidateRuleset(ruleset)
	} else {
		// Use default ruleset if not provided
		var err error
		ruleset, err = h.rulesProvider.GetRules(r.Context())
		if err != nil {
			h.logger.Warn("failed to load default ruleset", "error", err)
		}
	}

	result, err := rules.EvaluateRules(req.Features, req.BaseScore, &ruleset, rules.EvalOptions{Debug: debug})
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "rule evaluation failed")
		return
	}

	requestID := requestid.FromContext(r.Context())
	span := trace.SpanFromContext(r.Context())

	resp := EvaluateRulesResponse{
		FinalScore:         result.FinalScore,
		BaselineScore:      req.BaseScore,
		MatchedRules:       result.MatchedRules,
		Explanations:       result.Explanations,
		ShadowMatchedRules: result.ShadowMatchedRules,
		ShadowExplanations: result.ShadowExplanations,
		Rejected:           result.Rejected,
		RuleSetVersion:     result.RulesVersion,
		Warnings:           warnings,
		EvaluationTimeMS:   result.EvaluationTimeMS,
		PerRuleTimingsMS:   result.PerRuleTimingsMS,
	}

	if req.ShadowMode {
		resp.ShadowScore = result.FinalScore
		resp.FinalScore = req.BaseScore
		resp.Rejected = false
	}

	// Logging and Tracing
	h.logger.Info("RulesEvent",
		"request_id", requestID,
		"rules_version", result.RulesVersion,
		"baseline_score", req.BaseScore,
		"final_score", resp.FinalScore,
		"shadow_mode", req.ShadowMode,
		"shadow_score", resp.ShadowScore,
		"matches", len(result.MatchedRules),
		"evaluation_time_ms", result.EvaluationTimeMS,
	)

	span.SetAttributes(
		attribute.String("rules.request_id", requestID),
		attribute.String("rules.version", result.RulesVersion),
		attribute.Int("rules.baseline_score", req.BaseScore),
		attribute.Int("rules.final_score", resp.FinalScore),
		attribute.Int("rules.match_count", len(result.MatchedRules)),
		attribute.Bool("rules.shadow_mode", req.ShadowMode),
		attribute.Float64("rules.evaluation_time_ms", result.EvaluationTimeMS),
	)
	if req.ShadowMode {
		span.SetAttributes(attribute.Int("rules.shadow_score", resp.ShadowScore))
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

func computeDiffSeverity(respA, respB EvaluateRulesResponse, diff RulesDiffSummary) string {
	// Breaking rules:
	// 1. reject added/removed
	if respA.Rejected != respB.Rejected {
		return "breaking"
	}

	// 2. action type change for same rule_id
	actionMapA := make(map[string]string)
	for _, exp := range respA.Explanations {
		actionMapA[exp.RuleID] = exp.Action
	}
	for _, exp := range respB.Explanations {
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
