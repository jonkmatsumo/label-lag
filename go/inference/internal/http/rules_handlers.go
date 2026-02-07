package httpserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/jonkmatsumo/label-lag/go/inference/internal/requestid"
	"github.com/jonkmatsumo/label-lag/go/inference/internal/rules"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/encoding/protojson"
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

type EvaluateRulesDiffResponse struct {
	A                     EvaluateRulesResponse  `json:"a"`
	B                     EvaluateRulesResponse  `json:"b"`
	Diff                  rules.RulesDiffSummary `json:"diff"`
	TotalEvaluationTimeMS float64                `json:"total_evaluation_time_ms"`
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
	Diff rules.RulesDiffSummary  `json:"diff"`
}

func (h *Handler) handleSandboxEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	debug := r.URL.Query().Get("debug") == "true"

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	var req EvaluateRulesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}
	defer r.Body.Close()

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

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	var req EvaluateRulesDiffRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}
	defer r.Body.Close()

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

	// Compute diff summary using shared logic
	diff := rules.ComputeDiff(resA, resB, req.ShadowMode)

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
	sort.Slice(matchedRules, func(i, j int) bool {
		return matchedRules[i].RuleID < matchedRules[j].RuleID
	})

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
	sort.Slice(shadowMatchedRules, func(i, j int) bool {
		return shadowMatchedRules[i].RuleID < shadowMatchedRules[j].RuleID
	})

	finalScore := result.FinalScore
	var shadowScore *int
	if shadowMode {
		sScore := result.FinalScore
		shadowScore = &sScore
		finalScore = baseScore
	}

	explanations := result.Explanations
	sort.Slice(explanations, func(i, j int) bool {
		return explanations[i].RuleID < explanations[j].RuleID
	})

	shadowExplanations := result.ShadowExplanations
	sort.Slice(shadowExplanations, func(i, j int) bool {
		return shadowExplanations[i].RuleID < shadowExplanations[j].RuleID
	})

	return SandboxEvaluateResponse{
		FinalScore:         finalScore,
		BaselineScore:      baseScore,
		ShadowScore:        shadowScore,
		MatchedRules:       matchedRules,
		Explanations:       explanations,
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

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	var req EvaluateRulesDiffRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}
	defer r.Body.Close()

	// Helper to evaluate a single ruleset
	eval := func(rs *rules.RuleSet) (rules.RuleResult, EvaluateRulesResponse, error) {
		ruleset := rules.RuleSet{}
		var warnings []rules.Conflict
		if rs != nil {
			validRules, errs := rules.FilterValidRules(rs.Rules)
			if len(errs) > 0 {
				return rules.RuleResult{}, EvaluateRulesResponse{}, fmt.Errorf("invalid rules: %v", errs)
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
			return rules.RuleResult{}, EvaluateRulesResponse{}, err
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
		return result, resp, nil
	}

	resultA, respA, err := eval(req.RuleSetA)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	resultB, respB, err := eval(req.RuleSetB)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	requestID := requestid.FromContext(r.Context())
	span := trace.SpanFromContext(r.Context())

	// Compute diff using shared logic
	diff := rules.ComputeDiff(resultA, resultB, req.ShadowMode)

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

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	var req EvaluateRulesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}
	defer r.Body.Close()

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

	sort.Strings(resp.MatchedRules)
	sort.Strings(resp.ShadowMatchedRules)

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

func (h *Handler) handleListRules(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	statusFilter := r.URL.Query().Get("status")
	includeArchived := r.URL.Query().Get("include_archived") == "true"

	resp, err := h.analyticsClient.ListRules(r.Context(), &crudv1.ListRulesRequest{
		Status:          statusFilter,
		IncludeArchived: includeArchived,
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	// Wrap response to match current API contract if needed, or just return list
	// Python returned: {"version": "live", "rules": [...]}
	// We'll mimic this structure using a map for now or define a struct
	response := map[string]any{
		"version": "live",
		"rules":   resp.Rules,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		writeJSONError(w, http.StatusInternalServerError, "failed to encode response")
	}
}

func (h *Handler) handleCreateRule(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	var rule crudv1.Rule
	if err := json.NewDecoder(r.Body).Decode(&rule); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}
	defer r.Body.Close()

	// Force default status if not set
	if rule.Status == "" {
		rule.Status = "draft"
	}

	// Reuse SaveRule
	_, err := h.analyticsClient.SaveRule(r.Context(), &crudv1.SaveRuleRequest{Rule: &rule})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}

func (h *Handler) handleGetRule(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ruleID := r.PathValue("rule_id")
	if ruleID == "" {
		writeJSONError(w, http.StatusBadRequest, "rule_id required")
		return
	}

	resp, err := h.analyticsClient.GetRule(r.Context(), &crudv1.GetRuleRequest{RuleId: ruleID})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	// Unwrap rule
	w.Header().Set("Content-Type", "application/json")
	// Use protojson for proper proto marshalling
	payload, _ := protojson.MarshalOptions{EmitUnpopulated: true, UseProtoNames: true}.Marshal(resp.Rule)
	w.Write(payload)
}

func (h *Handler) handleUpdateRule(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ruleID := r.PathValue("rule_id")
	if ruleID == "" {
		writeJSONError(w, http.StatusBadRequest, "rule_id required")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	var rule crudv1.Rule
	if err := json.NewDecoder(r.Body).Decode(&rule); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}
	defer r.Body.Close()

	// Ensure ID matches path
	if rule.Id != "" && rule.Id != ruleID {
		writeJSONError(w, http.StatusBadRequest, "rule_id mismatch")
		return
	}
	rule.Id = ruleID

	_, err := h.analyticsClient.SaveRule(r.Context(), &crudv1.SaveRuleRequest{Rule: &rule})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}

func (h *Handler) handleDeleteRule(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ruleID := r.PathValue("rule_id")
	if ruleID == "" {
		writeJSONError(w, http.StatusBadRequest, "rule_id required")
		return
	}

	_, err := h.analyticsClient.DeleteRule(r.Context(), &crudv1.DeleteRuleRequest{RuleId: ruleID})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}

func (h *Handler) handleListRuleVersions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ruleID := r.PathValue("rule_id")
	if ruleID == "" {
		writeJSONError(w, http.StatusBadRequest, "rule_id required")
		return
	}

	// Parse limit/offset
	limit := 100
	offset := 0

	resp, err := h.analyticsClient.ListRuleVersions(r.Context(), &crudv1.ListRuleVersionsRequest{
		RuleId: ruleID,
		Limit:  int32(limit),
		Offset: int32(offset),
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	// Use protojson for response
	payload, _ := protojson.MarshalOptions{EmitUnpopulated: true, UseProtoNames: true}.Marshal(resp)
	w.Write(payload)
}

func (h *Handler) handleGetRuleVersion(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ruleID := r.PathValue("rule_id")
	versionID := r.PathValue("version_id")
	if ruleID == "" || versionID == "" {
		writeJSONError(w, http.StatusBadRequest, "rule_id and version_id required")
		return
	}

	resp, err := h.analyticsClient.GetRuleVersion(r.Context(), &crudv1.GetRuleVersionRequest{
		RuleId:    ruleID,
		VersionId: versionID,
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	payload, _ := protojson.MarshalOptions{EmitUnpopulated: true, UseProtoNames: true}.Marshal(resp)
	w.Write(payload)
}

func (h *Handler) handleRuleReadiness(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ruleID := r.PathValue("rule_id")
	if ruleID == "" {
		writeJSONError(w, http.StatusBadRequest, "rule_id required")
		return
	}

	resp, err := h.analyticsClient.GetRuleReadiness(r.Context(), &crudv1.GetRuleReadinessRequest{RuleId: ruleID})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	payload, _ := protojson.MarshalOptions{EmitUnpopulated: true, UseProtoNames: true}.Marshal(resp)
	w.Write(payload)
}

func (h *Handler) handlePublishRule(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ruleID := r.PathValue("rule_id")
	if ruleID == "" {
		writeJSONError(w, http.StatusBadRequest, "rule_id required")
		return
	}

	type PublishRequest struct {
		VersionID string `json:"version_id"`
		Reason    string `json:"reason"`
		Actor     string `json:"actor"`
	}
	var req PublishRequest
	// Body optional
	if r.Body != nil && r.ContentLength > 0 {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}

	resp, err := h.analyticsClient.PublishRuleVersion(r.Context(), &crudv1.PublishRuleVersionRequest{
		RuleId:    ruleID,
		VersionId: req.VersionID,
		Reason:    req.Reason,
		Actor:     req.Actor,
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	payload, _ := protojson.MarshalOptions{EmitUnpopulated: true, UseProtoNames: true}.Marshal(resp)
	w.Write(payload)
}

func (h *Handler) handleRuleDiff(w http.ResponseWriter, r *http.Request) {
	// GET /rules/{rule_id}/diff?version_a=...&version_b=...
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ruleID := r.PathValue("rule_id")
	versionA := r.URL.Query().Get("version_a")
	versionB := r.URL.Query().Get("version_b") // "live" or empty

	if ruleID == "" || versionA == "" {
		writeJSONError(w, http.StatusBadRequest, "rule_id and version_a required")
		return
	}
	if versionB == "" {
		versionB = "active"
	}

	resp, err := h.analyticsClient.DiffRuleVersions(r.Context(), &crudv1.DiffRuleVersionsRequest{
		RuleId:   ruleID,
		VersionA: versionA,
		VersionB: versionB,
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	payload, _ := protojson.MarshalOptions{EmitUnpopulated: true, UseProtoNames: true}.Marshal(resp)
	w.Write(payload)
}
