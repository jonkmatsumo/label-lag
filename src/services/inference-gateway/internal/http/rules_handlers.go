package httpserver

import (
	"encoding/json"
	"net/http"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
)

type EvaluateRulesRequest struct {
	Features  map[string]any `json:"features"`
	BaseScore int            `json:"base_score"`
	RuleSet   *rules.RuleSet `json:"ruleset,omitempty"`
}

type EvaluateRulesResponse struct {
	FinalScore         int                 `json:"final_score"`
	MatchedRules       []string            `json:"matched_rules"`
	Explanations       []rules.Explanation `json:"explanations"`
	ShadowMatchedRules []string            `json:"shadow_matched_rules"`
	ShadowExplanations []rules.Explanation `json:"shadow_explanations"`
	Rejected           bool                `json:"rejected"`
	RuleSetVersion     string              `json:"ruleset_version"`
}

func (h *Handler) handleEvaluateRules(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var req EvaluateRulesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	ruleset := rules.RuleSet{}
	if req.RuleSet != nil {
		ruleset = *req.RuleSet
	} else {
		// Use default ruleset if not provided
		var err error
		ruleset, err = h.rulesProvider.GetRules(r.Context())
		if err != nil {
			h.logger.Warn("failed to load default ruleset", "error", err)
		}
	}

	result, err := rules.EvaluateRules(req.Features, req.BaseScore, ruleset)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "rule evaluation failed")
		return
	}

	resp := EvaluateRulesResponse{
		FinalScore:         result.FinalScore,
		MatchedRules:       result.MatchedRules,
		Explanations:       result.Explanations,
		ShadowMatchedRules: result.ShadowMatchedRules,
		ShadowExplanations: result.ShadowExplanations,
		Rejected:           result.Rejected,
		RuleSetVersion:     ruleset.Version,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}
