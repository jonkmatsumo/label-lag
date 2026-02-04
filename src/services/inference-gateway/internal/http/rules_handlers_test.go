package httpserver

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
)

func TestHandleEvaluateRules(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, rules.NewEmptyProvider(), 1024)

	payload := EvaluateRulesRequest{
		Features:  map[string]any{"velocity_24h": 10},
		BaseScore: 50,
		RuleSet: &rules.RuleSet{
			Version: "test_v1",
			Rules: []rules.Rule{
				{
					ID:     "rule1",
					Field:  "velocity_24h",
					Op:     ">",
					Value:  5,
					Action: "reject",
					Status: rules.RuleStatusActive,
				},
			},
		},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/evaluate/rules", bytes.NewReader(body))
	rec := httptest.NewRecorder()

	handler.handleEvaluateRules(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var resp EvaluateRulesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.FinalScore != 99 {
		t.Errorf("expected final score 99, got %d", resp.FinalScore)
	}
	if !resp.Rejected {
		t.Errorf("expected rejected true")
	}
	if len(resp.MatchedRules) != 1 || resp.MatchedRules[0] != "rule1" {
		t.Errorf("expected rule1 to match")
	}
}

func TestHandleEvaluateRulesDiff(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, rules.NewEmptyProvider(), 1024)

	payload := EvaluateRulesDiffRequest{
		Features:  map[string]any{"velocity_24h": 10},
		BaseScore: 50,
		RuleSetA: &rules.RuleSet{
			Version: "v1",
			Rules: []rules.Rule{
				{ID: "rule1", Field: "velocity_24h", Op: ">", Value: 5, Action: "override_score", Score: intPtr(60), Status: rules.RuleStatusActive},
			},
		},
		RuleSetB: &rules.RuleSet{
			Version: "v2",
			Rules: []rules.Rule{
				{ID: "rule1", Field: "velocity_24h", Op: ">", Value: 5, Action: "override_score", Score: intPtr(80), Status: rules.RuleStatusActive},
				{ID: "rule2", Field: "velocity_24h", Op: ">", Value: 15, Action: "reject", Status: rules.RuleStatusActive},
			},
		},
	}

	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/evaluate/rules/diff", bytes.NewReader(body))
	rec := httptest.NewRecorder()

	handler.handleEvaluateRulesDiff(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var resp EvaluateRulesDiffResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)

	if resp.Diff.ScoreDelta != 20 {
		t.Errorf("expected score delta 20, got %d", resp.Diff.ScoreDelta)
	}
	if len(resp.Diff.MatchedRulesAdded) != 0 {
		// rule2 doesn't match because velocity is 10
		t.Errorf("expected 0 added rules, got %v", resp.Diff.MatchedRulesAdded)
	}
}

func intPtr(v int) *int {
	return &v
}
