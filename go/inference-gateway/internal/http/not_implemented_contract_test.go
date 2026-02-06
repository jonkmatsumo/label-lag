package httpserver

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
)

func TestNotImplementedContractIsFlat(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, rules.NewEmptyProvider(), 1024)

	req := httptest.NewRequest(http.MethodPost, "/analytics/attribution", nil)
	req = req.WithContext(requestid.WithRequestID(req.Context(), "req-501"))
	rec := httptest.NewRecorder()

	handler.handleNotImplemented(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("expected status %d, got %d", http.StatusNotImplemented, rec.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if payload["error"] != "not_implemented" {
		t.Fatalf("expected error not_implemented, got %v", payload["error"])
	}
	if _, ok := payload["error"].(map[string]any); ok {
		t.Fatalf("expected flat error string, got nested object")
	}
	if payload["path"] != "/analytics/attribution" {
		t.Fatalf("expected path to be set, got %v", payload["path"])
	}
	if payload["method"] != http.MethodPost {
		t.Fatalf("expected method to be set, got %v", payload["method"])
	}
	if payload["request_id"] != "req-501" {
		t.Fatalf("expected request_id req-501, got %v", payload["request_id"])
	}
}
