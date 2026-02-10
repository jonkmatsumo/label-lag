package httpserver

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
)

func TestDecisionAPIGating(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{}

	tests := []struct {
		name           string
		enableDecisions bool
		path           string
		expectedStatus int
	}{
		{
			name:           "list decisions disabled",
			enableDecisions: false,
			path:           "/decisions",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "get decision disabled",
			enableDecisions: false,
			path:           "/decisions/req-1",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "rule impact disabled",
			enableDecisions: false,
			path:           "/analytics/rules/rule-1/impact",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "list decisions enabled",
			enableDecisions: true,
			path:           "/decisions",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "get decision enabled",
			enableDecisions: true,
			path:           "/decisions/req-1",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "rule impact enabled",
			enableDecisions: true,
			path:           "/analytics/rules/rule-1/impact",
			expectedStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := NewHandler(logger, nil, stub, nil, nil, rules.NewEmptyProvider(), 1024, "", "", tt.enableDecisions, false, false, false)
			mux := http.NewServeMux()
			handler.Register(mux)

			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, req)

			if rec.Code != tt.expectedStatus {
				t.Errorf("expected status %d, got %d", tt.expectedStatus, rec.Code)
			}
		})
	}
}
