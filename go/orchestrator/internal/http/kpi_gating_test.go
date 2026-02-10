package httpserver

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
)

func TestKpiAPIGating(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{}

	tests := []struct {
		name           string
		enableKpis     bool
		path           string
		expectedStatus int
	}{
		{
			name:           "kpis disabled",
			enableKpis:     false,
			path:           "/kpis",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "volume disabled",
			enableKpis:     false,
			path:           "/volume",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "confusion matrix disabled",
			enableKpis:     false,
			path:           "/analytics/confusion-matrix",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "kpis enabled",
			enableKpis:     true,
			path:           "/kpis",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "volume enabled",
			enableKpis:     true,
			path:           "/volume",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "confusion matrix enabled",
			enableKpis:     true,
			path:           "/analytics/confusion-matrix",
			expectedStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := NewHandler(logger, nil, stub, nil, nil, rules.NewEmptyProvider(), 1024, "", "", false, tt.enableKpis, false)
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
