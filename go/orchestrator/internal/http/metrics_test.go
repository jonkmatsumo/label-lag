package httpserver

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestMetricsEndpoint(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))

	mux := http.NewServeMux()
	mux.HandleFunc("/test", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	// Wrap with middleware
	h := metricsMiddleware(logger, mux)

	// Register metrics handler on the same mux for the test
	handler := &Handler{}
	mux.HandleFunc("/metrics", handler.handleMetrics)

	// 1. Hit a route to initialize metrics
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	// 2. Hit /metrics
	req = httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "orchestrator_http_requests_total") {
		t.Errorf("expected metrics to contain orchestrator_http_requests_total. Body: %s", body)
	}
}

func TestMetricsInstrumentation(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))

	// Reset counters for clean test
	httpRequestsTotal.Reset()
	httpRequestDuration.Reset()

	mux := http.NewServeMux()
	mux.HandleFunc("/test", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		w.Write([]byte("ok"))
	})

	handler := metricsMiddleware(logger, mux)

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d", http.StatusAccepted, rec.Code)
	}

	// Verify metrics
	count := testutil.ToFloat64(httpRequestsTotal.WithLabelValues("GET", "/test", "202", "missing"))
	if count != 1 {
		t.Errorf("expected count 1, got %v", count)
	}
}

func TestNormalizeRoute(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		{"/ready", "/ready"},
		{"/decisions/req-123", "/decisions/{request_id}"},
		{"/decisions/req-123/trace", "/decisions/{request_id}/trace"},
		{"/jobs/job-456", "/jobs/{id}"},
		{"/jobs/job-456/events", "/jobs/{id}/events"},
		{"/dataset/profiles/prof-789", "/dataset/profiles/{id}"},
		{"/analytics/rules/rule-01/impact", "/analytics/rules/{rule_id}/impact"},
		{"/analytics/overview", "/analytics/overview"},
		{"/unknown/path/segment", "/unknown/path/segment"},
	}

	for _, tc := range tests {
		t.Run(tc.path, func(t *testing.T) {
			got := normalizeRoute(tc.path)
			if got != tc.expected {
				t.Errorf("normalizeRoute(%q) = %q; want %q", tc.path, got, tc.expected)
			}
		})
	}
}

func TestTenantPresenceLabel(t *testing.T) {
	if got := tenantPresenceLabel(""); got != "missing" {
		t.Fatalf("expected missing, got %q", got)
	}
	if got := tenantPresenceLabel("tenant-1"); got != "present" {
		t.Fatalf("expected present, got %q", got)
	}
}
