package httpserver

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTenantIDFromRequestDefaultsWhenMissing(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	if got := tenantIDFromRequest(req); got != "default" {
		t.Fatalf("expected default tenant id, got %q", got)
	}
}

func TestTenancyMiddlewarePropagatesHeader(t *testing.T) {
	var gotTenantID string
	next := http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		gotTenantID = tenantIDFromRequest(r)
	})

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.Header.Set("X-Tenant-Id", "tenant-acme")
	rec := httptest.NewRecorder()

	tenancyMiddleware(next).ServeHTTP(rec, req)

	if gotTenantID != "tenant-acme" {
		t.Fatalf("expected tenant-acme, got %q", gotTenantID)
	}
}

func TestTenancyMiddlewareUsesDefaultWhenHeaderMissing(t *testing.T) {
	var gotTenantID string
	next := http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		gotTenantID = tenantIDFromRequest(r)
	})

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	tenancyMiddleware(next).ServeHTTP(rec, req)

	if gotTenantID != "default" {
		t.Fatalf("expected default tenant id, got %q", gotTenantID)
	}
}

func TestTenancyMiddlewareRejectsMissingTenantHeaderForAnalyticsRoutes(t *testing.T) {
	next := http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Fatalf("next handler should not be called when tenant header is missing")
	})

	req := httptest.NewRequest(http.MethodGet, "/analytics/overview", nil)
	rec := httptest.NewRecorder()

	tenancyMiddleware(next).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
	body, err := io.ReadAll(rec.Body)
	if err != nil {
		t.Fatalf("failed to read response body: %v", err)
	}
	if string(body) != `{"detail":"missing X-Tenant-Id"}` {
		t.Fatalf("unexpected response body: %s", string(body))
	}
}

func TestRequiresTenantHeader(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{path: "/analytics/overview", want: true},
		{path: "/jobs", want: true},
		{path: "/rules/rule-1", want: true},
		{path: "/rules/sandbox/evaluate", want: false},
		{path: "/metrics", want: false},
		{path: "/health", want: false},
	}

	for _, tc := range cases {
		if got := requiresTenantHeader(tc.path); got != tc.want {
			t.Fatalf("requiresTenantHeader(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}
