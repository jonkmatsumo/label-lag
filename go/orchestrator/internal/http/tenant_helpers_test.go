package httpserver

import (
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
