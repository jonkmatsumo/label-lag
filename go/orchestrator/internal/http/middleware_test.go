package httpserver

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/tenant"
)

func TestStatusResponseWriterTracksStatusAndBytes(t *testing.T) {
	rec := httptest.NewRecorder()
	writer := &statusResponseWriter{ResponseWriter: rec}

	writer.WriteHeader(http.StatusTeapot)
	_, _ = writer.Write([]byte("hello"))

	if writer.status != http.StatusTeapot {
		t.Fatalf("expected status %d, got %d", http.StatusTeapot, writer.status)
	}
	if writer.bytes != 5 {
		t.Fatalf("expected 5 bytes, got %d", writer.bytes)
	}

	rec2 := httptest.NewRecorder()
	writer2 := &statusResponseWriter{ResponseWriter: rec2}
	_, _ = writer2.Write([]byte("ok"))

	if writer2.status != http.StatusOK {
		t.Fatalf("expected default status %d, got %d", http.StatusOK, writer2.status)
	}
	if writer2.bytes != 2 {
		t.Fatalf("expected 2 bytes, got %d", writer2.bytes)
	}
}

func TestRateLimitMiddleware(t *testing.T) {
	// Reset limiters for test
	limitMu.Lock()
	limiters = make(map[string]*tenantLimiter)
	limitMu.Unlock()

	handler := rateLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	t.Run("Throttles after burst", func(t *testing.T) {
		tenantID := "test-tenant-1"
		// Default burst is 20. Let's hit it 20 times.
		for i := 0; i < 20; i++ {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			req = req.WithContext(tenant.WithTenantID(req.Context(), tenantID))
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("expected status 200 at request %d, got %d", i, rec.Code)
			}
		}

		// 21st request should be throttled
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req = req.WithContext(tenant.WithTenantID(req.Context(), tenantID))
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusTooManyRequests {
			t.Fatalf("expected status 429, got %d", rec.Code)
		}
	})

	t.Run("Isolates tenants", func(t *testing.T) {
		tenantA := "tenant-a"
		tenantB := "tenant-b"

		// Exhaust burst for tenant A
		for i := 0; i < 20; i++ {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			req = req.WithContext(tenant.WithTenantID(req.Context(), tenantA))
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
		}

		// Tenant A should be throttled
		reqA := httptest.NewRequest(http.MethodGet, "/", nil)
		reqA = reqA.WithContext(tenant.WithTenantID(reqA.Context(), tenantA))
		recA := httptest.NewRecorder()
		handler.ServeHTTP(recA, reqA)
		if recA.Code != http.StatusTooManyRequests {
			t.Fatalf("expected tenant A to be throttled, got %d", recA.Code)
		}

		// Tenant B should NOT be throttled
		reqB := httptest.NewRequest(http.MethodGet, "/", nil)
		reqB = reqB.WithContext(tenant.WithTenantID(reqB.Context(), tenantB))
		recB := httptest.NewRecorder()
		handler.ServeHTTP(recB, reqB)
		if recB.Code != http.StatusOK {
			t.Fatalf("expected tenant B to be OK, got %d", recB.Code)
		}
	})
}
