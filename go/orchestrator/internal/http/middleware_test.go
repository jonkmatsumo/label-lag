package httpserver

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

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

// resetLimiterMap clears the sync.Map for test isolation.
func resetLimiterMap() {
	limiterMap.Range(func(key, _ any) bool {
		limiterMap.Delete(key)
		return true
	})
}

func TestRateLimitMiddleware(t *testing.T) {
	resetLimiterMap()

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

func TestRateLimitMetricIncrements(t *testing.T) {
	resetLimiterMap()

	handler := rateLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	tenantID := "rate-limit-metric-test"

	// Exhaust burst (20 requests)
	for i := 0; i < 20; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req = req.WithContext(tenant.WithTenantID(req.Context(), tenantID))
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
	}

	// Record baseline
	before := testutil.ToFloat64(rateLimitedTotal.WithLabelValues("true"))

	// This request should be rate-limited
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req = req.WithContext(tenant.WithTenantID(req.Context(), tenantID))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429, got %d", rec.Code)
	}

	after := testutil.ToFloat64(rateLimitedTotal.WithLabelValues("true"))
	if after-before != 1 {
		t.Errorf("expected rate_limited_total to increment by 1, got delta %v", after-before)
	}
}

func TestRateLimitNoTenantLabel(t *testing.T) {
	resetLimiterMap()

	handler := rateLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	// Exhaust burst for "default" tenant (no tenant header)
	for i := 0; i < 20; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		// No tenant in context — will use "default"
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
	}

	before := testutil.ToFloat64(rateLimitedTotal.WithLabelValues("false"))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429, got %d", rec.Code)
	}

	after := testutil.ToFloat64(rateLimitedTotal.WithLabelValues("false"))
	if after-before != 1 {
		t.Errorf("expected rate_limited_total(false) to increment by 1, got delta %v", after-before)
	}
}

func TestCleanupStaleLimiters(t *testing.T) {
	resetLimiterMap()

	// Insert a "stale" limiter with lastSeen well in the past
	tl := &tenantLimiter{
		limiter: nil,
	}
	tl.lastSeen.Store(0) // epoch = very old
	limiterMap.Store("stale-tenant", tl)

	// Insert a "fresh" limiter
	fresh := &tenantLimiter{
		limiter: nil,
	}
	fresh.lastSeen.Store(int64(^uint64(0) >> 1)) // far future
	limiterMap.Store("fresh-tenant", fresh)

	cleanupStaleLimiters()

	if _, ok := limiterMap.Load("stale-tenant"); ok {
		t.Error("expected stale-tenant to be cleaned up")
	}
	if _, ok := limiterMap.Load("fresh-tenant"); !ok {
		t.Error("expected fresh-tenant to survive cleanup")
	}
}

func TestRateLimitConcurrentAccess(t *testing.T) {
	resetLimiterMap()

	handler := rateLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(tid string) {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			req = req.WithContext(tenant.WithTenantID(req.Context(), tid))
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
		}("concurrent-tenant")
	}
	wg.Wait()
	// If we reach here without a race/panic, the test passes
}
