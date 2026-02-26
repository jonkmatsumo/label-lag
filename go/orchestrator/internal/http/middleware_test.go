package httpserver

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"golang.org/x/time/rate"

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
	globalLimiter = rate.NewLimiter(rate.Limit(defaultGlobalRateLimitRPS), defaultGlobalRateLimitBurst)
	globalRateLimitedTotal.Reset()
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

func TestGlobalRateLimitMiddleware_AppliesAcrossTenants(t *testing.T) {
	resetLimiterMap()
	globalLimiter = rate.NewLimiter(rate.Limit(1), 1)

	handler := rateLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	reqA := httptest.NewRequest(http.MethodGet, "/analytics/overview", nil)
	reqA = reqA.WithContext(tenant.WithTenantID(reqA.Context(), "tenant-a"))
	recA := httptest.NewRecorder()
	handler.ServeHTTP(recA, reqA)
	if recA.Code != http.StatusOK {
		t.Fatalf("expected first request to pass, got %d", recA.Code)
	}

	reqB := httptest.NewRequest(http.MethodGet, "/analytics/overview", nil)
	reqB = reqB.WithContext(tenant.WithTenantID(reqB.Context(), "tenant-b"))
	recB := httptest.NewRecorder()
	handler.ServeHTTP(recB, reqB)
	if recB.Code != http.StatusTooManyRequests {
		t.Fatalf("expected global 429 for rotating tenant, got %d", recB.Code)
	}

	if got := testutil.ToFloat64(globalRateLimitedTotal.WithLabelValues("/analytics/overview", "GET", "429")); got != 1 {
		t.Fatalf("expected global limiter metric to increment by 1, got %v", got)
	}
}

func TestResetGlobalLimiterFromEnv_AppliesConfiguredValues(t *testing.T) {
	t.Setenv(globalRateLimitRPSEnv, "123.5")
	t.Setenv(globalRateLimitBurstEnv, "77")

	resetGlobalLimiterFromEnv()

	if got := float64(globalLimiter.Limit()); got != 123.5 {
		t.Fatalf("expected global limiter rps=123.5, got %v", got)
	}
	if got := globalLimiter.Burst(); got != 77 {
		t.Fatalf("expected global limiter burst=77, got %d", got)
	}
}

func TestGlobalRateLimitMetricLabelsRemainBounded(t *testing.T) {
	resetLimiterMap()
	globalLimiter = rate.NewLimiter(rate.Limit(1), 1)

	handler := rateLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	// First request consumes the single token.
	req1 := httptest.NewRequest(http.MethodGet, "/analytics/rules/rule-123/impact", nil)
	req1 = req1.WithContext(tenant.WithTenantID(req1.Context(), "tenant-a"))
	handler.ServeHTTP(httptest.NewRecorder(), req1)

	// Second request is globally limited.
	req2 := httptest.NewRequest(http.MethodGet, "/analytics/rules/rule-456/impact", nil)
	req2 = req2.WithContext(tenant.WithTenantID(req2.Context(), "tenant-b"))
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusTooManyRequests {
		t.Fatalf("expected status 429, got %d", rec2.Code)
	}

	counter, err := globalRateLimitedTotal.GetMetricWithLabelValues("/analytics/rules/{rule_id}/impact", "GET", "429")
	if err != nil {
		t.Fatalf("failed to fetch labeled metric: %v", err)
	}
	metric := &dto.Metric{}
	if err := counter.Write(metric); err != nil {
		t.Fatalf("failed to read metric labels: %v", err)
	}

	if len(metric.GetLabel()) != 3 {
		t.Fatalf("expected exactly 3 labels, got %d", len(metric.GetLabel()))
	}
	labelNames := map[string]bool{}
	for _, l := range metric.GetLabel() {
		labelNames[l.GetName()] = true
	}
	if !labelNames["route"] || !labelNames["method"] || !labelNames["status"] {
		t.Fatalf("expected labels route/method/status, got %#v", labelNames)
	}
}

func TestRateLimitMiddleware_PerTenantStillIsolatedWhenGlobalHasHeadroom(t *testing.T) {
	resetLimiterMap()
	globalLimiter = rate.NewLimiter(rate.Limit(1000), 1000)

	handler := rateLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	tenantA := "tenant-a"
	tenantB := "tenant-b"

	for i := 0; i < 20; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req = req.WithContext(tenant.WithTenantID(req.Context(), tenantA))
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
	}

	reqA := httptest.NewRequest(http.MethodGet, "/", nil)
	reqA = reqA.WithContext(tenant.WithTenantID(reqA.Context(), tenantA))
	recA := httptest.NewRecorder()
	handler.ServeHTTP(recA, reqA)
	if recA.Code != http.StatusTooManyRequests {
		t.Fatalf("expected tenant A to be throttled, got %d", recA.Code)
	}

	reqB := httptest.NewRequest(http.MethodGet, "/", nil)
	reqB = reqB.WithContext(tenant.WithTenantID(reqB.Context(), tenantB))
	recB := httptest.NewRecorder()
	handler.ServeHTTP(recB, reqB)
	if recB.Code != http.StatusOK {
		t.Fatalf("expected tenant B to remain unaffected, got %d", recB.Code)
	}
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

func TestTenantLimiter_UpdateLastSeen(t *testing.T) {
	tl := &tenantLimiter{}
	tl.lastSeen.Store(100)

	// Update with newer time
	tl.updateLastSeen(200)
	if tl.lastSeen.Load() != 200 {
		t.Errorf("expected 200, got %d", tl.lastSeen.Load())
	}

	// Update with older time (should be ignored)
	tl.updateLastSeen(150)
	if tl.lastSeen.Load() != 200 {
		t.Errorf("expected 200, got %d", tl.lastSeen.Load())
	}

	// Update with same time
	tl.updateLastSeen(200)
	if tl.lastSeen.Load() != 200 {
		t.Errorf("expected 200, got %d", tl.lastSeen.Load())
	}
}

func TestRateLimitTenantsGauge(t *testing.T) {
	resetLimiterMap()
	rateLimitTenants.Set(0)

	handler := rateLimitMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	// 1. Initial request for tenant-1
	req1 := httptest.NewRequest(http.MethodGet, "/", nil)
	req1 = req1.WithContext(tenant.WithTenantID(req1.Context(), "tenant-1"))
	handler.ServeHTTP(httptest.NewRecorder(), req1)

	if val := testutil.ToFloat64(rateLimitTenants); val != 1 {
		t.Errorf("expected gauge 1 after first tenant, got %v", val)
	}

	// 2. Second request for SAME tenant-1
	handler.ServeHTTP(httptest.NewRecorder(), req1)
	if val := testutil.ToFloat64(rateLimitTenants); val != 1 {
		t.Errorf("expected gauge still 1 after same tenant, got %v", val)
	}

	// 3. Request for tenant-2
	req2 := httptest.NewRequest(http.MethodGet, "/", nil)
	req2 = req2.WithContext(tenant.WithTenantID(req2.Context(), "tenant-2"))
	handler.ServeHTTP(httptest.NewRecorder(), req2)

	if val := testutil.ToFloat64(rateLimitTenants); val != 2 {
		t.Errorf("expected gauge 2 after second tenant, got %v", val)
	}

	// 4. Cleanup should update gauge
	// Manually make tenant-1 stale
	val, _ := limiterMap.Load("tenant-1")
	tl1 := val.(*tenantLimiter)
	tl1.lastSeen.Store(time.Now().Add(-2 * time.Hour).UnixNano())

	cleanupStaleLimiters()

	if val := testutil.ToFloat64(rateLimitTenants); val != 1 {
		t.Errorf("expected gauge 1 after cleanup of 1 tenant, got %v", val)
	}
}
