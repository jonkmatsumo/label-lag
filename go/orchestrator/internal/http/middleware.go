package httpserver

import (
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/tenant"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/time/rate"
)

var (
	globalRateLimitedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "orchestrator_global_rate_limited_total",
		Help: "Total requests rejected by the global rate limiter.",
	}, []string{"route", "method", "status"})
	rateLimitedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "orchestrator_rate_limited_total",
		Help: "Total requests rejected by the rate limiter.",
	}, []string{"tenant_present"})
	rateLimitTenants = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "orchestrator_rate_limit_tenants_total",
		Help: "Total number of active tenant limiters.",
	})

	globalLimiter = newGlobalLimiterFromEnv()
)

type tenantLimiter struct {
	limiter  *rate.Limiter
	lastSeen atomic.Int64 // Unix nanoseconds
}

const (
	defaultTenantRateLimitRPS   = 10.0
	defaultTenantRateLimitBurst = 20

	// Conservative process-wide defaults. Override via env when higher throughput is safe.
	defaultGlobalRateLimitRPS   = 200.0
	defaultGlobalRateLimitBurst = 400

	globalRateLimitRPSEnv   = "INFERENCE_GATEWAY_GLOBAL_RATE_LIMIT_RPS"
	globalRateLimitBurstEnv = "INFERENCE_GATEWAY_GLOBAL_RATE_LIMIT_BURST"
)

// updateLastSeen ensures the timestamp only moves forward, protecting
// against clock skew or out-of-order updates in high concurrency.
func (tl *tenantLimiter) updateLastSeen(now int64) {
	for {
		old := tl.lastSeen.Load()
		if now <= old {
			return
		}
		if tl.lastSeen.CompareAndSwap(old, now) {
			return
		}
	}
}

// limiterMap uses sync.Map to avoid holding a global mutex on the hot
// path. Keys are tenant ID strings, values are *tenantLimiter.
var limiterMap sync.Map

func init() {
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		for range ticker.C {
			cleanupStaleLimiters()
		}
	}()
}

// cleanupStaleLimiters iterates without holding a global lock; each
// delete is lock-free on the sync.Map.
func cleanupStaleLimiters() {
	now := time.Now().UnixNano()
	var count float64
	limiterMap.Range(func(key, value any) bool {
		tl := value.(*tenantLimiter)
		if now-tl.lastSeen.Load() > int64(1*time.Hour) {
			limiterMap.Delete(key)
		} else {
			count++
		}
		return true
	})
	rateLimitTenants.Set(count)
}

func newGlobalLimiterFromEnv() *rate.Limiter {
	rps := parsePositiveFloatEnv(globalRateLimitRPSEnv, defaultGlobalRateLimitRPS)
	burst := parsePositiveIntEnv(globalRateLimitBurstEnv, defaultGlobalRateLimitBurst)
	return rate.NewLimiter(rate.Limit(rps), burst)
}

func resetGlobalLimiterFromEnv() {
	globalLimiter = newGlobalLimiterFromEnv()
}

func parsePositiveFloatEnv(key string, fallback float64) float64 {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || parsed <= 0 {
		slog.Warn("invalid rate limiter float env var", "key", key, "value", value, "fallback", fallback)
		return fallback
	}
	return parsed
}

func parsePositiveIntEnv(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		slog.Warn("invalid rate limiter int env var", "key", key, "value", value, "fallback", fallback)
		return fallback
	}
	return parsed
}

func rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !globalLimiter.Allow() {
			statusCode := http.StatusTooManyRequests
			globalRateLimitedTotal.WithLabelValues(normalizeRoute(r.URL.Path), r.Method, strconv.Itoa(statusCode)).Inc()
			writeJSONError(w, r, statusCode, "global rate limit exceeded")
			return
		}

		tenantID := tenant.FromContext(r.Context())
		tenantPresent := "true"
		if tenantID == "" {
			tenantID = "default"
			tenantPresent = "false"
		}

		val, loaded := limiterMap.LoadOrStore(tenantID, &tenantLimiter{
			limiter: rate.NewLimiter(rate.Limit(defaultTenantRateLimitRPS), defaultTenantRateLimitBurst),
		})
		if !loaded {
			rateLimitTenants.Inc()
		}
		tl := val.(*tenantLimiter)
		tl.updateLastSeen(time.Now().UnixNano())

		if !tl.limiter.Allow() {
			rateLimitedTotal.WithLabelValues(tenantPresent).Inc()
			writeJSONError(w, r, http.StatusTooManyRequests, "rate limit exceeded")
			return
		}

		next.ServeHTTP(w, r)
	})
}

func tenancyMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenantID := r.Header.Get("X-Tenant-Id")
		if requiresTenantHeader(r.URL.Path) && tenantID == "" {
			writeJSONError(w, r, http.StatusBadRequest, "missing X-Tenant-Id")
			return
		}
		// No default fallback. requiresTenantHeader enforces requirement for analytics.

		ctx := tenant.WithTenantID(r.Context(), tenantID)
		span := trace.SpanFromContext(ctx)
		span.SetAttributes(attribute.String("tenant_id", tenantID))

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func requestIDMiddleware(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		requestID := r.Header.Get("X-Request-Id")
		if requestID == "" {
			requestID = requestid.Generate()
		}

		w.Header().Set("X-Request-Id", requestID)

		ctx := requestid.WithRequestID(r.Context(), requestID)
		tracer := otel.Tracer("inference-gateway")
		ctx, span := tracer.Start(ctx, "HTTP "+r.Method+" "+r.URL.Path, trace.WithAttributes(
			attribute.String("http.method", r.Method),
			attribute.String("http.target", r.URL.Path),
			attribute.String("request_id", requestID),
		))
		defer span.End()

		rec := &statusResponseWriter{ResponseWriter: w}
		next.ServeHTTP(rec, r.WithContext(ctx))

		status := rec.status
		if status == 0 {
			status = http.StatusOK
		}

		span.SetAttributes(
			attribute.Int("http.status_code", status),
			attribute.Int("http.response_content_length", rec.bytes),
		)
		if status >= http.StatusBadRequest {
			span.SetStatus(codes.Error, http.StatusText(status))
		}

		logger.Info("request completed",
			"method", r.Method,
			"path", r.URL.Path,
			"request_id", requestID,
			"status", status,
			"duration", time.Since(start),
			"bytes", rec.bytes,
		)
	})
}

type statusResponseWriter struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (w *statusResponseWriter) WriteHeader(statusCode int) {
	w.status = statusCode
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *statusResponseWriter) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	n, err := w.ResponseWriter.Write(data)
	w.bytes += n
	return n, err
}
