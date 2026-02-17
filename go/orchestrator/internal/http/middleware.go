package httpserver

import (
	"log/slog"
	"net/http"
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
	rateLimitedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "orchestrator_rate_limited_total",
		Help: "Total requests rejected by the rate limiter.",
	}, []string{"tenant_present"})
)

type tenantLimiter struct {
	limiter  *rate.Limiter
	lastSeen atomic.Int64 // Unix nanoseconds
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
	limiterMap.Range(func(key, value any) bool {
		tl := value.(*tenantLimiter)
		if now-tl.lastSeen.Load() > int64(1*time.Hour) {
			limiterMap.Delete(key)
		}
		return true
	})
}

func rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenantID := tenant.FromContext(r.Context())
		tenantPresent := "true"
		if tenantID == "" {
			tenantID = "default"
			tenantPresent = "false"
		}

		val, _ := limiterMap.LoadOrStore(tenantID, &tenantLimiter{
			limiter: rate.NewLimiter(rate.Limit(10), 20),
		})
		tl := val.(*tenantLimiter)
		tl.lastSeen.Store(time.Now().UnixNano())

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
		if tenantID == "" {
			tenantID = "default"
		}

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
