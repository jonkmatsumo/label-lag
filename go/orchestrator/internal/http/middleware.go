package httpserver

import (
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/tenant"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/time/rate"
)

type tenantLimiter struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

var (
	limiters = make(map[string]*tenantLimiter)
	limitMu  sync.Mutex
)

func init() {
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		for range ticker.C {
			limitMu.Lock()
			for tid, l := range limiters {
				if time.Since(l.lastSeen) > 1*time.Hour {
					delete(limiters, tid)
				}
			}
			limitMu.Unlock()
		}
	}()
}

func rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenantID := tenant.FromContext(r.Context())
		if tenantID == "" {
			tenantID = "default"
		}

		limitMu.Lock()
		l, ok := limiters[tenantID]
		if !ok {
			// Default: 10 req/s, burst 20
			l = &tenantLimiter{
				limiter:  rate.NewLimiter(rate.Limit(10), 20),
				lastSeen: time.Now(),
			}
			limiters[tenantID] = l
		}
		l.lastSeen = time.Now()
		limitMu.Unlock()

		if !l.limiter.Allow() {
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
