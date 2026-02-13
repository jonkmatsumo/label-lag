package httpserver

import (
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	// Metrics label policy:
	// - Never use unbounded labels (tenant IDs, user IDs, request IDs, model names).
	// - For tenancy visibility use a bounded dimension: tenant=present|missing.
	httpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "orchestrator_http_requests_total",
		Help: "Total number of HTTP requests.",
	}, []string{"method", "route", "status", "tenant"})

	httpRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "orchestrator_http_request_duration_seconds",
		Help:    "Duration of HTTP requests in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"method", "route", "tenant"})
)

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func metricsMiddleware(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(rw, r)

		duration := time.Since(start).Seconds()
		route := normalizeRoute(r.URL.Path)
		tenant := tenantPresenceLabel(r.Header.Get("X-Tenant-Id"))

		httpRequestsTotal.WithLabelValues(r.Method, route, strconv.Itoa(rw.statusCode), tenant).Inc()
		httpRequestDuration.WithLabelValues(r.Method, route, tenant).Observe(duration)
	})
}

func tenantPresenceLabel(headerValue string) string {
	if strings.TrimSpace(headerValue) == "" {
		return "missing"
	}
	return "present"
}

// normalizeRoute maps raw paths to templates to avoid cardinality explosion.
func normalizeRoute(path string) string {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) == 0 {
		return "/"
	}

	// Known routes with dynamic parameters
	if parts[0] == "decisions" && len(parts) > 1 {
		if len(parts) == 3 && parts[2] == "trace" {
			return "/decisions/{request_id}/trace"
		}
		return "/decisions/{request_id}"
	}
	if parts[0] == "analytics" && len(parts) > 1 {
		if parts[1] == "rules" && len(parts) > 2 {
			if len(parts) == 4 && parts[3] == "impact" {
				return "/analytics/rules/{rule_id}/impact"
			}
			return "/analytics/rules/{rule_id}"
		}
	}
	if parts[0] == "jobs" && len(parts) > 1 {
		if len(parts) == 3 && parts[2] == "events" {
			return "/jobs/{id}/events"
		}
		return "/jobs/{id}"
	}
	if parts[0] == "dataset" && parts[1] == "profiles" && len(parts) > 2 {
		return "/dataset/profiles/{id}"
	}
	if parts[0] == "training-runs" && len(parts) > 1 {
		return "/training-runs/{id}"
	}
	if parts[0] == "rules" && len(parts) > 1 {
		// rules/{rule_id}/...
		if len(parts) >= 3 {
			switch parts[2] {
			case "history":
				return "/rules/{rule_id}/history"
			case "versions":
				return "/rules/{rule_id}/versions/{version_id}"
			case "publish":
				return "/rules/{rule_id}/publish"
			case "readiness":
				return "/rules/{rule_id}/readiness"
			case "diff":
				return "/rules/{rule_id}/diff"
			}
		}
		return "/rules/{rule_id}"
	}

	// Default: return fixed segments, or the original path if simple
	return "/" + strings.Join(parts, "/")
}

func (h *Handler) handleMetrics(w http.ResponseWriter, r *http.Request) {
	promhttp.Handler().ServeHTTP(w, r)
}
