package httpserver

import (
	"errors"
	"net/http"
	"strings"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/tenant"
)

// tenantIDFromRequest returns the tenant ID propagated by middleware and
// guarantees a stable default for handlers/tests that bypass middleware.
func tenantIDFromRequest(r *http.Request) string {
	tenantID := tenant.FromContext(r.Context())
	if tenantID == "" {
		return "default"
	}
	return tenantID
}

// mustTenantID ensures a tenant ID is present in the request context.
// It returns an error if the tenant ID is missing, enforcing the contract
// that analytics endpoints must be tenant-scoped.
func mustTenantID(r *http.Request) (string, error) {
	tenantID := tenant.FromContext(r.Context())
	if tenantID == "" {
		return "", errors.New("missing X-Tenant-Id")
	}
	return tenantID, nil
}

// Tenant contract: analytics/readiness surfaces are tenant-scoped and require
// X-Tenant-Id. Non-analytics endpoints may continue to use the middleware
// fallback tenant for backward compatibility.
//
// requiresTenantHeader reports whether a route is tenant-scoped for analytics
// and must include X-Tenant-Id.
func requiresTenantHeader(path string) bool {
	if path == "/metrics/shadow/comparison" || strings.HasPrefix(path, "/metrics/shadow/comparison/") {
		return true
	}

	if strings.HasPrefix(path, "/rules/sandbox/") {
		return false
	}

	for _, prefix := range []string{
		"/analytics",
		"/decisions",
		"/kpis",
		"/volume",
		"/jobs",
		"/dataset",
		"/training-runs",
		"/models/versions",
		"/metrics/series",
		"/monitoring/drift",
		"/backtest",
		"/rules",
		"/data/clear",
		"/data/generate",
		"/train",
		"/models/deploy",
	} {
		if path == prefix || strings.HasPrefix(path, prefix+"/") {
			return true
		}
	}

	return false
}
