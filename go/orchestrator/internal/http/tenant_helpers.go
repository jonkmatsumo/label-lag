package httpserver

import (
	"net/http"

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
