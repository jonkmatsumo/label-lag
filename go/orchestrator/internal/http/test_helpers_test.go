package httpserver

import (
	"net/http"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/tenant"
)

const testTenantID = "tenant-1"

func withTenantRequest(req *http.Request) *http.Request {
	req.Header.Set("X-Tenant-Id", testTenantID)
	return req.WithContext(tenant.WithTenantID(req.Context(), testTenantID))
}
