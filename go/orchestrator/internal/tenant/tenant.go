package tenant

import (
	"context"
)

type contextKey int

const (
	tenantIDKey contextKey = iota
)

// FromContext extracts the tenant ID from the context.
func FromContext(ctx context.Context) string {
	if tid, ok := ctx.Value(tenantIDKey).(string); ok {
		return tid
	}
	return ""
}

// WithTenantID returns a new context with the tenant ID.
func WithTenantID(ctx context.Context, tenantID string) context.Context {
	return context.WithValue(ctx, tenantIDKey, tenantID)
}
