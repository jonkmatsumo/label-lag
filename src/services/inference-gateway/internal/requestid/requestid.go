package requestid

import (
	"context"
	"crypto/rand"
	"encoding/hex"
)

type key struct{}

func WithRequestID(ctx context.Context, requestID string) context.Context {
	return context.WithValue(ctx, key{}, requestID)
}

func FromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if v, ok := ctx.Value(key{}).(string); ok {
		return v
	}
	return ""
}

func Generate() string {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "req_unknown"
	}
	return "req_" + hex.EncodeToString(buf)
}
