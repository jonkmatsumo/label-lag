package obs

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

type contextKey string

const (
	requestIDKey contextKey = "x-request-id"
)

func RequestIDInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if ok {
		if ids := md.Get("x-request-id"); len(ids) > 0 {
			ctx = context.WithValue(ctx, requestIDKey, ids[0])
		}
	}
	return handler(ctx, req)
}
