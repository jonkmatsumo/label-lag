package obs

import (
	"context"
	"log/slog"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func LoggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()

	// Create context with logger loaded with method info
	logger := slog.With("method", info.FullMethod)

	resp, err := handler(ctx, req)

	duration := time.Since(start)

	if err != nil {
		st, _ := status.FromError(err)
		logger.Error("request failed",
			"duration", duration,
			"code", st.Code().String(),
			"error", err,
		)
	} else {
		logger.Info("request completed",
			"duration", duration,
			"code", codes.OK.String(),
		)
	}

	return resp, err
}
