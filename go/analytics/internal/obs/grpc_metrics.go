package obs

import (
	"context"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

var (
	grpcRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "analytics_grpc_requests_total",
		Help: "Total number of gRPC requests.",
	}, []string{"method", "code"})

	grpcRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "analytics_grpc_request_duration_seconds",
		Help:    "Duration of gRPC requests in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"method"})
)

// MetricsInterceptor records gRPC request counts and duration.
func MetricsInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()

	resp, err := handler(ctx, req)

	duration := time.Since(start).Seconds()
	code := status.Code(err).String()

	grpcRequestsTotal.WithLabelValues(info.FullMethod, code).Inc()
	grpcRequestDuration.WithLabelValues(info.FullMethod).Observe(duration)

	return resp, err
}
