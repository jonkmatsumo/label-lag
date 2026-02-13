package obs

import (
	"context"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

var (
	// Metrics label policy:
	// - Keep labels bounded and low-cardinality.
	// - Never use tenant_id (or other identifiers) as a metric label.
	grpcRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "analytics_grpc_requests_total",
		Help: "Total number of gRPC requests.",
	}, []string{"service", "method", "code"})

	grpcRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "analytics_grpc_request_duration_seconds",
		Help:    "Duration of gRPC requests in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"service", "method"})
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
	serviceName, methodName := splitFullMethod(info.FullMethod)

	grpcRequestsTotal.WithLabelValues(serviceName, methodName, code).Inc()
	grpcRequestDuration.WithLabelValues(serviceName, methodName).Observe(duration)

	return resp, err
}

func splitFullMethod(fullMethod string) (serviceName, methodName string) {
	parts := strings.Split(strings.TrimPrefix(fullMethod, "/"), "/")
	if len(parts) != 2 {
		return "unknown", "unknown"
	}
	return parts[0], parts[1]
}
