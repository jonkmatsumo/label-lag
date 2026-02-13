package obs

import (
	"context"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestMetricsInterceptorRecordsMethodAndCode(t *testing.T) {
	grpcRequestsTotal.Reset()
	grpcRequestDuration.Reset()

	info := &grpc.UnaryServerInfo{
		FullMethod: "/crud.v1.AnalyticsService/GetJobEvents",
	}

	_, err := MetricsInterceptor(context.Background(), nil, info, func(context.Context, interface{}) (interface{}, error) {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument error, got %v", status.Code(err))
	}

	metric := &dto.Metric{}
	if err := grpcRequestsTotal.WithLabelValues(
		"crud.v1.AnalyticsService",
		"GetJobEvents",
		"InvalidArgument",
	).Write(metric); err != nil {
		t.Fatalf("failed to read metric value: %v", err)
	}
	count := metric.GetCounter().GetValue()
	if count != 1 {
		t.Fatalf("expected request count 1, got %v", count)
	}
}

func TestSplitFullMethodHandlesUnexpectedInput(t *testing.T) {
	service, method := splitFullMethod("invalid")
	if service != "unknown" || method != "unknown" {
		t.Fatalf("expected unknown/unknown, got %q/%q", service, method)
	}
}
