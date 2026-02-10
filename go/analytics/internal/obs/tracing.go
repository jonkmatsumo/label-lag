package obs

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strconv"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

const (
	traceSamplerRatioEnv     = "OTEL_TRACES_SAMPLER_RATIO"
	defaultTraceSamplerRatio = 0.1
)

// InitTracer initializes an OTLP exporter, and configures the corresponding trace provider.
func InitTracer(ctx context.Context) (*sdktrace.TracerProvider, error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		// Use default temporary endpoint or return nil if we don't want to enforce tracing without config
		return nil, nil
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName("analytics-crud"),
			semconv.ServiceVersion("0.1.0"),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Set up trace exporter
	traceExporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithInsecure(),
		otlptracegrpc.WithEndpoint(endpoint),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace exporter: %w", err)
	}

	// Register the trace exporter with a TracerProvider, using a batch
	// span processor to aggregate spans before export.
	ratio := parseTraceSamplerRatio()
	bsp := sdktrace.NewBatchSpanProcessor(traceExporter)
	tracerProvider := sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(ratio))),
		sdktrace.WithResource(res),
		sdktrace.WithSpanProcessor(bsp),
	)

	// set global propagator to tracecontext (the default is no-op).
	otel.SetTextMapPropagator(propagation.TraceContext{})
	otel.SetTracerProvider(tracerProvider)

	return tracerProvider, nil
}

func parseTraceSamplerRatio() float64 {
	value := os.Getenv(traceSamplerRatioEnv)
	if value == "" {
		return defaultTraceSamplerRatio
	}

	ratio, err := strconv.ParseFloat(value, 64)
	if err != nil {
		slog.Warn("invalid trace sampler ratio, using default", "key", traceSamplerRatioEnv, "value", value, "default", defaultTraceSamplerRatio)
		return defaultTraceSamplerRatio
	}
	if ratio < 0 {
		slog.Warn("trace sampler ratio below 0.0, clamping", "key", traceSamplerRatioEnv, "value", value)
		return 0.0
	}
	if ratio > 1 {
		slog.Warn("trace sampler ratio above 1.0, clamping", "key", traceSamplerRatioEnv, "value", value)
		return 1.0
	}
	return ratio
}
