package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	grpcclient "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/grpc"
	httpserver "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/http"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	ctx := context.Background()
	tp, err := initTracer(ctx)
	if err != nil {
		logger.Error("failed to initialize tracer", "error", err)
	} else if tp != nil {
		defer func() {
			if err := tp.Shutdown(ctx); err != nil {
				logger.Error("failed to shutdown tracer provider", "error", err)
			}
		}()
		logger.Info("opentelemetry tracer initialized")
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}

	inferenceClient, err := grpcclient.NewInferenceClient("", 0)
	if err != nil {
		logger.Error("failed to create inference client", "error", err)
		os.Exit(1)
	}
	defer func() {
		if err := inferenceClient.Close(); err != nil {
			logger.Warn("failed to close inference client", "error", err)
		}
	}()

	handler := httpserver.NewHandler(logger, inferenceClient, rules.NewEmptyProvider())
	srv := httpserver.NewServer("0.0.0.0:"+port, logger, handler)

	errCh := make(chan error, 1)
	go func() {
		logger.Info("server listening", "address", srv.Addr)
		errCh <- srv.ListenAndServe()
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	select {
	case <-stop:
		logger.Info("shutdown requested")
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			logger.Error("server error", "error", err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("shutdown error", "error", err)
	}
}

func initTracer(ctx context.Context) (*sdktrace.TracerProvider, error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		return nil, nil
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName("inference-gateway"),
			semconv.ServiceVersion("0.1.0"),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	traceExporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithInsecure(),
		otlptracegrpc.WithEndpoint(endpoint),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace exporter: %w", err)
	}

	bsp := sdktrace.NewBatchSpanProcessor(traceExporter)
	tracerProvider := sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
		sdktrace.WithResource(res),
		sdktrace.WithSpanProcessor(bsp),
	)

	otel.SetTextMapPropagator(propagation.TraceContext{})
	otel.SetTracerProvider(tracerProvider)

	return tracerProvider, nil
}
