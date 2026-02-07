package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	grpcclient "github.com/jonkmatsumo/label-lag/go/inference/internal/grpc"
	httpserver "github.com/jonkmatsumo/label-lag/go/inference/internal/http"
	"github.com/jonkmatsumo/label-lag/go/inference/internal/rules"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"google.golang.org/grpc"
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

	analyticsClient, err := grpcclient.NewAnalyticsClient("", 0)
	if err != nil {
		logger.Error("failed to create analytics client", "error", err)
		os.Exit(1)
	}
	defer func() {
		if err := analyticsClient.Close(); err != nil {
			logger.Warn("failed to close analytics client", "error", err)
		}
	}()

	rulesProvider := rules.Provider(rules.NewEmptyProvider())
	if apiURL := os.Getenv("INFERENCE_GATEWAY_API_URL"); apiURL != "" {
		ttl := 5 * time.Minute
		if ttlStr := os.Getenv("INFERENCE_GATEWAY_RULES_TTL"); ttlStr != "" {
			if d, err := time.ParseDuration(ttlStr); err == nil {
				ttl = d
			}
		}
		logger.Info("using API rules provider", "url", apiURL, "ttl", ttl)
		rulesProvider = rules.NewAPIProvider(apiURL, ttl)
	} else if rulesPath := os.Getenv("INFERENCE_GATEWAY_RULES_PATH"); rulesPath != "" {
		fileProvider, err := rules.NewFileProvider(rulesPath, logger)
		if err != nil {
			logger.Error("failed to load rules file", "error", err)
			os.Exit(1)
		}
		logger.Info("using file rules provider", "path", rulesPath)
		rulesProvider = fileProvider
	}

	maxBodyBytes := int64(1 << 20)
	if maxBodyStr := os.Getenv("INFERENCE_GATEWAY_MAX_BODY_BYTES"); maxBodyStr != "" {
		if parsed, err := strconv.ParseInt(maxBodyStr, 10, 64); err == nil && parsed > 0 {
			maxBodyBytes = parsed
		} else {
			logger.Warn("invalid INFERENCE_GATEWAY_MAX_BODY_BYTES", "value", maxBodyStr)
		}
	}

	handler := httpserver.NewHandler(logger, inferenceClient, analyticsClient, rulesProvider, maxBodyBytes,
		getEnv("PYTHON_API_URL", "http://api:8000"),
		getEnv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
	)
	readTimeout := parseDurationEnv(logger, "INFERENCE_GATEWAY_READ_TIMEOUT", 10*time.Second)
	writeTimeout := parseDurationEnv(logger, "INFERENCE_GATEWAY_WRITE_TIMEOUT", 30*time.Second)
	idleTimeout := parseDurationEnv(logger, "INFERENCE_GATEWAY_IDLE_TIMEOUT", 60*time.Second)

	// Start HTTP server
	srv := httpserver.NewServer("0.0.0.0:"+port, logger, handler, readTimeout, writeTimeout, idleTimeout)

	errCh := make(chan error, 2)
	go func() {
		logger.Info("http server listening", "address", srv.Addr)
		errCh <- srv.ListenAndServe()
	}()

	// Start gRPC server
	grpcPort := os.Getenv("GRPC_PORT")
	if grpcPort == "" {
		grpcPort = "50505"
	}
	lis, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%s", grpcPort))
	if err != nil {
		logger.Error("failed to listen for grpc", "error", err)
		os.Exit(1)
	}
	grpcServer := grpc.NewServer()
	gatewayServer := grpcclient.NewGatewayServer()
	gatewayServer.Register(grpcServer)

	go func() {
		logger.Info("grpc server listening", "address", lis.Addr())
		errCh <- grpcServer.Serve(lis)
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

	// Graceful shutdown
	grpcServer.GracefulStop()

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

func parseDurationEnv(logger *slog.Logger, key string, fallback time.Duration) time.Duration {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil || parsed <= 0 {
		logger.Warn("invalid duration env var", "key", key, "value", value)
		return fallback
	}
	return parsed
}

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
