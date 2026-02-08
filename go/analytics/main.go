package main

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/config"
	coreDB "github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/generator"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/obs"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/service"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/store"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	_ "github.com/lib/pq"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

// loggingInterceptor logs the details of each gRPC request and response.

func main() {
	// Configure structured logging
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	// Build context
	ctx := context.Background()

	// Initialize OpenTelemetry
	tp, err := obs.InitTracer(ctx)
	if err != nil {
		slog.Error("failed to initialize tracer", "error", err)
	} else if tp != nil {
		defer func() {
			if err := tp.Shutdown(ctx); err != nil {
				slog.Error("failed to shutdown tracer provider", "error", err)
			}
		}()
		slog.Info("opentelemetry tracer initialized")
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}

	dbURL, err := config.ResolveDatabaseURL(os.Getenv)
	if err != nil {
		slog.Error("failed to resolve database url", "error", err)
		os.Exit(1)
	}
	if dbURL == "" {
		slog.Error("DATABASE_URL is required but not set")
		os.Exit(1)
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		slog.Error("failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	if err := coreDB.InitDB(db); err != nil {
		slog.Error("failed to initialize database", "error", err)
		os.Exit(1)
	}

	// Configure connection pool
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(time.Hour)

	if err := db.Ping(); err != nil {
		slog.Warn("failed to ping database", "error", err)
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", port))
	if err != nil {
		slog.Error("failed to listen", "error", err)
		os.Exit(1)
	}

	// Add interceptors: logging and otel tracing
	opts := []grpc.ServerOption{
		grpc.ChainUnaryInterceptor(
			obs.RequestIDInterceptor,
			obs.LoggingInterceptor,
		),
		grpc.StatsHandler(otelgrpc.NewServerHandler()),
	}
	s := grpc.NewServer(opts...)
	analyticsStore := store.NewSQLStore(db)

	// Initialize generator registry
	// GENERATOR_CONFIG_PATH: Path to the generator configuration file
	// (default: go/analytics/config/generator.yaml)
	reg := generator.NewGeneratorRegistry()
	configPath := os.Getenv("GENERATOR_CONFIG_PATH")
	if configPath == "" {
		configPath = "go/analytics/config/generator.yaml" // Default for local dev if run from root
	}
	if err := reg.LoadConfig(configPath); err != nil {
		slog.Error("failed to load generator config", "error", err, "path", configPath)
		os.Exit(1)
	}

	svc := service.NewService(analyticsStore, reg)
	pb.RegisterAnalyticsServiceServer(s, svc)

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(s, healthServer)
	updateHealthStatus(context.Background(), db, healthServer, logger)

	// Register reflection service on gRPC server.
	reflection.Register(s)

	slog.Info("server listening", "address", lis.Addr())

	// Handle graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := s.Serve(lis); err != nil {
			slog.Error("failed to serve", "error", err)
			os.Exit(1)
		}
	}()

	healthCtx, healthCancel := context.WithCancel(context.Background())
	healthTicker := time.NewTicker(10 * time.Second)
	go func() {
		defer healthTicker.Stop()
		for {
			select {
			case <-healthCtx.Done():
				return
			case <-healthTicker.C:
				updateHealthStatus(context.Background(), db, healthServer, logger)
			}
		}
	}()

	<-stop
	healthCancel()
	slog.Info("shutting down gRPC server...")
	s.GracefulStop()
}

func updateHealthStatus(ctx context.Context, db *sql.DB, healthServer *health.Server, logger *slog.Logger) error {
	if err := db.PingContext(ctx); err != nil {
		if logger != nil {
			logger.Warn("database health check failed", "error", err)
		}
		healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		return err
	}
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	return nil
}
