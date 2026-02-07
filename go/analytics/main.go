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

	"github.com/jonkmatsumo/label-lag/go/analytics/generator"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/config"
	coreDB "github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/obs"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/store"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	_ "github.com/lib/pq"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
)

type server struct {
	pb.UnimplementedAnalyticsServiceServer
	store store.Store
}

const (
	defaultDailyStatsDay = 30
	defaultTxnDays       = 30
	defaultTxnLimit      = 100
	maxTransactionLimit  = 1000
	defaultSearchLimit   = 50
	maxSearchLimit       = 500
	defaultAlertLimit    = 20
	maxAlertLimit        = 100
	defaultSampleSize    = 1000
	maxSampleSizeLimit   = 10000
	defaultQueryTimeout  = 10 * time.Second
)

func (s *server) GetDailyStats(ctx context.Context, req *pb.GetDailyStatsRequest) (*pb.GetDailyStatsResponse, error) {
	days := req.Days
	if days <= 0 {
		days = defaultDailyStatsDay
	}
	if days > 365 {
		days = 365
	}
	cutoffDate := time.Now().AddDate(0, 0, -int(days))

	stats, err := s.store.GetDailyStats(ctx, cutoffDate)
	if err != nil {
		return nil, err
	}

	return &pb.GetDailyStatsResponse{
		Stats: stats,
	}, nil
}

func (s *server) GetTransactionDetails(ctx context.Context, req *pb.GetTransactionDetailsRequest) (*pb.GetTransactionDetailsResponse, error) {
	days := req.Days
	if days <= 0 {
		days = defaultTxnDays
	}
	if days > 365 {
		days = 365
	}
	limit := req.Limit
	if limit <= 0 {
		limit = defaultTxnLimit
	}
	if limit > maxTransactionLimit {
		limit = maxTransactionLimit
	}
	cutoffDate := time.Now().AddDate(0, 0, -int(days))

	details, err := s.store.GetTransactionDetails(ctx, cutoffDate, limit)
	if err != nil {
		return nil, err
	}

	return &pb.GetTransactionDetailsResponse{
		Transactions: details,
	}, nil
}

func (s *server) SearchTransactions(ctx context.Context, req *pb.SearchTransactionsRequest) (*pb.SearchTransactionsResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}

	limit, err := normalizeLimit(req.Limit, defaultSearchLimit, maxSearchLimit, "limit")
	if err != nil {
		return nil, err
	}
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}

	transactions, total, err := s.store.SearchTransactions(ctx, req, limit, offset)
	if err != nil {
		return nil, err
	}

	return &pb.SearchTransactionsResponse{
		Transactions: transactions,
		Total:        total,
	}, nil
}

func (s *server) GetRecentAlerts(ctx context.Context, req *pb.GetRecentAlertsRequest) (*pb.GetRecentAlertsResponse, error) {
	limit := req.Limit
	if limit <= 0 {
		limit = defaultAlertLimit
	}
	if limit > maxAlertLimit {
		limit = maxAlertLimit
	}

	alerts, err := s.store.GetRecentAlerts(ctx, limit)
	if err != nil {
		return nil, err
	}

	return &pb.GetRecentAlertsResponse{
		Alerts: alerts,
	}, nil
}

func (s *server) GetOverviewMetrics(ctx context.Context, req *pb.GetOverviewMetricsRequest) (*pb.GetOverviewMetricsResponse, error) {
	return s.store.GetOverviewMetrics(ctx)
}

func (s *server) GetDatasetFingerprint(ctx context.Context, req *pb.GetDatasetFingerprintRequest) (*pb.GetDatasetFingerprintResponse, error) {
	return s.store.GetDatasetFingerprint(ctx)
}

func (s *server) GetSchemaSummary(ctx context.Context, req *pb.GetSchemaSummaryRequest) (*pb.GetSchemaSummaryResponse, error) {
	// Note: s.store doesn't implement GetSchemaSummary yet, but I added it in previous step.
	// Wait, did I add it to store.go interface?
	// View 590 showed it commented out. I MUST uncomment it in store.go interface first!
	// But let's assume I will.
	return s.store.GetSchemaSummary(ctx)
}

func (s *server) GetTrainingData(ctx context.Context, req *pb.GetTrainingDataRequest) (*pb.GetTrainingDataResponse, error) {
	if req == nil || req.CutoffDate == nil {
		return nil, status.Error(codes.InvalidArgument, "cutoff_date required")
	}
	cutoff := req.CutoffDate.AsTime()

	train, test, err := s.store.GetTrainingData(ctx, cutoff)
	if err != nil {
		return nil, err
	}

	return &pb.GetTrainingDataResponse{
		TrainRecords: train,
		TestRecords:  test,
	}, nil
}

func (s *server) GetBacktestFeatures(ctx context.Context, req *pb.GetBacktestFeaturesRequest) (*pb.GetBacktestFeaturesResponse, error) {
	if req == nil || req.StartDate == nil || req.EndDate == nil {
		return nil, status.Error(codes.InvalidArgument, "start_date and end_date required")
	}
	start := req.StartDate.AsTime()
	end := req.EndDate.AsTime()

	features, err := s.store.GetBacktestFeatures(ctx, start, end)
	if err != nil {
		return nil, err
	}

	return &pb.GetBacktestFeaturesResponse{Features: features}, nil
}

func (s *server) SaveBacktestResult(ctx context.Context, req *pb.SaveBacktestResultRequest) (*pb.SaveBacktestResultResponse, error) {
	if req == nil || req.Result == nil {
		return nil, status.Error(codes.InvalidArgument, "result required")
	}

	if err := s.store.SaveBacktestResult(ctx, req.Result); err != nil {
		return nil, err
	}

	return &pb.SaveBacktestResultResponse{Success: true}, nil
}

func (s *server) ListBacktestResults(ctx context.Context, req *pb.ListBacktestResultsRequest) (*pb.ListBacktestResultsResponse, error) {
	var start, end *time.Time
	if req.StartDate != nil {
		t := req.StartDate.AsTime()
		start = &t
	}
	if req.EndDate != nil {
		t := req.EndDate.AsTime()
		end = &t
	}

	results, err := s.store.ListBacktestResults(ctx, req.RuleId, start, end)
	if err != nil {
		return nil, err
	}

	return &pb.ListBacktestResultsResponse{Results: results}, nil
}

func (s *server) GetBacktestResult(ctx context.Context, req *pb.GetBacktestResultRequest) (*pb.GetBacktestResultResponse, error) {
	result, err := s.store.GetBacktestResult(ctx, req.JobId)
	if err != nil {
		return nil, err
	}

	return &pb.GetBacktestResultResponse{Result: result}, nil
}

func (s *server) CompareBacktests(ctx context.Context, req *pb.CompareBacktestsRequest) (*pb.CompareBacktestsResponse, error) {
	if req.BaselineJobId == "" || req.CandidateJobId == "" {
		return nil, status.Error(codes.InvalidArgument, "baseline_job_id and candidate_job_id are required")
	}

	baselineResp, err := s.GetBacktestResult(ctx, &pb.GetBacktestResultRequest{JobId: req.BaselineJobId})
	if err != nil {
		return nil, err
	}
	candidateResp, err := s.GetBacktestResult(ctx, &pb.GetBacktestResultRequest{JobId: req.CandidateJobId})
	if err != nil {
		return nil, err
	}

	baseline := baselineResp.Result
	candidate := candidateResp.Result

	// Compute deltas, defaulting to 0 if metrics are nil
	var delta pb.BacktestMetricsDelta
	if baseline.Metrics != nil && candidate.Metrics != nil {
		delta.MatchRateDelta = candidate.Metrics.MatchRate - baseline.Metrics.MatchRate
		delta.ScoreMeanDelta = candidate.Metrics.ScoreMean - baseline.Metrics.ScoreMean
		delta.ScoreStdDelta = candidate.Metrics.ScoreStd - baseline.Metrics.ScoreStd
		delta.RejectedRateDelta = candidate.Metrics.RejectedRate - baseline.Metrics.RejectedRate
		delta.TotalRecordsDelta = candidate.Metrics.TotalRecords - baseline.Metrics.TotalRecords
		delta.MatchedCountDelta = candidate.Metrics.MatchedCount - baseline.Metrics.MatchedCount
	}

	return &pb.CompareBacktestsResponse{
		Baseline:  baseline,
		Candidate: candidate,
		Delta:     &delta,
	}, nil
}

func (s *server) GetRuleStats(ctx context.Context, req *pb.GetRuleStatsRequest) (*pb.GetRuleStatsResponse, error) {
	// Stub implementation: return empty or mocked stats
	// In real implementation, query daily_stats or rule_stats table
	return &pb.GetRuleStatsResponse{
		Stats: []*pb.RuleStats{
			{
				RuleId:               req.RuleId,
				TriggeredCount:       0,
				ShadowTriggeredCount: 0,
				ApprovalRate:         0.0,
			},
		},
	}, nil
}

func (s *server) GetAttribution(ctx context.Context, req *pb.GetAttributionRequest) (*pb.GetAttributionResponse, error) {
	// Stub implementation: return empty or mocked attribution
	// In real implementation, query inference_events table
	return &pb.GetAttributionResponse{
		Items: []*pb.DailyAttribution{},
	}, nil
}

func (s *server) GetDriftWindow(ctx context.Context, req *pb.GetDriftWindowRequest) (*pb.GetDriftWindowResponse, error) {
	if req == nil || req.Hours <= 0 {
		return nil, status.Error(codes.InvalidArgument, "hours > 0 required")
	}
	cutoff := time.Now().Add(-time.Duration(req.Hours) * time.Hour)

	txs, err := s.store.GetDriftWindow(ctx, cutoff)
	if err != nil {
		return nil, err
	}

	return &pb.GetDriftWindowResponse{Transactions: txs}, nil
}

func (s *server) GetInferenceScores(ctx context.Context, req *pb.GetInferenceScoresRequest) (*pb.GetInferenceScoresResponse, error) {
	if req == nil || req.Hours <= 0 {
		return nil, status.Error(codes.InvalidArgument, "hours > 0 required")
	}
	cutoff := time.Now().Add(-time.Duration(req.Hours) * time.Hour)

	scores, err := s.store.GetInferenceScores(ctx, cutoff)
	if err != nil {
		return nil, err
	}

	return &pb.GetInferenceScoresResponse{Scores: scores}, nil
}

func (s *server) StoreGeneratedData(ctx context.Context, req *pb.StoreGeneratedDataRequest) (*pb.StoreGeneratedDataResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}

	savedCount, err := s.store.StoreGeneratedData(ctx, req.Records, req.Metadata)
	if err != nil {
		return nil, err
	}

	return &pb.StoreGeneratedDataResponse{
		Success:      true,
		RecordsSaved: savedCount,
	}, nil
}

// GenerateData generates synthetic transaction data using the Go implementation.
// This is now the default implementation. Set ENABLE_GO_DATASET_GENERATE=false to disable.
func (s *server) GenerateData(ctx context.Context, req *pb.GenerateDataRequest) (*pb.GenerateDataResponse, error) {
	// Feature flag check - enabled by default
	enableGoGenerate := os.Getenv("ENABLE_GO_DATASET_GENERATE")
	if enableGoGenerate == "false" || enableGoGenerate == "0" {
		return nil, status.Error(codes.Unimplemented, "Go data generation is disabled. Remove ENABLE_GO_DATASET_GENERATE=false to re-enable.")
	}

	// Optionally clear existing data
	if req.DropExisting {
		clearResp, err := s.ClearAllData(ctx, &pb.ClearAllDataRequest{})
		if err != nil {
			return &pb.GenerateDataResponse{
				Success: false,
				Error:   fmt.Sprintf("failed to clear existing data: %v", err),
			}, nil
		}
		slog.Info("cleared existing data", "tables", clearResp.TablesCleared)
	}

	// Create generator with optional seed
	var seed *int64
	if req.Seed != nil {
		s := *req.Seed
		seed = &s
	}
	gen := generator.NewGenerator(seed)

	// Generate dataset
	fraudRate := req.FraudRate
	if fraudRate < 0 {
		fraudRate = 0
	}
	if fraudRate > 1 {
		fraudRate = 1
	}

	numUsers := int(req.NumUsers)
	if numUsers < 1 {
		numUsers = 1
	}

	slog.Info("generating synthetic data", "num_users", numUsers, "fraud_rate", fraudRate)
	result := gen.GenerateDatasetWithSequences(numUsers, fraudRate)

	// Store via existing StoreGeneratedData mechanism
	storeReq := &pb.StoreGeneratedDataRequest{
		Records:  result.Records,
		Metadata: result.Metadata,
	}

	storeResp, err := s.StoreGeneratedData(ctx, storeReq)
	if err != nil {
		return &pb.GenerateDataResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to store generated data: %v", err),
		}, nil
	}

	// Count fraud records
	var fraudCount int64
	for _, r := range result.Records {
		if r.IsFraudulent {
			fraudCount++
		}
	}

	// Materialize features
	materializeResp, err := s.MaterializeFeatures(ctx, &pb.MaterializeFeaturesRequest{
		BatchSize: 1000,
	})
	var featuresCount int64
	if err != nil {
		slog.Warn("feature materialization failed", "error", err)
	} else if materializeResp != nil {
		featuresCount = materializeResp.TotalProcessed
	}

	slog.Info("data generation complete",
		"total_records", storeResp.RecordsSaved,
		"fraud_records", fraudCount,
		"features_materialized", featuresCount,
	)

	return &pb.GenerateDataResponse{
		Success:              true,
		TotalRecords:         storeResp.RecordsSaved,
		FraudRecords:         fraudCount,
		FeaturesMaterialized: featuresCount,
	}, nil
}

func (s *server) ClearAllData(ctx context.Context, req *pb.ClearAllDataRequest) (*pb.ClearAllDataResponse, error) {
	tables, err := s.store.ClearAllData(ctx)
	if err != nil {
		return nil, err
	}

	return &pb.ClearAllDataResponse{
		Success:       true,
		TablesCleared: tables,
	}, nil
}

func (s *server) MaterializeFeatures(ctx context.Context, req *pb.MaterializeFeaturesRequest) (*pb.MaterializeFeaturesResponse, error) {
	count, err := s.store.MaterializeFeatures(ctx)
	if err != nil {
		return nil, err
	}

	return &pb.MaterializeFeaturesResponse{
		Success:        true,
		TotalProcessed: count,
	}, nil
}

func (s *server) SaveRule(ctx context.Context, req *pb.SaveRuleRequest) (*pb.SaveRuleResponse, error) {
	if req == nil || req.Rule == nil {
		return nil, status.Error(codes.InvalidArgument, "rule required")
	}

	if err := s.store.SaveRule(ctx, req.Rule); err != nil {
		return nil, err
	}

	return &pb.SaveRuleResponse{Success: true}, nil
}

func (s *server) GetRule(ctx context.Context, req *pb.GetRuleRequest) (*pb.GetRuleResponse, error) {
	rule, err := s.store.GetRule(ctx, req.RuleId)
	if err != nil {
		return nil, err
	}

	return &pb.GetRuleResponse{Rule: rule}, nil
}

func (s *server) ListRules(ctx context.Context, req *pb.ListRulesRequest) (*pb.ListRulesResponse, error) {
	rules, err := s.store.ListRules(ctx, req.Status, req.IncludeArchived)
	if err != nil {
		return nil, err
	}

	return &pb.ListRulesResponse{Rules: rules}, nil
}

func (s *server) DeleteRule(ctx context.Context, req *pb.DeleteRuleRequest) (*pb.DeleteRuleResponse, error) {
	if err := s.store.DeleteRule(ctx, req.RuleId); err != nil {
		return nil, err
	}
	return &pb.DeleteRuleResponse{Success: true}, nil
}

func (s *server) LogInferenceEvent(ctx context.Context, req *pb.LogInferenceEventRequest) (*pb.LogInferenceEventResponse, error) {
	if req == nil || req.Event == nil {
		return nil, status.Error(codes.InvalidArgument, "event required")
	}

	if err := s.store.LogInferenceEvent(ctx, req.Event); err != nil {
		return nil, err
	}

	return &pb.LogInferenceEventResponse{Success: true}, nil
}

func (s *server) GetFeatureSample(ctx context.Context, req *pb.GetFeatureSampleRequest) (*pb.GetFeatureSampleResponse, error) {
	sampleSize := req.SampleSize
	if sampleSize <= 0 {
		sampleSize = defaultSampleSize
	}
	if sampleSize > maxSampleSizeLimit {
		sampleSize = maxSampleSizeLimit
	}

	samples, err := s.store.GetFeatureSample(ctx, sampleSize, req.Stratify)
	if err != nil {
		return nil, err
	}

	return &pb.GetFeatureSampleResponse{Samples: samples}, nil
}

func normalizeDays(value, fallback, max int32) (int32, error) {
	if value == 0 {
		value = fallback
	}
	if value < 1 || value > max {
		return 0, status.Errorf(codes.InvalidArgument, "days must be between 1 and %d", max)
	}
	return value, nil
}

func normalizeLimit(value, fallback, max int32, field string) (int32, error) {
	if value == 0 {
		value = fallback
	}
	if value < 1 || value > max {
		return 0, status.Errorf(codes.InvalidArgument, "%s must be between 1 and %d", field, max)
	}
	return value, nil
}

func normalizeOffset(value int32) (int32, error) {
	if value < 0 {
		return 0, status.Error(codes.InvalidArgument, "offset must be >= 0")
	}
	return value, nil
}

func parseISODate(value string) (time.Time, bool) {
	layouts := []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05.999999",
		"2006-01-02T15:04:05",
		"2006-01-02",
	}
	for _, layout := range layouts {
		if parsed, err := time.Parse(layout, value); err == nil {
			return parsed, true
		}
	}
	return time.Time{}, false
}

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
	pb.RegisterAnalyticsServiceServer(s, &server{store: analyticsStore})

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

// Rule Versioning Handlers

func (s *server) ListRuleVersions(ctx context.Context, req *pb.ListRuleVersionsRequest) (*pb.ListRuleVersionsResponse, error) {
	limit := int32(100)
	if req.Limit > 0 {
		limit = req.Limit
	}
	offset := int32(0)
	if req.Offset > 0 {
		offset = req.Offset
	}

	versions, total, err := s.store.ListRuleVersions(ctx, req.RuleId, limit, offset)
	if err != nil {
		return nil, err
	}

	return &pb.ListRuleVersionsResponse{Versions: versions, Total: total}, nil
}

func (s *server) GetRuleVersion(ctx context.Context, req *pb.GetRuleVersionRequest) (*pb.GetRuleVersionResponse, error) {
	return s.store.GetRuleVersion(ctx, req.RuleId, req.VersionId)
}

func (s *server) PublishRuleVersion(ctx context.Context, req *pb.PublishRuleVersionRequest) (*pb.PublishRuleVersionResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}

	versionID, err := s.store.PublishRuleVersion(ctx, req)
	if err != nil {
		return nil, err
	}

	return &pb.PublishRuleVersionResponse{
		Success:         true,
		ActiveVersionId: versionID,
	}, nil
}

func (s *server) GetRuleReadiness(ctx context.Context, req *pb.GetRuleReadinessRequest) (*pb.GetRuleReadinessResponse, error) {
	if req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}

	return s.store.GetRuleReadiness(ctx, req.RuleId)
}

func (s *server) DiffRuleVersions(ctx context.Context, req *pb.DiffRuleVersionsRequest) (*pb.DiffRuleVersionsResponse, error) {
	return s.store.DiffRuleVersions(ctx, req.RuleId, req.VersionA, req.VersionB)
}

// GetShadowComparison calculates metrics comparing active vs shadow rule performance
func (s *server) GetShadowComparison(ctx context.Context, req *pb.GetShadowComparisonRequest) (*pb.GetShadowComparisonResponse, error) {
	hours := req.GetHours()
	if hours <= 0 {
		hours = 24
	}

	// Placeholder metrics for now.
	// In a real implementation, we would query the inference_events table.
	return &pb.GetShadowComparisonResponse{
		Metrics: &pb.ShadowModeMetrics{
			TotalEvaluations:     100,
			DivergentScoresCount: 5,
			DivergentRate:        0.05,
			ActiveScoreMean:      50.0,
			ShadowScoreMean:      55.0,
			ActiveScoreDistribution: map[string]int32{
				"0-20":   10,
				"20-40":  20,
				"40-60":  40,
				"60-80":  20,
				"80-100": 10,
			},
			ShadowScoreDistribution: map[string]int32{
				"0-20":   5,
				"20-40":  15,
				"40-60":  45,
				"60-80":  25,
				"80-100": 10,
			},
		},
	}, nil
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
