package service

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/generator"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/store"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	commonv1 "github.com/jonkmatsumo/label-lag/go/common/proto/v1"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type Service struct {
	pb.UnimplementedAnalyticsServiceServer
	store    store.Store
	registry *generator.GeneratorRegistry
}

func NewService(store store.Store, registry *generator.GeneratorRegistry) *Service {
	return &Service{
		store:    store,
		registry: registry,
	}
}

const (
	defaultDailyStatsDay  = 30
	defaultTxnDays        = 30
	defaultTxnLimit       = 100
	maxTransactionLimit   = 1000
	defaultSearchLimit    = 50
	maxSearchLimit        = 500
	defaultAlertLimit     = 50
	maxAlertLimit         = 250
	defaultSampleSize     = 1000
	maxSampleSizeLimit    = 10000
	defaultRuleImpactDays = 7
	maxRuleImpactDays     = 90
)

var allowedDecisions = map[string]bool{
	store.DecisionApprove: true,
	store.DecisionReview:  true,
	store.DecisionReject:  true,
}

func (s *Service) GetDailyStats(ctx context.Context, req *pb.GetDailyStatsRequest) (*pb.GetDailyStatsResponse, error) {
	days, err := normalizeDays(req.Days, defaultDailyStatsDay, 365)
	if err != nil {
		return nil, err
	}
	cutoffDate := time.Now().AddDate(0, 0, -int(days))

	stats, err := s.store.GetDailyStats(ctx, cutoffDate, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetDailyStatsResponse{
		Stats: stats,
	}, nil
}

func (s *Service) GetTransactionDetails(ctx context.Context, req *pb.GetTransactionDetailsRequest) (*pb.GetTransactionDetailsResponse, error) {
	days, err := normalizeDays(req.Days, defaultTxnDays, 365)
	if err != nil {
		return nil, err
	}
	limit, truncated := normalizeLimit(req.Limit, defaultTxnLimit, maxTransactionLimit)
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}
	cutoffDate := time.Now().AddDate(0, 0, -int(days))

	details, err := s.store.GetTransactionDetails(ctx, cutoffDate, limit, offset, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetTransactionDetailsResponse{
		Transactions: details,
		Meta: &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		},
	}, nil
}

func (s *Service) SearchTransactions(ctx context.Context, req *pb.SearchTransactionsRequest) (*pb.SearchTransactionsResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}

	limit, truncated := normalizeLimit(req.Limit, defaultSearchLimit, maxSearchLimit)
	req.Limit = limit

	transactions, nextCursor, meta, err := s.store.SearchTransactions(ctx, req)
	if err != nil {
		return nil, err
	}

	// Ensure meta accurately reflects the service-level clamping as well
	if meta == nil {
		meta = &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		}
	} else {
		meta.Truncated = meta.Truncated || truncated
	}

	resp := &pb.SearchTransactionsResponse{
		Transactions: transactions,
		Truncated:    meta.Truncated,
		NextCursor:   nextCursor,
		Meta:         meta,
	}
	if nextCursor != "" {
		resp.Pagination = &commonv1.CursorPageResponse{
			NextCursor: nextCursor,
		}
	}
	return resp, nil
}

func (s *Service) GetRecentAlerts(ctx context.Context, req *pb.GetRecentAlertsRequest) (*pb.GetRecentAlertsResponse, error) {
	limit, truncated := normalizeLimit(req.Limit, defaultAlertLimit, maxAlertLimit)
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}

	alerts, err := s.store.GetRecentAlerts(ctx, limit, offset, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetRecentAlertsResponse{
		Alerts: alerts,
		Meta: &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		},
	}, nil
}

func (s *Service) GetOverviewMetrics(ctx context.Context, req *pb.GetOverviewMetricsRequest) (*pb.GetOverviewMetricsResponse, error) {
	return s.store.GetOverviewMetrics(ctx, req.TenantId)
}

func (s *Service) GetTrainingData(ctx context.Context, req *pb.GetTrainingDataRequest) (*pb.GetTrainingDataResponse, error) {
	if req == nil || req.CutoffDate == nil {
		return nil, status.Error(codes.InvalidArgument, "cutoff_date required")
	}
	cutoff := req.CutoffDate.AsTime()

	train, test, err := s.store.GetTrainingData(ctx, cutoff, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetTrainingDataResponse{
		TrainRecords: train,
		TestRecords:  test,
	}, nil
}

func (s *Service) GetBacktestFeatures(ctx context.Context, req *pb.GetBacktestFeaturesRequest) (*pb.GetBacktestFeaturesResponse, error) {
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

func (s *Service) SaveBacktestResult(ctx context.Context, req *pb.SaveBacktestResultRequest) (*pb.SaveBacktestResultResponse, error) {
	if req == nil || req.Result == nil {
		return nil, status.Error(codes.InvalidArgument, "result required")
	}

	if err := s.store.SaveBacktestResult(ctx, req.Result); err != nil {
		return nil, err
	}

	return &pb.SaveBacktestResultResponse{Success: true}, nil
}

func (s *Service) ListBacktestResults(ctx context.Context, req *pb.ListBacktestResultsRequest) (*pb.ListBacktestResultsResponse, error) {
	var start, end *time.Time
	if req.StartDate != nil {
		t := req.StartDate.AsTime()
		start = &t
	}
	if req.EndDate != nil {
		t := req.EndDate.AsTime()
		end = &t
	}

	limit, truncated := normalizeLimit(req.Limit, 50, 250)
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}

	results, err := s.store.ListBacktestResults(ctx, req.RuleId, start, end, limit, offset)
	if err != nil {
		return nil, err
	}

	return &pb.ListBacktestResultsResponse{
		Results: results,
		Meta: &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		},
	}, nil
}

func (s *Service) GetBacktestResult(ctx context.Context, req *pb.GetBacktestResultRequest) (*pb.GetBacktestResultResponse, error) {
	result, err := s.store.GetBacktestResult(ctx, req.JobId)
	if err != nil {
		return nil, err
	}

	return &pb.GetBacktestResultResponse{Result: result}, nil
}

func (s *Service) CompareBacktests(ctx context.Context, req *pb.CompareBacktestsRequest) (*pb.CompareBacktestsResponse, error) {
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

	var delta pb.BacktestMetricsDelta
	if baseline.Metrics != nil && candidate.Metrics != nil {
		delta.MatchRateDelta = candidate.Metrics.MatchRate - baseline.Metrics.MatchRate
		delta.ScoreMeanDelta = candidate.Metrics.ScoreMean - baseline.Metrics.ScoreMean
		delta.ScoreStdDelta = candidate.Metrics.ScoreStd - baseline.Metrics.ScoreStd
	}

	return &pb.CompareBacktestsResponse{
		Baseline:  baseline,
		Candidate: candidate,
		Delta:     &delta,
	}, nil
}

func (s *Service) GetShadowComparison(ctx context.Context, req *pb.GetShadowComparisonRequest) (*pb.GetShadowComparisonResponse, error) {
	hours, err := normalizeDays(req.Hours, 24, 720)
	if err != nil {
		return nil, err
	}

	metrics, err := s.store.GetShadowComparison(ctx, hours, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetShadowComparisonResponse{Metrics: metrics}, nil
}

func (s *Service) GetDatasetProfile(ctx context.Context, req *pb.GetDatasetProfileRequest) (*pb.GetDatasetProfileResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}

	limitFeatures, _ := normalizeLimit(req.LimitFeatures, 50, 500)
	numBuckets := req.NumBuckets
	if numBuckets <= 0 {
		numBuckets = 10
	}
	if numBuckets < 2 {
		numBuckets = 2
	}
	if numBuckets > 50 {
		numBuckets = 50
	}

	profile, err := s.store.GetDatasetProfile(ctx, req.DatasetId, limitFeatures, numBuckets, "")
	if err != nil {
		return nil, err
	}
	return profile, nil
}

func (s *Service) GenerateData(ctx context.Context, req *pb.GenerateDataRequest) (*pb.GenerateDataResponse, error) {
	tracer := otel.Tracer("analytics-crud")
	ctx, span := tracer.Start(ctx, "GenerateData", trace.WithAttributes(
		attribute.Int("num_users", int(req.NumUsers)),
		attribute.Float64("fraud_rate", req.FraudRate),
		attribute.String("idempotency_key", req.IdempotencyKey),
	))
	defer span.End()

	slog.Info("Starting dataset generation",
		"num_users", req.NumUsers,
		"fraud_rate", req.FraudRate,
		"idempotency_key", req.IdempotencyKey)

	// ENABLE_GO_DATASET_GENERATE: Set to "true" to enable this endpoint.
	if os.Getenv("ENABLE_GO_DATASET_GENERATE") != "true" {
		return nil, status.Error(codes.Unimplemented, "dataset generation is disabled")
	}

	// Validation
	if req.NumUsers <= 0 {
		return nil, status.Error(codes.InvalidArgument, "num_users must be positive")
	}
	if req.NumUsers > 10000 {
		return nil, status.Error(codes.InvalidArgument, "num_users exceeds maximum limit of 10000")
	}
	if req.FraudRate < 0 || req.FraudRate > 1.0 {
		return nil, status.Error(codes.InvalidArgument, "fraud_rate must be between 0.0 and 1.0")
	}

	// Idempotency check
	if req.IdempotencyKey != "" {
		existing, jobStatus, err := s.store.GetGenerationJob(ctx, req.IdempotencyKey)
		if err != nil {
			return nil, err
		}
		if existing != nil {
			if jobStatus == "in_progress" {
				return nil, status.Error(codes.AlreadyExists, "generation job already in progress")
			}
			return existing, nil
		}

		if err := s.store.CreateGenerationJob(ctx, req.IdempotencyKey); err != nil {
			return nil, err
		}
	}

	seed := time.Now().UnixNano()
	if req.Seed != nil {
		seed = *req.Seed
	}

	gen := generator.NewGenerator(&seed, s.registry)
	result := gen.GenerateDatasetWithSequences(ctx, int(req.NumUsers), req.FraudRate)

	// Check for context cancellation after heavy generation
	if ctx.Err() != nil {
		if req.IdempotencyKey != "" {
			s.store.FailGenerationJob(context.Background(), req.IdempotencyKey, ctx.Err().Error())
		}
		return nil, ctx.Err()
	}

	// Count fraud records for the response
	var fraudCount int64
	for _, r := range result.Records {
		if r.IsFraudulent {
			fraudCount++
		}
	}

	if req.DropExisting {
		if _, err := s.store.ClearAllData(ctx); err != nil {
			if req.IdempotencyKey != "" {
				s.store.FailGenerationJob(context.Background(), req.IdempotencyKey, err.Error())
			}
			return nil, err
		}
	}

	count, err := s.store.StoreGeneratedData(ctx, result.Records, result.Metadata)
	if err != nil {
		if req.IdempotencyKey != "" {
			s.store.FailGenerationJob(context.Background(), req.IdempotencyKey, err.Error())
		}
		return nil, err
	}

	// Trigger feature materialization automatically after generation
	materialized, _ := s.store.MaterializeFeatures(ctx)

	resp := &pb.GenerateDataResponse{
		Success:              true,
		TotalRecords:         count,
		FraudRecords:         fraudCount,
		FeaturesMaterialized: materialized,
	}

	if req.IdempotencyKey != "" {
		if err := s.store.CompleteGenerationJob(ctx, req.IdempotencyKey, resp); err != nil {
			slog.Error("failed to complete generation job", "error", err, "idempotency_key", req.IdempotencyKey)
		}
	}

	span.SetAttributes(
		attribute.Int64("total_records", resp.TotalRecords),
		attribute.Int64("fraud_records", resp.FraudRecords),
		attribute.Int64("features_materialized", resp.FeaturesMaterialized),
	)

	slog.Info("Completed dataset generation",
		"total_records", resp.TotalRecords,
		"fraud_records", resp.FraudRecords,
		"features_materialized", resp.FeaturesMaterialized,
		"idempotency_key", req.IdempotencyKey)

	return resp, nil
}

func (s *Service) ClearAllData(ctx context.Context, _ *pb.ClearAllDataRequest) (*pb.ClearAllDataResponse, error) {
	tables, err := s.store.ClearAllData(ctx)
	if err != nil {
		return nil, err
	}
	return &pb.ClearAllDataResponse{Success: true, TablesCleared: tables}, nil
}

func (s *Service) MaterializeFeatures(ctx context.Context, _ *pb.MaterializeFeaturesRequest) (*pb.MaterializeFeaturesResponse, error) {
	count, err := s.store.MaterializeFeatures(ctx)
	if err != nil {
		return nil, err
	}

	return &pb.MaterializeFeaturesResponse{
		Success:        true,
		TotalProcessed: count,
	}, nil
}

func (s *Service) SaveRule(ctx context.Context, req *pb.SaveRuleRequest) (*pb.SaveRuleResponse, error) {
	if req == nil || req.Rule == nil {
		return nil, status.Error(codes.InvalidArgument, "rule required")
	}

	if err := s.store.SaveRule(ctx, req.Rule, req.TenantId); err != nil {
		return nil, err
	}

	return &pb.SaveRuleResponse{Success: true}, nil
}

func (s *Service) GetRule(ctx context.Context, req *pb.GetRuleRequest) (*pb.GetRuleResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}
	resp, err := s.store.GetRule(ctx, req.RuleId, req.TenantId)
	if err != nil {
		return nil, err
	}
	return &pb.GetRuleResponse{Rule: resp}, nil
}

func (s *Service) ListRules(ctx context.Context, req *pb.ListRulesRequest) (*pb.ListRulesResponse, error) {
	rules, err := s.store.ListRules(ctx, req.Status, req.IncludeArchived, req.TenantId)
	if err != nil {
		return nil, err
	}
	return &pb.ListRulesResponse{Rules: rules}, nil
}

func (s *Service) DeleteRule(ctx context.Context, req *pb.DeleteRuleRequest) (*pb.DeleteRuleResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}
	err := s.store.DeleteRule(ctx, req.RuleId, req.TenantId)
	if err != nil {
		return nil, err
	}
	return &pb.DeleteRuleResponse{Success: true}, nil
}

func (s *Service) LogInferenceEvent(ctx context.Context, req *pb.LogInferenceEventRequest) (*pb.LogInferenceEventResponse, error) {
	if req == nil || req.Event == nil {
		return nil, status.Error(codes.InvalidArgument, "event required")
	}
	if req.Event.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "event.tenant_id required")
	}
	err := s.store.LogInferenceEvent(ctx, req.Event)
	if err != nil {
		return nil, err
	}
	return &pb.LogInferenceEventResponse{Success: true}, nil
}

func (s *Service) GetFeatureSample(ctx context.Context, req *pb.GetFeatureSampleRequest) (*pb.GetFeatureSampleResponse, error) {
	sampleSize := req.SampleSize
	if sampleSize <= 0 {
		sampleSize = defaultSampleSize
	}
	if sampleSize > maxSampleSizeLimit {
		sampleSize = maxSampleSizeLimit
	}

	samples, err := s.store.GetFeatureSample(ctx, sampleSize, req.Stratify, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetFeatureSampleResponse{Samples: samples}, nil
}

func (s *Service) GetDriftWindow(ctx context.Context, req *pb.GetDriftWindowRequest) (*pb.GetDriftWindowResponse, error) {
	if req == nil || req.Hours <= 0 {
		return nil, status.Error(codes.InvalidArgument, "hours > 0 required")
	}
	cutoff := time.Now().Add(-time.Duration(req.Hours) * time.Hour)

	txs, err := s.store.GetDriftWindow(ctx, cutoff, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetDriftWindowResponse{Transactions: txs}, nil
}

func (s *Service) GetInferenceScores(ctx context.Context, req *pb.GetInferenceScoresRequest) (*pb.GetInferenceScoresResponse, error) {
	if req == nil || req.Hours <= 0 {
		return nil, status.Error(codes.InvalidArgument, "hours > 0 required")
	}
	cutoff := time.Now().Add(-time.Duration(req.Hours) * time.Hour)

	scores, err := s.store.GetInferenceScores(ctx, cutoff, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetInferenceScoresResponse{Scores: scores}, nil
}

func (s *Service) ListRuleVersions(ctx context.Context, req *pb.ListRuleVersionsRequest) (*pb.ListRuleVersionsResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}

	limit, truncated := normalizeLimit(req.Limit, 50, 250)
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}

	versions, total, err := s.store.ListRuleVersions(ctx, req.RuleId, limit, offset, req.TenantId)
	if err != nil {
		return nil, errVersionNotFound(err)
	}

	return &pb.ListRuleVersionsResponse{
		Versions: versions,
		Total:    total,
		Meta: &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		},
	}, nil
}

func (s *Service) GetRuleVersion(ctx context.Context, req *pb.GetRuleVersionRequest) (*pb.GetRuleVersionResponse, error) {
	if req == nil || req.RuleId == "" || req.VersionId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id and version_id required")
	}
	return s.store.GetRuleVersion(ctx, req.RuleId, req.VersionId, req.TenantId)
}

func (s *Service) PublishRuleVersion(ctx context.Context, req *pb.PublishRuleVersionRequest) (*pb.PublishRuleVersionResponse, error) {
	if req == nil || req.RuleId == "" || req.VersionId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id and version_id required")
	}

	activeID, err := s.store.PublishRuleVersion(ctx, req)
	if err != nil {
		return nil, errVersionNotFound(err)
	}

	return &pb.PublishRuleVersionResponse{
		Success:         true,
		ActiveVersionId: activeID,
	}, nil
}

func (s *Service) DiffRuleVersions(ctx context.Context, req *pb.DiffRuleVersionsRequest) (*pb.DiffRuleVersionsResponse, error) {
	if req == nil || req.RuleId == "" || req.VersionA == "" || req.VersionB == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id and both version IDs required")
	}
	return s.store.DiffRuleVersions(ctx, req.RuleId, req.VersionA, req.VersionB, req.TenantId)
}

func (s *Service) GetRuleReadiness(ctx context.Context, req *pb.GetRuleReadinessRequest) (*pb.GetRuleReadinessResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}
	return s.store.GetRuleReadiness(ctx, req.RuleId, req.TenantId)
}

func (s *Service) GetRuleStats(ctx context.Context, req *pb.GetRuleStatsRequest) (*pb.GetRuleStatsResponse, error) {
	days, err := normalizeDays(req.Days, 30, 365)
	if err != nil {
		return nil, err
	}
	cutoff := time.Now().AddDate(0, 0, -int(days))

	stats, err := s.store.GetRuleStats(ctx, req.RuleId, cutoff, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetRuleStatsResponse{
		Stats: stats,
	}, nil
}

func (s *Service) GetAttribution(ctx context.Context, req *pb.GetAttributionRequest) (*pb.GetAttributionResponse, error) {
	days, err := normalizeDays(req.Days, 7, 90)
	if err != nil {
		return nil, err
	}
	limit, truncated := normalizeLimit(req.Limit, 50, 250)
	cutoff := time.Now().AddDate(0, 0, -int(days))

	items, err := s.store.GetAttribution(ctx, cutoff, limit, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetAttributionResponse{
		Items: items,
		Meta: &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		},
	}, nil
}

func (s *Service) GetLatestUserFeatures(ctx context.Context, req *pb.GetLatestUserFeaturesRequest) (*pb.GetLatestUserFeaturesResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}

	features, found, err := s.store.GetLatestUserFeatures(ctx, req.UserId, req.TenantId)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, status.Errorf(codes.NotFound, "no features found for user %s", req.UserId)
	}

	return &pb.GetLatestUserFeaturesResponse{
		Features: features,
	}, nil
}

func (s *Service) BatchGetLatestUserFeatures(ctx context.Context, req *pb.BatchGetLatestUserFeaturesRequest) (*pb.BatchGetLatestUserFeaturesResponse, error) {
	if len(req.UserIds) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one user_id is required")
	}
	if len(req.UserIds) > 500 {
		return nil, status.Error(codes.InvalidArgument, "batch size cannot exceed 500")
	}

	results, err := s.store.BatchGetLatestUserFeatures(ctx, req.UserIds, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.BatchGetLatestUserFeaturesResponse{
		Features: results,
	}, nil
}

func (s *Service) ListDecisions(ctx context.Context, req *pb.ListDecisionsRequest) (*pb.ListDecisionsResponse, error) {
	limit, truncated := normalizeLimit(req.Limit, 50, 250)
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}
	req.Limit = limit
	req.Offset = offset

	// If cursor pagination is provided, normalize it too
	if req.Pagination != nil {
		req.Pagination.Limit, _ = normalizeLimit(req.Pagination.Limit, limit, 250)
	}

	if req.Decision != "" {
		req.Decision = strings.ToUpper(strings.TrimSpace(req.Decision))
		if !allowedDecisions[req.Decision] {
			return nil, status.Errorf(codes.InvalidArgument, "invalid decision: %s", req.Decision)
		}
	}

	if req.MinScore > 0 && req.MaxScore > 0 && req.MinScore > req.MaxScore {
		return nil, status.Error(codes.InvalidArgument, "min_score must be <= max_score")
	}

	if req.StartDate != nil && req.EndDate != nil && req.StartDate.AsTime().After(req.EndDate.AsTime()) {
		return nil, status.Error(codes.InvalidArgument, "start_date must be <= end_date")
	}

	decisions, total, nextCursor, err := s.store.ListDecisions(ctx, req)
	if err != nil {
		return nil, err
	}

	resp := &pb.ListDecisionsResponse{
		Decisions: decisions,
		Pagination: &commonv1.CursorPageResponse{
			NextCursor: nextCursor,
		},
		Meta: &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		},
	}
	if req.Pagination == nil || req.Pagination.Cursor == "" {
		resp.Pagination.Total = &total
	}
	return resp, nil
}

func (s *Service) GetDecision(ctx context.Context, req *pb.GetDecisionRequest) (*pb.GetDecisionResponse, error) {
	if req.RequestId == "" {
		return nil, status.Error(codes.InvalidArgument, "request_id required")
	}

	decision, err := s.store.GetDecision(ctx, req.RequestId, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetDecisionResponse{
		Decision: decision,
	}, nil
}

func (s *Service) GetDecisionTrace(ctx context.Context, req *pb.GetDecisionTraceRequest) (*pb.GetDecisionTraceResponse, error) {
	if req.RequestId == "" {
		return nil, status.Error(codes.InvalidArgument, "request_id required")
	}

	trace, err := s.store.GetDecisionTrace(ctx, req.RequestId, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetDecisionTraceResponse{
		Trace: trace,
	}, nil
}

func (s *Service) GetRuleImpact(ctx context.Context, req *pb.GetRuleImpactRequest) (*pb.GetRuleImpactResponse, error) {
	if req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}

	startDate, endDate, err := mergeWindowFromEnvelope(req.StartDate, req.EndDate, req.GetQuery())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "query.start_time and query.end_time must be RFC3339 or YYYY-MM-DD")
	}
	req.StartDate = startDate
	req.EndDate = endDate

	if req.StartDate == nil {
		cutoff := time.Now().AddDate(0, 0, -int(defaultRuleImpactDays))
		req.StartDate = timestamppb.New(cutoff)
	}

	if req.EndDate != nil && req.StartDate.AsTime().After(req.EndDate.AsTime()) {
		return nil, status.Error(codes.InvalidArgument, "start_date must be <= end_date")
	}

	resp, err := s.store.GetRuleImpact(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp != nil && resp.Meta == nil {
		resp.Meta = &pb.AnalyticsMeta{
			Truncated: false,
		}
	}
	return resp, nil
}

func (s *Service) GetKpis(ctx context.Context, req *pb.GetKpisRequest) (*pb.GetKpisResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}

	startTime, endTime, err := mergeWindowFromEnvelope(req.StartTime, req.EndTime, req.GetQuery())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "query.start_time and query.end_time must be RFC3339 or YYYY-MM-DD")
	}
	req.StartTime = startTime
	req.EndTime = endTime
	req.GroupBy = mergeGranularityFromEnvelope(req.GroupBy, req.GetQuery(), "")

	if req.GroupBy != "" && req.GroupBy != "day" && req.GroupBy != "hour" {
		return nil, status.Error(codes.InvalidArgument, "group_by must be 'day' or 'hour'")
	}

	if req.StartTime != nil && req.EndTime != nil {
		start := req.StartTime.AsTime()
		end := req.EndTime.AsTime()
		if start.After(end) {
			return nil, status.Error(codes.InvalidArgument, "start_time must be <= end_time")
		}
	}

	resp, err := s.store.GetKpis(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp != nil && resp.Meta == nil {
		resp.Meta = &pb.AnalyticsMeta{
			Truncated: false,
		}
	}
	return resp, nil
}

func (s *Service) GetVolumeSeries(ctx context.Context, req *pb.GetVolumeSeriesRequest) (*pb.GetVolumeSeriesResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}

	startTime, endTime, err := mergeWindowFromEnvelope(req.StartTime, req.EndTime, req.GetQuery())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "query.start_time and query.end_time must be RFC3339 or YYYY-MM-DD")
	}
	req.StartTime = startTime
	req.EndTime = endTime
	req.Granularity = mergeGranularityFromEnvelope(req.Granularity, req.GetQuery(), "")

	if req.Granularity != "day" && req.Granularity != "hour" {
		if req.Granularity == "" {
			req.Granularity = "day"
		} else {
			return nil, status.Error(codes.InvalidArgument, "granularity must be 'day' or 'hour'")
		}
	}

	if req.StartTime != nil && req.EndTime != nil {
		start := req.StartTime.AsTime()
		end := req.EndTime.AsTime()
		if start.After(end) {
			return nil, status.Error(codes.InvalidArgument, "start_time must be <= end_time")
		}
	}

	resp, err := s.store.GetVolumeSeries(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp != nil && resp.Meta == nil {
		resp.Meta = &pb.AnalyticsMeta{
			Truncated: false,
		}
	}
	return resp, nil
}

func (s *Service) GetConfusionMatrix(ctx context.Context, req *pb.GetConfusionMatrixRequest) (*pb.GetConfusionMatrixResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}

	startTime, endTime, err := mergeWindowFromEnvelope(req.StartTime, req.EndTime, req.GetQuery())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "query.start_time and query.end_time must be RFC3339 or YYYY-MM-DD")
	}
	req.StartTime = startTime
	req.EndTime = endTime

	if req.StartTime != nil && req.EndTime != nil && req.StartTime.AsTime().After(req.EndTime.AsTime()) {
		return nil, status.Error(codes.InvalidArgument, "start_time must be <= end_time")
	}

	resp, err := s.store.GetConfusionMatrix(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp != nil && resp.Meta == nil {
		resp.Meta = &pb.AnalyticsMeta{
			Truncated: false,
		}
	}
	return resp, nil
}

func (s *Service) StoreGeneratedData(ctx context.Context, req *pb.StoreGeneratedDataRequest) (*pb.StoreGeneratedDataResponse, error) {
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

// Helpers

func normalizeDays(value, fallback, max int32) (int32, error) {
	if value == 0 {
		value = fallback
	}
	if value < 1 || value > max {
		return 0, status.Errorf(codes.InvalidArgument, "days must be between 1 and %d", max)
	}
	return value, nil
}

func normalizeLimit(value, fallback, max int32) (int32, bool) {
	if value <= 0 {
		return fallback, false
	}
	if value > max {
		return max, true
	}
	return value, false
}

func normalizeOffset(value int32) (int32, error) {
	if value < 0 {
		return 0, status.Error(codes.InvalidArgument, "offset must be >= 0")
	}
	return value, nil
}

func errVersionNotFound(err error) error {
	if status.Code(err) == codes.NotFound {
		return status.Error(codes.NotFound, "rule or version not found")
	}
	return err
}
