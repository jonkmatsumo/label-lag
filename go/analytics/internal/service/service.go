package service

import (
	"context"
	"os"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/generator"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/store"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type Service struct {
	pb.UnimplementedAnalyticsServiceServer
	store store.Store
}

func NewService(store store.Store) *Service {
	return &Service{
		store: store,
	}
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
)

func (s *Service) GetDailyStats(ctx context.Context, req *pb.GetDailyStatsRequest) (*pb.GetDailyStatsResponse, error) {
	days, err := normalizeDays(req.Days, defaultDailyStatsDay, 365)
	if err != nil {
		return nil, err
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

func (s *Service) GetTransactionDetails(ctx context.Context, req *pb.GetTransactionDetailsRequest) (*pb.GetTransactionDetailsResponse, error) {
	days, err := normalizeDays(req.Days, defaultTxnDays, 365)
	if err != nil {
		return nil, err
	}
	limit, err := normalizeLimit(req.Limit, defaultTxnLimit, maxTransactionLimit, "limit")
	if err != nil {
		return nil, err
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

func (s *Service) SearchTransactions(ctx context.Context, req *pb.SearchTransactionsRequest) (*pb.SearchTransactionsResponse, error) {
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

func (s *Service) GetRecentAlerts(ctx context.Context, req *pb.GetRecentAlertsRequest) (*pb.GetRecentAlertsResponse, error) {
	limit, err := normalizeLimit(req.Limit, defaultAlertLimit, maxAlertLimit, "limit")
	if err != nil {
		return nil, err
	}

	alerts, err := s.store.GetRecentAlerts(ctx, limit)
	if err != nil {
		return nil, err
	}

	return &pb.GetRecentAlertsResponse{
		Alerts: alerts,
	}, nil
}

func (s *Service) GetOverviewMetrics(ctx context.Context, req *pb.GetOverviewMetricsRequest) (*pb.GetOverviewMetricsResponse, error) {
	return s.store.GetOverviewMetrics(ctx)
}

func (s *Service) GetDatasetFingerprint(ctx context.Context, req *pb.GetDatasetFingerprintRequest) (*pb.GetDatasetFingerprintResponse, error) {
	return s.store.GetDatasetFingerprint(ctx)
}

func (s *Service) GetSchemaSummary(ctx context.Context, req *pb.GetSchemaSummaryRequest) (*pb.GetSchemaSummaryResponse, error) {
	return s.store.GetSchemaSummary(ctx)
}

func (s *Service) GetTrainingData(ctx context.Context, req *pb.GetTrainingDataRequest) (*pb.GetTrainingDataResponse, error) {
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

	results, err := s.store.ListBacktestResults(ctx, req.RuleId, start, end)
	if err != nil {
		return nil, err
	}

	return &pb.ListBacktestResultsResponse{Results: results}, nil
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

	metrics, err := s.store.GetShadowComparison(ctx, hours)
	if err != nil {
		return nil, err
	}

	return &pb.GetShadowComparisonResponse{Metrics: metrics}, nil
}

func (s *Service) GenerateData(ctx context.Context, req *pb.GenerateDataRequest) (*pb.GenerateDataResponse, error) {
	if os.Getenv("ENABLE_GO_DATASET_GENERATE") != "true" {
		return nil, status.Error(codes.Unimplemented, "dataset generation is disabled")
	}

	seed := time.Now().UnixNano()
	gen := generator.NewGenerator(&seed)
	result := gen.GenerateDatasetWithSequences(int(req.NumUsers), req.FraudRate)

	count, err := s.store.StoreGeneratedData(ctx, result.Records, result.Metadata)
	if err != nil {
		return nil, err
	}

	return &pb.GenerateDataResponse{
		Success:      true,
		TotalRecords: count,
	}, nil
}

func (s *Service) ClearAllData(ctx context.Context, req *pb.ClearAllDataRequest) (*pb.ClearAllDataResponse, error) {
	tables, err := s.store.ClearAllData(ctx)
	if err != nil {
		return nil, err
	}
	return &pb.ClearAllDataResponse{Success: true, TablesCleared: tables}, nil
}

func (s *Service) MaterializeFeatures(ctx context.Context, req *pb.MaterializeFeaturesRequest) (*pb.MaterializeFeaturesResponse, error) {
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

	if err := s.store.SaveRule(ctx, req.Rule); err != nil {
		return nil, err
	}

	return &pb.SaveRuleResponse{Success: true}, nil
}

func (s *Service) GetRule(ctx context.Context, req *pb.GetRuleRequest) (*pb.GetRuleResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}
	rule, err := s.store.GetRule(ctx, req.RuleId)
	if err != nil {
		return nil, err
	}
	return &pb.GetRuleResponse{Rule: rule}, nil
}

func (s *Service) ListRules(ctx context.Context, req *pb.ListRulesRequest) (*pb.ListRulesResponse, error) {
	rules, err := s.store.ListRules(ctx, req.Status, req.IncludeArchived)
	if err != nil {
		return nil, err
	}
	return &pb.ListRulesResponse{Rules: rules}, nil
}

func (s *Service) DeleteRule(ctx context.Context, req *pb.DeleteRuleRequest) (*pb.DeleteRuleResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}
	err := s.store.DeleteRule(ctx, req.RuleId)
	if err != nil {
		return nil, err
	}
	return &pb.DeleteRuleResponse{Success: true}, nil
}

func (s *Service) LogInferenceEvent(ctx context.Context, req *pb.LogInferenceEventRequest) (*pb.LogInferenceEventResponse, error) {
	if req == nil || req.Event == nil {
		return nil, status.Error(codes.InvalidArgument, "event required")
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

	samples, err := s.store.GetFeatureSample(ctx, sampleSize, req.Stratify)
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

	txs, err := s.store.GetDriftWindow(ctx, cutoff)
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

	scores, err := s.store.GetInferenceScores(ctx, cutoff)
	if err != nil {
		return nil, err
	}

	return &pb.GetInferenceScoresResponse{Scores: scores}, nil
}

func (s *Service) ListRuleVersions(ctx context.Context, req *pb.ListRuleVersionsRequest) (*pb.ListRuleVersionsResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}

	limit, err := normalizeLimit(req.Limit, 100, 1000, "limit")
	if err != nil {
		return nil, err
	}
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}

	versions, total, err := s.store.ListRuleVersions(ctx, req.RuleId, limit, offset)
	if err != nil {
		return nil, errVersionNotFound(err)
	}

	return &pb.ListRuleVersionsResponse{
		Versions: versions,
		Total:    total,
	}, nil
}

func (s *Service) GetRuleVersion(ctx context.Context, req *pb.GetRuleVersionRequest) (*pb.GetRuleVersionResponse, error) {
	if req == nil || req.RuleId == "" || req.VersionId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id and version_id required")
	}
	return s.store.GetRuleVersion(ctx, req.RuleId, req.VersionId)
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
	return s.store.DiffRuleVersions(ctx, req.RuleId, req.VersionA, req.VersionB)
}

func (s *Service) GetRuleReadiness(ctx context.Context, req *pb.GetRuleReadinessRequest) (*pb.GetRuleReadinessResponse, error) {
	if req == nil || req.RuleId == "" {
		return nil, status.Error(codes.InvalidArgument, "rule_id required")
	}
	return s.store.GetRuleReadiness(ctx, req.RuleId)
}

func (s *Service) GetRuleStats(ctx context.Context, req *pb.GetRuleStatsRequest) (*pb.GetRuleStatsResponse, error) {
	// Stub implementation from main.go
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

func (s *Service) GetAttribution(ctx context.Context, req *pb.GetAttributionRequest) (*pb.GetAttributionResponse, error) {
	// Stub implementation from main.go
	return &pb.GetAttributionResponse{
		Items: []*pb.DailyAttribution{},
	}, nil
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

func normalizeLimit(value, fallback, max int32, field string) (int32, error) {
	if value == 0 {
		return fallback, nil
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

func errVersionNotFound(err error) error {
	if status.Code(err) == codes.NotFound {
		return status.Error(codes.NotFound, "rule or version not found")
	}
	return err
}
