package store

import (
	"context"
	"time"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/mock"
)

// MockStore is a mock implementation of the Store interface for testing.
type MockStore struct {
	mock.Mock
}

var _ Store = (*MockStore)(nil)

func (m *MockStore) GetDailyStats(ctx context.Context, cutoffDate time.Time) ([]*pb.DailyStat, error) {
	args := m.Called(ctx, cutoffDate)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.DailyStat), args.Error(1)
}

func (m *MockStore) GetTransactionDetails(ctx context.Context, cutoffDate time.Time, limit, offset int32) ([]*pb.TransactionDetail, error) {
	args := m.Called(ctx, cutoffDate, limit, offset)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.TransactionDetail), args.Error(1)
}

func (m *MockStore) SearchTransactions(ctx context.Context, req *pb.SearchTransactionsRequest, limit, offset int32) ([]*pb.TransactionDetail, int64, error) {
	args := m.Called(ctx, req, limit, offset)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*pb.TransactionDetail), args.Get(1).(int64), args.Error(2)
}

func (m *MockStore) GetRecentAlerts(ctx context.Context, limit, offset int32) ([]*pb.Alert, error) {
	args := m.Called(ctx, limit, offset)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.Alert), args.Error(1)
}

func (m *MockStore) GetOverviewMetrics(ctx context.Context) (*pb.GetOverviewMetricsResponse, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetOverviewMetricsResponse), args.Error(1)
}

func (m *MockStore) GetShadowComparison(ctx context.Context, hours int32) (*pb.ShadowModeMetrics, error) {
	args := m.Called(ctx, hours)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.ShadowModeMetrics), args.Error(1)
}

func (m *MockStore) GetTrainingData(ctx context.Context, cutoff time.Time) ([]*pb.TransactionDetail, []*pb.TransactionDetail, error) {
	args := m.Called(ctx, cutoff)
	// Handle nil returns safely
	var train, test []*pb.TransactionDetail
	if args.Get(0) != nil {
		train = args.Get(0).([]*pb.TransactionDetail)
	}
	if args.Get(1) != nil {
		test = args.Get(1).([]*pb.TransactionDetail)
	}
	return train, test, args.Error(2)
}

func (m *MockStore) ListRuleVersions(ctx context.Context, ruleID string, limit, offset int32) ([]*pb.Rule, int64, error) {
	args := m.Called(ctx, ruleID, limit, offset)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*pb.Rule), args.Get(1).(int64), args.Error(2)
}

func (m *MockStore) GetRuleVersion(ctx context.Context, ruleID, versionID string) (*pb.GetRuleVersionResponse, error) {
	args := m.Called(ctx, ruleID, versionID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetRuleVersionResponse), args.Error(1)
}

func (m *MockStore) PublishRuleVersion(ctx context.Context, req *pb.PublishRuleVersionRequest) (string, error) {
	args := m.Called(ctx, req)
	return args.String(0), args.Error(1)
}

func (m *MockStore) GetRuleReadiness(ctx context.Context, ruleID string) (*pb.GetRuleReadinessResponse, error) {
	args := m.Called(ctx, ruleID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetRuleReadinessResponse), args.Error(1)
}

func (m *MockStore) DiffRuleVersions(ctx context.Context, ruleID, vA, vB string) (*pb.DiffRuleVersionsResponse, error) {
	args := m.Called(ctx, ruleID, vA, vB)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.DiffRuleVersionsResponse), args.Error(1)
}

func (m *MockStore) SaveRule(ctx context.Context, r *pb.Rule) error {
	args := m.Called(ctx, r)
	return args.Error(0)
}

func (m *MockStore) GetRule(ctx context.Context, ruleID string) (*pb.Rule, error) {
	args := m.Called(ctx, ruleID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.Rule), args.Error(1)
}

func (m *MockStore) ListRules(ctx context.Context, statusFilter string, includeArchived bool) ([]*pb.Rule, error) {
	args := m.Called(ctx, statusFilter, includeArchived)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.Rule), args.Error(1)
}

func (m *MockStore) DeleteRule(ctx context.Context, ruleID string) error {
	args := m.Called(ctx, ruleID)
	return args.Error(0)
}

func (m *MockStore) GetBacktestFeatures(ctx context.Context, start, end time.Time) ([]*pb.BacktestFeatureVector, error) {
	args := m.Called(ctx, start, end)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.BacktestFeatureVector), args.Error(1)
}

func (m *MockStore) SaveBacktestResult(ctx context.Context, res *pb.BacktestResult) error {
	args := m.Called(ctx, res)
	return args.Error(0)
}

func (m *MockStore) ListBacktestResults(ctx context.Context, ruleID string, start, end *time.Time, limit, offset int32) ([]*pb.BacktestResult, error) {
	args := m.Called(ctx, ruleID, start, end, limit, offset)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.BacktestResult), args.Error(1)
}

func (m *MockStore) GetBacktestResult(ctx context.Context, jobID string) (*pb.BacktestResult, error) {
	args := m.Called(ctx, jobID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.BacktestResult), args.Error(1)
}

func (m *MockStore) StoreGeneratedData(ctx context.Context, records []*pb.GeneratedRecord, metadata []*pb.EvaluationMetadata) (int64, error) {
	args := m.Called(ctx, records, metadata)
	return args.Get(0).(int64), args.Error(1)
}

func (m *MockStore) ClearAllData(ctx context.Context) ([]string, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]string), args.Error(1)
}

func (m *MockStore) MaterializeFeatures(ctx context.Context) (int64, error) {
	args := m.Called(ctx)
	return args.Get(0).(int64), args.Error(1)
}

func (m *MockStore) LogInferenceEvent(ctx context.Context, event *pb.InferenceEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

func (m *MockStore) GetFeatureSample(ctx context.Context, sampleSize int32, stratify bool) ([]*pb.FeatureSample, error) {
	args := m.Called(ctx, sampleSize, stratify)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.FeatureSample), args.Error(1)
}

func (m *MockStore) GetDriftWindow(ctx context.Context, cutoff time.Time) ([]*pb.TransactionDetail, error) {
	args := m.Called(ctx, cutoff)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.TransactionDetail), args.Error(1)
}

func (m *MockStore) GetInferenceScores(ctx context.Context, cutoff time.Time) ([]int32, error) {
	args := m.Called(ctx, cutoff)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]int32), args.Error(1)
}

func (m *MockStore) GetGenerationJob(ctx context.Context, key string) (*pb.GenerateDataResponse, string, error) {
	args := m.Called(ctx, key)
	if args.Get(0) == nil {
		return nil, "", args.Error(2)
	}
	return args.Get(0).(*pb.GenerateDataResponse), args.String(1), args.Error(2)
}

func (m *MockStore) CreateGenerationJob(ctx context.Context, key string) error {
	args := m.Called(ctx, key)
	return args.Error(0)
}

func (m *MockStore) CompleteGenerationJob(ctx context.Context, key string, resp *pb.GenerateDataResponse) error {
	args := m.Called(ctx, key, resp)
	return args.Error(0)
}

func (m *MockStore) FailGenerationJob(ctx context.Context, key string, errMsg string) error {
	args := m.Called(ctx, key, errMsg)
	return args.Error(0)
}

func (m *MockStore) GetDatasetProfile(ctx context.Context, datasetID string, limitFeatures, numBuckets int32) (*pb.GetDatasetProfileResponse, error) {
	args := m.Called(ctx, datasetID, limitFeatures, numBuckets)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetDatasetProfileResponse), args.Error(1)
}
func (m *MockStore) GetRuleStats(ctx context.Context, ruleID string, cutoff time.Time) ([]*pb.RuleStats, error) {
	args := m.Called(ctx, ruleID, cutoff)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.RuleStats), args.Error(1)
}

func (m *MockStore) GetAttribution(ctx context.Context, cutoff time.Time, limit int32) ([]*pb.DailyAttribution, error) {
	args := m.Called(ctx, cutoff, limit)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.DailyAttribution), args.Error(1)
}

func (m *MockStore) GetLatestUserFeatures(ctx context.Context, userID string) (*pb.UserFeatures, bool, error) {
	args := m.Called(ctx, userID)
	if args.Get(0) == nil {
		return nil, args.Bool(1), args.Error(2)
	}
	return args.Get(0).(*pb.UserFeatures), args.Bool(1), args.Error(2)
}

func (m *MockStore) BatchGetLatestUserFeatures(ctx context.Context, userIDs []string) (map[string]*pb.UserFeatures, error) {
	args := m.Called(ctx, userIDs)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(map[string]*pb.UserFeatures), args.Error(1)
}

func (m *MockStore) ListDecisions(ctx context.Context, req *pb.ListDecisionsRequest) ([]*pb.DecisionSummary, int64, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*pb.DecisionSummary), args.Get(1).(int64), args.Error(2)
}

func (m *MockStore) GetDecision(ctx context.Context, requestID string) (*pb.InferenceEvent, error) {
	args := m.Called(ctx, requestID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.InferenceEvent), args.Error(1)
}

func (m *MockStore) GetDecisionTrace(ctx context.Context, requestID string) ([]*pb.RuleImpact, error) {
	args := m.Called(ctx, requestID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.RuleImpact), args.Error(1)
}

func (m *MockStore) GetRuleImpact(ctx context.Context, req *pb.GetRuleImpactRequest) (*pb.GetRuleImpactResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetRuleImpactResponse), args.Error(1)
}

func (m *MockStore) GetKpis(ctx context.Context, req *pb.GetKpisRequest) (*pb.GetKpisResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetKpisResponse), args.Error(1)
}

func (m *MockStore) GetVolumeSeries(ctx context.Context, req *pb.GetVolumeSeriesRequest) (*pb.GetVolumeSeriesResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetVolumeSeriesResponse), args.Error(1)
}

func (m *MockStore) GetConfusionMatrix(ctx context.Context, req *pb.GetConfusionMatrixRequest) (*pb.GetConfusionMatrixResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetConfusionMatrixResponse), args.Error(1)
}

func (m *MockStore) ListJobs(ctx context.Context, req *pb.ListJobsRequest) ([]*pb.Job, int64, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*pb.Job), args.Get(1).(int64), args.Error(2)
}

func (m *MockStore) GetJob(ctx context.Context, jobID string) (*pb.Job, error) {
	args := m.Called(ctx, jobID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.Job), args.Error(1)
}

func (m *MockStore) GetJobEvents(ctx context.Context, jobID string, limit, offset int32) ([]*pb.JobEvent, error) {
	args := m.Called(ctx, jobID, limit, offset)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.JobEvent), args.Error(1)
}

func (m *MockStore) SaveDatasetProfile(ctx context.Context, profile *pb.DatasetProfile) error {
	args := m.Called(ctx, profile)
	return args.Error(0)
}

func (m *MockStore) GetDatasetProfileCached(ctx context.Context, profileID string) (*pb.DatasetProfile, error) {
	args := m.Called(ctx, profileID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.DatasetProfile), args.Error(1)
}

func (m *MockStore) ListDatasetProfiles(ctx context.Context, req *pb.ListDatasetProfilesRequest) ([]*pb.DatasetProfile, int64, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*pb.DatasetProfile), args.Get(1).(int64), args.Error(2)
}

func (m *MockStore) GetLatestDatasetProfile(ctx context.Context) (*pb.GetLatestDatasetProfileResponse, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetLatestDatasetProfileResponse), args.Error(1)
}

func (m *MockStore) GetJobSummary(ctx context.Context, req *pb.GetJobSummaryRequest) ([]*pb.JobSummaryBucket, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.JobSummaryBucket), args.Error(1)
}

func (m *MockStore) SaveTrainingRun(ctx context.Context, run *pb.TrainingRun) error {
	args := m.Called(ctx, run)
	return args.Error(0)
}

func (m *MockStore) ListTrainingRuns(ctx context.Context, req *pb.ListTrainingRunsRequest) ([]*pb.TrainingRun, int64, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*pb.TrainingRun), args.Get(1).(int64), args.Error(2)
}

func (m *MockStore) ListModelVersions(ctx context.Context, req *pb.ListModelVersionsRequest) ([]*pb.TrainingRun, int64, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*pb.TrainingRun), args.Get(1).(int64), args.Error(2)
}

func (m *MockStore) GetTrainingRun(ctx context.Context, runID string) (*pb.TrainingRun, error) {
	args := m.Called(ctx, runID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.TrainingRun), args.Error(1)
}

func (m *MockStore) GetMetricSeries(ctx context.Context, modelName, metricName string, start, end time.Time) ([]*pb.MetricPoint, error) {
	args := m.Called(ctx, modelName, metricName, start, end)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*pb.MetricPoint), args.Error(1)
}
