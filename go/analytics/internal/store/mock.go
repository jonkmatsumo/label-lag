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

func (m *MockStore) GetTransactionDetails(ctx context.Context, cutoffDate time.Time, limit int32) ([]*pb.TransactionDetail, error) {
	args := m.Called(ctx, cutoffDate, limit)
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

func (m *MockStore) GetRecentAlerts(ctx context.Context, limit int32) ([]*pb.Alert, error) {
	args := m.Called(ctx, limit)
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

func (m *MockStore) GetDatasetFingerprint(ctx context.Context) (*pb.GetDatasetFingerprintResponse, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetDatasetFingerprintResponse), args.Error(1)
}

func (m *MockStore) GetSchemaSummary(ctx context.Context) (*pb.GetSchemaSummaryResponse, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetSchemaSummaryResponse), args.Error(1)
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

func (m *MockStore) ListBacktestResults(ctx context.Context, ruleID string, start, end *time.Time) ([]*pb.BacktestResult, error) {
	args := m.Called(ctx, ruleID, start, end)
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

func (m *MockStore) GetDatasetProfile(ctx context.Context, datasetID string, limitFeatures, numBuckets int32) (*pb.GetDatasetProfileResponse, error) {
	args := m.Called(ctx, datasetID, limitFeatures, numBuckets)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*pb.GetDatasetProfileResponse), args.Error(1)
}
