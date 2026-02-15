package store

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"
	"strings"
	"time"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
)

// Store defines the data access methods for the analytics service.
type Store interface {
	// Analytics
	GetDailyStats(ctx context.Context, cutoffDate time.Time, tenantID string) ([]*pb.DailyStat, error)
	GetTransactionDetails(ctx context.Context, cutoffDate time.Time, limit, offset int32, tenantID string) ([]*pb.TransactionDetail, error)
	SearchTransactions(ctx context.Context, req *pb.SearchTransactionsRequest) ([]*pb.TransactionDetail, int64, error)
	GetRecentAlerts(ctx context.Context, limit, offset int32, tenantID string) ([]*pb.Alert, error)
	GetOverviewMetrics(ctx context.Context, tenantID string) (*pb.GetOverviewMetricsResponse, error)

	// Analytics
	GetShadowComparison(ctx context.Context, hours int32, tenantID string) (*pb.ShadowModeMetrics, error)
	GetRuleStats(ctx context.Context, ruleID string, cutoff time.Time, tenantID string) ([]*pb.RuleStats, error)
	GetAttribution(ctx context.Context, cutoff time.Time, limit int32, tenantID string) ([]*pb.DailyAttribution, error)

	// Decisions (Phase 2)
	ListDecisions(ctx context.Context, req *pb.ListDecisionsRequest) ([]*pb.DecisionSummary, int64, string, error)
	GetDecision(ctx context.Context, requestID string, tenantID string) (*pb.InferenceEvent, error)
	GetDecisionTrace(ctx context.Context, requestID string, tenantID string) ([]*pb.RuleImpact, error)
	GetRuleImpact(ctx context.Context, req *pb.GetRuleImpactRequest) (*pb.GetRuleImpactResponse, error)

	// Dashboard Aggregates (Phase 3)
	GetKpis(ctx context.Context, req *pb.GetKpisRequest) (*pb.GetKpisResponse, error)
	GetVolumeSeries(ctx context.Context, req *pb.GetVolumeSeriesRequest) (*pb.GetVolumeSeriesResponse, error)
	GetConfusionMatrix(ctx context.Context, req *pb.GetConfusionMatrixRequest) (*pb.GetConfusionMatrixResponse, error)

	// Jobs (Phase A1)
	ListJobs(ctx context.Context, req *pb.ListJobsRequest) ([]*pb.Job, int64, string, error)
	GetJob(ctx context.Context, jobID string, tenantID string) (*pb.Job, error)
	GetJobEvents(ctx context.Context, req *pb.GetJobEventsRequest) ([]*pb.JobEvent, error)
	GetJobSummary(ctx context.Context, req *pb.GetJobSummaryRequest) ([]*pb.JobSummaryBucket, error)
	CancelJob(ctx context.Context, jobID string, tenantID string) error
	RetryJob(ctx context.Context, jobID string, tenantID string) (string, error)

	// Dataset Profiles (Phase A2)
	SaveDatasetProfile(ctx context.Context, profile *pb.DatasetProfile) error
	GetDatasetProfileCached(ctx context.Context, profileID string, tenantID string) (*pb.DatasetProfile, error)
	ListDatasetProfiles(ctx context.Context, req *pb.ListDatasetProfilesRequest) ([]*pb.DatasetProfile, int64, string, error)
	GetLatestDatasetProfile(ctx context.Context, tenantID string) (*pb.GetLatestDatasetProfileResponse, error)

	// Training Runs (Phase B)
	SaveTrainingRun(ctx context.Context, run *pb.TrainingRun) error
	ListTrainingRuns(ctx context.Context, req *pb.ListTrainingRunsRequest) ([]*pb.TrainingRun, int64, string, error)
	GetTrainingRun(ctx context.Context, runID string, tenantID string) (*pb.TrainingRun, error)
	ListModelVersions(ctx context.Context, req *pb.ListModelVersionsRequest) ([]*pb.TrainingRun, int64, string, error)
	GetMetricSeries(ctx context.Context, req *pb.GetMetricSeriesRequest) ([]*pb.MetricPoint, error)

	// Feature Hydration
	GetLatestUserFeatures(ctx context.Context, userID string, tenantID string) (*pb.UserFeatures, bool, error)
	BatchGetLatestUserFeatures(ctx context.Context, userIDs []string, tenantID string) (map[string]*pb.UserFeatures, error)

	// Training
	GetTrainingData(ctx context.Context, cutoff time.Time, tenantID string) (train []*pb.TransactionDetail, test []*pb.TransactionDetail, err error)

	// Rules
	ListRuleVersions(ctx context.Context, ruleID string, limit, offset int32, tenantID string) ([]*pb.Rule, int64, error)
	GetRuleVersion(ctx context.Context, ruleID, versionID string, tenantID string) (*pb.GetRuleVersionResponse, error)
	PublishRuleVersion(ctx context.Context, req *pb.PublishRuleVersionRequest) (string, error)
	GetRuleReadiness(ctx context.Context, ruleID string, tenantID string) (*pb.GetRuleReadinessResponse, error)
	DiffRuleVersions(ctx context.Context, ruleID, vA, vB string, tenantID string) (*pb.DiffRuleVersionsResponse, error)
	SaveRule(ctx context.Context, r *pb.Rule, tenantID string) error
	GetRule(ctx context.Context, ruleID string, tenantID string) (*pb.Rule, error)
	ListRules(ctx context.Context, statusFilter string, includeArchived bool, tenantID string) ([]*pb.Rule, error)
	DeleteRule(ctx context.Context, ruleID string, tenantID string) error

	// Backtest
	GetBacktestFeatures(ctx context.Context, start, end time.Time) ([]*pb.BacktestFeatureVector, error)
	SaveBacktestResult(ctx context.Context, res *pb.BacktestResult) error
	ListBacktestResults(ctx context.Context, ruleID string, start, end *time.Time, limit, offset int32) ([]*pb.BacktestResult, error)
	GetBacktestResult(ctx context.Context, jobID string) (*pb.BacktestResult, error)

	// Data Management
	StoreGeneratedData(ctx context.Context, records []*pb.GeneratedRecord, metadata []*pb.EvaluationMetadata) (int64, error)
	ClearAllData(ctx context.Context) ([]string, error)
	MaterializeFeatures(ctx context.Context) (int64, error)
	LogInferenceEvent(ctx context.Context, event *pb.InferenceEvent) error
	GetFeatureSample(ctx context.Context, sampleSize int32, stratify bool, tenantID string) ([]*pb.FeatureSample, error)
	GetDriftWindow(ctx context.Context, cutoff time.Time, tenantID string) ([]*pb.TransactionDetail, error)
	GetInferenceScores(ctx context.Context, cutoff time.Time, tenantID string) ([]int32, error)

	// Idempotency
	GetGenerationJob(ctx context.Context, key string) (*pb.GenerateDataResponse, string, error)
	CreateGenerationJob(ctx context.Context, key string) error
	CompleteGenerationJob(ctx context.Context, key string, resp *pb.GenerateDataResponse) error
	FailGenerationJob(ctx context.Context, key string, errMsg string) error

	// Profiling
	GetDatasetProfile(ctx context.Context, datasetID string, limitFeatures, numBuckets int32, tenantID string) (*pb.GetDatasetProfileResponse, error)
}

// SQLStore implements Store using a PostgreSQL database.
type SQLStore struct {
	db *sql.DB
}

const defaultQueryTimeout = 30 * time.Second

const (
	DecisionApprove = "APPROVE"
	DecisionReview  = "REVIEW"
	DecisionReject  = "REJECT"
)

// NewSQLStore creates a new SQLStore.
func NewSQLStore(db *sql.DB) *SQLStore {
	return &SQLStore{db: db}
}

func (s *SQLStore) getEstimatedCount(ctx context.Context, tableName string) (int64, error) {
	var estimate int64
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, "SELECT reltuples::bigint FROM pg_class WHERE relname = $1", tableName).Scan(&estimate)
	if err != nil {
		return 0, err
	}
	return estimate, nil
}

// Helper functions that might be shared

func parseISODate(dateStr string) (time.Time, bool) {
	// Try parsing with time.RFC3339
	t, err := time.Parse(time.RFC3339, dateStr)
	if err == nil {
		return t, true
	}
	// Try parsing YYYY-MM-DD
	t, err = time.Parse("2006-01-02", dateStr)
	if err == nil {
		return t, true
	}
	return time.Time{}, false
}

func getPostgresVersion(ctx context.Context, db *sql.DB) (int, error) {
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()
	var versionStr string
	err := db.QueryRowContext(queryCtx, "SELECT version()").Scan(&versionStr)
	if err != nil {
		return 0, err
	}

	parts := strings.Split(versionStr, " ")
	for i, part := range parts {
		if part == "PostgreSQL" && i+1 < len(parts) {
			versionParts := strings.Split(parts[i+1], ".")
			if len(versionParts) > 0 {
				major, err := strconv.Atoi(versionParts[0])
				if err == nil {
					return major, nil
				}
			}
		}
	}
	return 0, fmt.Errorf("could not parse postgres version: %s", versionStr)
}

type tableStats struct {
	minID      int64
	maxID      int64
	totalCount int64
}

func getTableStats(ctx context.Context, db *sql.DB, table string) (tableStats, error) {
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()
	var stats tableStats
	query := fmt.Sprintf("SELECT COALESCE(MIN(id), 0), COALESCE(MAX(id), 0), COUNT(*) FROM %s", table)
	err := db.QueryRowContext(queryCtx, query).Scan(&stats.minID, &stats.maxID, &stats.totalCount)
	if err != nil {
		return stats, err
	}
	return stats, nil
}
