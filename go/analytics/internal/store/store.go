package store

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"
	"strings"
	"time"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Store defines the data access methods for the analytics service.
type Store interface {
	// Analytics
	GetDailyStats(ctx context.Context, cutoffDate time.Time) ([]*pb.DailyStat, error)
	GetTransactionDetails(ctx context.Context, cutoffDate time.Time, limit, offset int32) ([]*pb.TransactionDetail, error)
	SearchTransactions(ctx context.Context, req *pb.SearchTransactionsRequest, limit, offset int32) ([]*pb.TransactionDetail, int64, error)
	GetRecentAlerts(ctx context.Context, limit, offset int32) ([]*pb.Alert, error)
	GetOverviewMetrics(ctx context.Context) (*pb.GetOverviewMetricsResponse, error)

	GetDatasetFingerprint(ctx context.Context) (*pb.GetDatasetFingerprintResponse, error)
	GetSchemaSummary(ctx context.Context) (*pb.GetSchemaSummaryResponse, error)

	// Analytics
	GetShadowComparison(ctx context.Context, hours int32) (*pb.ShadowModeMetrics, error)
	GetRuleStats(ctx context.Context, ruleID string, cutoff time.Time) ([]*pb.RuleStats, error)
	GetAttribution(ctx context.Context, cutoff time.Time, limit int32) ([]*pb.DailyAttribution, error)

	// Feature Hydration
	GetLatestUserFeatures(ctx context.Context, userID string) (*pb.UserFeatures, bool, error)
	BatchGetLatestUserFeatures(ctx context.Context, userIDs []string) (map[string]*pb.UserFeatures, error)

	// Training
	GetTrainingData(ctx context.Context, cutoff time.Time) (train []*pb.TransactionDetail, test []*pb.TransactionDetail, err error)

	// Rules
	ListRuleVersions(ctx context.Context, ruleID string, limit, offset int32) ([]*pb.Rule, int64, error)
	GetRuleVersion(ctx context.Context, ruleID, versionID string) (*pb.GetRuleVersionResponse, error)
	PublishRuleVersion(ctx context.Context, req *pb.PublishRuleVersionRequest) (string, error)
	GetRuleReadiness(ctx context.Context, ruleID string) (*pb.GetRuleReadinessResponse, error)
	DiffRuleVersions(ctx context.Context, ruleID, vA, vB string) (*pb.DiffRuleVersionsResponse, error)
	SaveRule(ctx context.Context, r *pb.Rule) error
	GetRule(ctx context.Context, ruleID string) (*pb.Rule, error)
	ListRules(ctx context.Context, statusFilter string, includeArchived bool) ([]*pb.Rule, error)
	DeleteRule(ctx context.Context, ruleID string) error

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
	GetFeatureSample(ctx context.Context, sampleSize int32, stratify bool) ([]*pb.FeatureSample, error)
	GetDriftWindow(ctx context.Context, cutoff time.Time) ([]*pb.TransactionDetail, error)
	GetInferenceScores(ctx context.Context, cutoff time.Time) ([]int32, error)

	// Idempotency
	GetGenerationJob(ctx context.Context, key string) (*pb.GenerateDataResponse, string, error)
	CreateGenerationJob(ctx context.Context, key string) error
	CompleteGenerationJob(ctx context.Context, key string, resp *pb.GenerateDataResponse) error
	FailGenerationJob(ctx context.Context, key string, errMsg string) error

	// Profiling
	GetDatasetProfile(ctx context.Context, datasetID string, limitFeatures, numBuckets int32) (*pb.GetDatasetProfileResponse, error)
}

// SQLStore implements Store using a PostgreSQL database.
type SQLStore struct {
	db *sql.DB
}

const defaultQueryTimeout = 30 * time.Second

// NewSQLStore creates a new SQLStore.
func NewSQLStore(db *sql.DB) *SQLStore {
	return &SQLStore{db: db}
}

func (s *SQLStore) getEstimatedCount(ctx context.Context, tableName string) (int64, error) {
	var estimate int64
	err := s.db.QueryRowContext(ctx, "SELECT reltuples::bigint FROM pg_class WHERE relname = $1", tableName).Scan(&estimate)
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

func normalizeLimit(reqLimit, defaultLimit, maxLimit int32, fieldName string) (int32, error) {
	if reqLimit < 0 {
		return 0, status.Errorf(codes.InvalidArgument, "%s must be non-negative", fieldName)
	}
	if reqLimit == 0 {
		return defaultLimit, nil
	}
	if reqLimit > maxLimit {
		return maxLimit, nil
	}
	return reqLimit, nil
}

func normalizeOffset(offset int32) (int32, error) {
	if offset < 0 {
		return 0, status.Error(codes.InvalidArgument, "offset must be non-negative")
	}
	return offset, nil
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

func normalizeDays(days, defaultDays, maxDays int32) (int32, error) {
	if days < 0 {
		return 0, status.Error(codes.InvalidArgument, "days must be non-negative")
	}
	if days == 0 {
		return defaultDays, nil
	}
	if days > maxDays {
		return maxDays, nil // Cap at max limit
	}
	return days, nil
}
