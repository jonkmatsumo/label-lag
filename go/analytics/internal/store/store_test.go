package store

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/lib/pq"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestGetDailyStats(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	rows := sqlmock.NewRows([]string{"date", "total_transactions", "fraud_count", "fraud_rate", "total_amount", "avg_z_score"}).
		AddRow(time.Now(), 100, 5, 5.0, 1000.0, 0.5)

	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	stats, err := s.GetDailyStats(context.Background(), time.Now().AddDate(0, 0, -30), "")
	require.NoError(t, err)
	require.NotEmpty(t, stats)
	assert.Equal(t, int64(100), stats[0].TotalTransactions)
	assert.Equal(t, int64(5), stats[0].FraudCount)
}

func TestGetOverviewMetrics(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	rows := sqlmock.NewRows([]string{
		"total_records", "fraud_records", "unique_users",
		"min_txn_ts", "max_txn_ts",
		"min_created", "max_created",
		"total_amount", "fraud_amount",
	}).AddRow(1000, 50, 100, time.Now(), time.Now(), time.Now(), time.Now(), 50000.0, 2500.0)

	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	resp, err := s.GetOverviewMetrics(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, int64(1000), resp.TotalRecords)
	assert.Equal(t, int64(50), resp.FraudRecords)
	assert.Equal(t, float64(5.0), resp.FraudRate)
}

func TestGetFeatureSample_Stratified(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	// 1. Version query
	mock.ExpectQuery("SELECT version").WillReturnRows(sqlmock.NewRows([]string{"version"}).AddRow("PostgreSQL 16.1"))

	// 2. Stats query
	mock.ExpectQuery("SELECT COALESCE\\(MIN\\(gr.id\\), 0\\), COALESCE\\(MAX\\(gr.id\\), 0\\), COUNT\\(\\*\\)\\s+FROM generated_records gr\\s+INNER JOIN inference_events ie ON gr.record_id = ie.request_id\\s+WHERE ie.tenant_id = \\$1").
		WithArgs("tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"min", "max", "count"}).AddRow(1, 1000, 1000))

	// 3. Fraud rate query
	mock.ExpectQuery("SELECT CAST").WithArgs("tenant-1").WillReturnRows(sqlmock.NewRows([]string{"rate"}).AddRow(0.05))

	// 4. Sampling queries
	mock.ExpectQuery("(?s)SELECT.*is_fraudulent\\s*=\\s*true.*").
		WithArgs("tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"record_id", "is_fraudulent", "velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score"}).
			AddRow("f1", true, 1.0, 1.0, 1.0))

	mock.ExpectQuery("(?s)SELECT.*is_fraudulent\\s*=\\s*false.*").
		WithArgs("tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"record_id", "is_fraudulent", "velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score"}).
			AddRow("nf1", false, 0.0, 0.0, 0.0))

	samples, err := s.GetFeatureSample(context.Background(), 20, true, "tenant-1")
	// Wait, GetFeatureSample in store interface: GetFeatureSample(ctx context.Context, sampleSize int32) ([]*pb.FeatureSample, error)
	// It doesn't take 'stratify' param?
	// Let's check store interface.
	require.NoError(t, err)
	require.Len(t, samples, 2)
}

func TestSearchTransactions(t *testing.T) {
	dbMock, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer dbMock.Close()

	s := NewSQLStore(dbMock)

	// We expect the query to limit to 26 (Limit: 25 + 1)
	mock.ExpectQuery(`(?s)SELECT em.record_id, em.user_id,.*WHERE em.user_id = \$1 AND gr.amount >= \$2 AND ie.tenant_id = \$3.*ORDER BY em.created_at DESC, em.record_id DESC LIMIT 26`).
		WithArgs("user-1", 12.5, "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"record_id",
			"user_id",
			"created_at",
			"is_train_eligible",
			"is_pre_fraud",
			"amount",
			"is_fraudulent",
			"fraud_type",
			"is_off_hours_txn",
			"merchant_risk_score",
			"velocity_24h",
			"amount_to_avg_ratio_30d",
			"balance_volatility_z_score",
			"numerical_features",
			"categorical_features",
		}).AddRow("rec-1", "user-1", time.Now(), true, true, 22.5, false, "none", false, 10, 1, 1.0, -0.3, []byte("{}"), []byte("{}")))

	minAmount := 12.5
	req := &pb.SearchTransactionsRequest{
		UserId:    "user-1",
		MinAmount: &minAmount,
		Limit:     25,
		TenantId:  "tenant-1",
	}

	details, nextCursor, meta, err := s.SearchTransactions(context.Background(), req)
	require.NoError(t, err)
	assert.Len(t, details, 1)
	assert.Equal(t, "rec-1", details[0].RecordId)
	assert.Empty(t, nextCursor)
	assert.NotNil(t, meta)
	assert.False(t, meta.Truncated)
}

func TestSearchTransactions_Truncated(t *testing.T) {
	dbMock, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer dbMock.Close()

	s := NewSQLStore(dbMock)

	// We expect the query to clamp to 500, but fetch 501
	mock.ExpectQuery(`(?s)SELECT em.record_id, em.user_id,.*WHERE em.user_id = \$1.*ORDER BY em.created_at DESC, em.record_id DESC LIMIT 501`).
		WithArgs("user-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"record_id", "user_id", "created_at", "is_train_eligible", "is_pre_fraud",
			"amount", "is_fraudulent", "fraud_type", "is_off_hours_txn", "merchant_risk_score",
			"velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score", "numerical_features", "categorical_features",
		}).AddRow("rec-1", "user-1", time.Now(), true, true, 22.5, false, "none", false, 10, 1, 1.0, -0.3, []byte("{}"), []byte("{}")))

	req := &pb.SearchTransactionsRequest{
		UserId: "user-1",
		Limit:  1000, // Should be clamped to 500, and truncated=true
	}

	details, nextCursor, meta, err := s.SearchTransactions(context.Background(), req)
	require.NoError(t, err)
	assert.Len(t, details, 1)   // Only returned 1 row
	assert.Empty(t, nextCursor) // Did not exceed our clamped limit
	assert.NotNil(t, meta)
	assert.True(t, meta.Truncated) // But truncated was set to true because requested > 500
	assert.Equal(t, int32(500), meta.EffectiveLimit)
}

func TestSearchTransactions_InvalidCursorReturnsInvalidArgument(t *testing.T) {
	dbMock, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer dbMock.Close()

	s := NewSQLStore(dbMock)

	_, _, _, err = s.SearchTransactions(context.Background(), &pb.SearchTransactionsRequest{
		Cursor: "not-a-valid-cursor",
		Limit:  25,
	})
	require.Error(t, err)
	st, ok := status.FromError(err)
	require.True(t, ok)
	assert.Equal(t, codes.InvalidArgument, st.Code())
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestSearchTransactions_CursorPagination_NoGapsOrDuplicates(t *testing.T) {
	dbMock, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer dbMock.Close()

	s := NewSQLStore(dbMock)
	cols := []string{
		"record_id", "user_id", "created_at", "is_train_eligible", "is_pre_fraud",
		"amount", "is_fraudulent", "fraud_type", "is_off_hours_txn", "merchant_risk_score",
		"velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score", "numerical_features", "categorical_features",
	}
	firstTS := time.Date(2025, 1, 10, 12, 0, 0, 0, time.UTC)
	secondTS := firstTS.Add(-1 * time.Minute)
	thirdTS := secondTS.Add(-1 * time.Minute)
	fourthTS := thirdTS.Add(-1 * time.Minute)

	mock.ExpectQuery(`(?s)SELECT em.record_id, em.user_id,.*ORDER BY em.created_at DESC, em.record_id DESC LIMIT 3`).
		WillReturnRows(sqlmock.NewRows(cols).
			AddRow("rec-3", "user-1", firstTS, true, false, 30.0, false, "", false, 10, 1, 1.0, 0.1, []byte("{}"), []byte("{}")).
			AddRow("rec-2", "user-1", firstTS, true, false, 20.0, false, "", false, 9, 1, 1.0, 0.1, []byte("{}"), []byte("{}")).
			AddRow("rec-1", "user-1", secondTS, true, false, 10.0, false, "", false, 8, 1, 1.0, 0.1, []byte("{}"), []byte("{}")))

	page1, nextCursor, meta1, err := s.SearchTransactions(context.Background(), &pb.SearchTransactionsRequest{
		Limit: 2,
	})
	require.NoError(t, err)
	require.Len(t, page1, 2)
	require.NotEmpty(t, nextCursor)
	require.NotNil(t, meta1)
	assert.False(t, meta1.Truncated)
	assert.Equal(t, "rec-3", page1[0].RecordId)
	assert.Equal(t, "rec-2", page1[1].RecordId)

	decoded, err := decodeTransactionCursor(nextCursor)
	require.NoError(t, err)
	require.NotNil(t, decoded)
	assert.Equal(t, "rec-2", decoded.RecordId)
	assert.True(t, firstTS.Equal(decoded.CreatedAt))

	mock.ExpectQuery(`(?s)SELECT em.record_id, em.user_id,.*WHERE \(em.created_at < \$1 OR \(em.created_at = \$2 AND em.record_id < \$3\)\).*ORDER BY em.created_at DESC, em.record_id DESC LIMIT 3`).
		WithArgs(decoded.CreatedAt, decoded.CreatedAt, decoded.RecordId).
		WillReturnRows(sqlmock.NewRows(cols).
			AddRow("rec-1", "user-1", secondTS, true, false, 10.0, false, "", false, 8, 1, 1.0, 0.1, []byte("{}"), []byte("{}")).
			AddRow("rec-0", "user-1", thirdTS, true, false, 9.0, false, "", false, 7, 1, 1.0, 0.1, []byte("{}"), []byte("{}")).
			AddRow("rec--1", "user-1", fourthTS, true, false, 8.0, false, "", false, 6, 1, 1.0, 0.1, []byte("{}"), []byte("{}")))

	page2, nextCursor2, meta2, err := s.SearchTransactions(context.Background(), &pb.SearchTransactionsRequest{
		Limit:  2,
		Cursor: nextCursor,
	})
	require.NoError(t, err)
	require.Len(t, page2, 2)
	require.NotNil(t, meta2)
	assert.False(t, meta2.Truncated)
	assert.Equal(t, "rec-1", page2[0].RecordId)
	assert.Equal(t, "rec-0", page2[1].RecordId)
	assert.NotEmpty(t, nextCursor2)

	seen := map[string]struct{}{}
	for _, tx := range append(page1, page2...) {
		if _, exists := seen[tx.RecordId]; exists {
			t.Fatalf("duplicate record across pages: %s", tx.RecordId)
		}
		seen[tx.RecordId] = struct{}{}
	}

	require.NoError(t, mock.ExpectationsWereMet())
}

func TestGetDatasetProfile(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	// 1. Total records count
	mock.ExpectQuery("SELECT COUNT").WithArgs("tenant-1").WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(100))

	// 2. Mock numeric feature profiling (one example: amount)
	mock.ExpectQuery("SELECT AVG\\(amount\\)").WillReturnRows(sqlmock.NewRows([]string{"mean", "stddev", "null_count", "min_val", "max_val"}).
		AddRow(100.0, 10.0, 0, 50.0, 150.0))

	// 3. Mock histogram query
	mock.ExpectQuery("SELECT WIDTH_BUCKET\\(amount").WillReturnRows(sqlmock.NewRows([]string{"bucket", "count"}).
		AddRow(1, 10).AddRow(2, 20).AddRow(3, 70))

	// Mock remaining features to avoid test failure due to unexpected queries
	for i := 0; i < 5; i++ {
		mock.ExpectQuery("SELECT AVG").WillReturnRows(sqlmock.NewRows([]string{"mean", "stddev", "null_count", "min_val", "max_val"}).
			AddRow(0.0, 0.0, 0, 0.0, 0.0))
	}

	resp, err := s.GetDatasetProfile(context.Background(), "test-dataset", 50, 10, "tenant-1")
	require.NoError(t, err)
	assert.Equal(t, int64(100), resp.TotalRecords)
	assert.NotEmpty(t, resp.FeatureProfiles)
	assert.Equal(t, "amount", resp.FeatureProfiles[0].Name)
}

func TestDiscoverJSONBKeys_DeterministicOrder(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	mock.ExpectQuery("SELECT DISTINCT kv\\.key").
		WithArgs(10).
		WillReturnRows(
			sqlmock.NewRows([]string{"key"}).
				AddRow("zeta").
				AddRow("alpha").
				AddRow("beta"),
		)

	keys, err := s.discoverJSONBKeys(context.Background(), "generated_records", "categorical_features", 10, "")
	require.NoError(t, err)
	assert.Equal(t, []string{"alpha", "beta", "zeta"}, keys)
	require.NoError(t, mock.ExpectationsWereMet())
}
func TestGetRuleStats(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	cutoff := time.Now().AddDate(0, 0, -30)
	rows := sqlmock.NewRows([]string{"rule_id", "triggered_count", "shadow_triggered_count", "approval_rate"}).
		AddRow("rule-1", 100, 10, 0.85)

	mock.ExpectQuery("SELECT").WithArgs(cutoff, "tenant-1").WillReturnRows(rows)

	stats, err := s.GetRuleStats(context.Background(), "", cutoff, "tenant-1")
	require.NoError(t, err)
	require.Len(t, stats, 1)
	assert.Equal(t, "rule-1", stats[0].RuleId)
	assert.Equal(t, int64(100), stats[0].TriggeredCount)
	assert.Equal(t, int64(10), stats[0].ShadowTriggeredCount)
	assert.Equal(t, 0.85, stats[0].ApprovalRate)
}
func TestGetAttribution(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	cutoff := time.Now().AddDate(0, 0, -7)
	rows := sqlmock.NewRows([]string{"date", "rule_id", "contribution_score", "volume"}).
		AddRow(time.Now(), "rule-2", 500, 50)

	mock.ExpectQuery("SELECT").WithArgs(cutoff, int32(20), "tenant-1").WillReturnRows(rows)

	items, err := s.GetAttribution(context.Background(), cutoff, 20, "tenant-1")
	require.NoError(t, err)
	require.Len(t, items, 1)
	assert.Equal(t, "rule-2", items[0].RuleId)
	assert.Equal(t, int64(500), items[0].ContributionScore)
}

func TestGetShadowComparison(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	rows := sqlmock.NewRows([]string{"total_evaluations", "divergent_scores_count", "active_score_mean", "shadow_score_mean"}).
		AddRow(100, 10, 50.0, 60.0)

	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	m, err := s.GetShadowComparison(context.Background(), 24, "tenant-1")
	require.NoError(t, err)
	assert.Equal(t, int64(100), m.TotalEvaluations)
	assert.Equal(t, int64(10), m.DivergentScoresCount)
	assert.Equal(t, 0.1, m.DivergentRate)
}
func TestGetLatestUserFeatures(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	rows := sqlmock.NewRows([]string{
		"record_id", "user_id", "snapshot_id", "computed_at",
		"velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score",
		"experimental_signals",
	}).AddRow("rec-1", "user-1", 101, time.Now(), 5, 1.2, 0.5, []byte(`{"bank_connections_24h": 2, "merchant_risk_score": 10}`))

	mock.ExpectQuery("SELECT").WithArgs("user-1", "tenant-1").WillReturnRows(rows)

	f, found, err := s.GetLatestUserFeatures(context.Background(), "user-1", "tenant-1")
	require.NoError(t, err)
	assert.True(t, found)
	assert.Equal(t, "user-1", f.UserId)
	assert.Equal(t, "101", f.SnapshotId)
	assert.Equal(t, int32(5), f.Velocity_24H)
	assert.Equal(t, int32(2), f.BankConnections_24H)
}

func TestBatchGetLatestUserFeatures(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	userIDs := []string{"user-1", "user-2"}
	rows := sqlmock.NewRows([]string{
		"record_id", "user_id", "snapshot_id", "computed_at",
		"velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score",
		"experimental_signals",
	}).
		AddRow("rec-1", "user-1", 101, time.Now(), 5, 1.2, 0.5, []byte(`{"bank_connections_24h": 2}`)).
		AddRow("rec-2", "user-2", 102, time.Now(), 10, 0.8, -0.3, []byte(`{"bank_connections_24h": 0}`))

	mock.ExpectQuery("SELECT").WithArgs(pq.Array(userIDs), "tenant-1").WillReturnRows(rows)

	results, err := s.BatchGetLatestUserFeatures(context.Background(), userIDs, "tenant-1")
	require.NoError(t, err)
	assert.Len(t, results, 2)
	assert.Contains(t, results, "user-1")
	assert.Contains(t, results, "user-2")
	assert.Equal(t, int32(5), results["user-1"].Velocity_24H)
}
