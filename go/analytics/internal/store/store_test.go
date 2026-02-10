package store

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetDailyStats(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	rows := sqlmock.NewRows([]string{"date", "total_transactions", "fraud_count", "fraud_rate", "total_amount", "avg_z_score"}).
		AddRow(time.Now(), 100, 5, 5.0, 1000.0, 0.5)

	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	stats, err := s.GetDailyStats(context.Background(), time.Now().AddDate(0, 0, -30))
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

	resp, err := s.GetOverviewMetrics(context.Background())
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
	mock.ExpectQuery("SELECT COALESCE\\(MIN\\(id\\), 0\\), COALESCE\\(MAX\\(id\\), 0\\), COUNT\\(\\*\\) FROM generated_records").
		WillReturnRows(sqlmock.NewRows([]string{"min", "max", "count"}).AddRow(1, 1000, 1000))

	// 3. Fraud rate query
	mock.ExpectQuery("SELECT CAST").WillReturnRows(sqlmock.NewRows([]string{"rate"}).AddRow(0.05))

	// 4. Sampling queries
	mock.ExpectQuery("(?s)SELECT.*is_fraudulent\\s*=\\s*true.*").
		WillReturnRows(sqlmock.NewRows([]string{"record_id", "is_fraudulent", "velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score"}).
			AddRow("f1", true, 1.0, 1.0, 1.0))

	mock.ExpectQuery("(?s)SELECT.*is_fraudulent\\s*=\\s*false.*").
		WillReturnRows(sqlmock.NewRows([]string{"record_id", "is_fraudulent", "velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score"}).
			AddRow("nf1", false, 0.0, 0.0, 0.0))

	samples, err := s.GetFeatureSample(context.Background(), 20, true) // Stratify is internal logic or param in store?
	// Wait, GetFeatureSample in store interface: GetFeatureSample(ctx context.Context, sampleSize int32) ([]*pb.FeatureSample, error)
	// It doesn't take 'stratify' param?
	// Let's check store interface.
	require.NoError(t, err)
	require.Len(t, samples, 2)
}

func TestSearchTransactions(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	mock.ExpectQuery(`(?s)SELECT COUNT\(\*\) FROM\s*\(.*SELECT em.record_id, em.user_id,.*WHERE em.user_id = \$1 AND gr.amount >= \$2\s*\)\s*as count_query`).
		WithArgs("user-1", 12.5).
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(2))

	mock.ExpectQuery(`(?s)SELECT em.record_id, em.user_id,.*WHERE em.user_id = \$1 AND gr.amount >= \$2.*ORDER BY em.created_at DESC LIMIT 25`).
		WithArgs("user-1", 12.5).
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
	}

	details, total, err := s.SearchTransactions(context.Background(), req, 25, 0)
	require.NoError(t, err)
	assert.Equal(t, int64(2), total)
	assert.Len(t, details, 1)
	assert.Equal(t, "rec-1", details[0].RecordId)
}

func TestGetDatasetProfile(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	// 1. Total records count
	mock.ExpectQuery("SELECT COUNT").WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(100))

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

	resp, err := s.GetDatasetProfile(context.Background(), "test-dataset", 50, 10)
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
	mock.ExpectQuery("SELECT DISTINCT key").
		WillReturnRows(
			sqlmock.NewRows([]string{"key"}).
				AddRow("zeta").
				AddRow("alpha").
				AddRow("beta"),
		)

	keys, err := s.discoverJSONBKeys(context.Background(), "generated_records", "categorical_features", 10)
	require.NoError(t, err)
	assert.Equal(t, []string{"alpha", "beta", "zeta"}, keys)
	require.NoError(t, mock.ExpectationsWereMet())
}
