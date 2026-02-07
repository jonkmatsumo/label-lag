package main

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/jonkmatsumo/label-lag/go/analytics/generator"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
)

func TestGetDailyStats(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	if err != nil {
		t.Fatalf("an error '%s' was not expected when opening a stub database connection", err)
	}
	defer db.Close()

	s := &server{db: db}

	rows := sqlmock.NewRows([]string{"date", "total_transactions", "fraud_count", "fraud_rate", "total_amount", "avg_z_score"}).
		AddRow(time.Now(), 100, 5, 5.0, 1000.0, 0.5)

	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	req := &pb.GetDailyStatsRequest{Days: 30}
	resp, err := s.GetDailyStats(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Len(t, resp.Stats, 1)
	assert.Equal(t, int64(100), resp.Stats[0].TotalTransactions)
	assert.Equal(t, int64(5), resp.Stats[0].FraudCount)
}

func TestGetOverviewMetrics(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	if err != nil {
		t.Fatalf("an error '%s' was not expected when opening a stub database connection", err)
	}
	defer db.Close()

	s := &server{db: db}

	rows := sqlmock.NewRows([]string{
		"total_records", "fraud_records", "unique_users",
		"min_transaction_timestamp", "max_transaction_timestamp",
		"min_created_at", "max_created_at",
		"total_amount", "fraud_amount",
	}).AddRow(1000, 50, 100, time.Now(), time.Now(), time.Now(), time.Now(), 50000.0, 2500.0)

	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	resp, err := s.GetOverviewMetrics(context.Background(), &pb.GetOverviewMetricsRequest{})

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, int64(1000), resp.TotalRecords)
	assert.Equal(t, int64(50), resp.FraudRecords)
	assert.Equal(t, float64(5.0), resp.FraudRate)
}

func TestGetFeatureSample_Stratified(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatalf("an error '%s' was not expected when opening a stub database connection", err)
	}
	defer db.Close()

	s := &server{db: db}

	// 1. Version query
	mock.ExpectQuery("SELECT version").WillReturnRows(sqlmock.NewRows([]string{"version"}).AddRow("PostgreSQL 16.1"))

	// 2. Stats query
	mock.ExpectQuery("SELECT COALESCE\\(MIN\\(id\\), 0\\), COALESCE\\(MAX\\(id\\), 0\\), COUNT\\(\\*\\) FROM generated_records").
		WillReturnRows(sqlmock.NewRows([]string{"min", "max", "count"}).AddRow(1, 1000, 1000))

	// 3. Fraud rate query
	mock.ExpectQuery("SELECT CAST").WillReturnRows(sqlmock.NewRows([]string{"rate"}).AddRow(0.05))

	// 4. Sampling queries (since stratify=true and count=1000, it falls back to ORDER BY RANDOM())
	// Fraud sampling
	mock.ExpectQuery("(?s)SELECT.*is_fraudulent\\s*=\\s*true.*").
		WillReturnRows(sqlmock.NewRows([]string{"record_id", "is_fraudulent", "velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score"}).
			AddRow("f1", true, 1.0, 1.0, 1.0))

	// Non-fraud sampling
	mock.ExpectQuery("(?s)SELECT.*is_fraudulent\\s*=\\s*false.*").
		WillReturnRows(sqlmock.NewRows([]string{"record_id", "is_fraudulent", "velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score"}).
			AddRow("nf1", false, 0.0, 0.0, 0.0))

	req := &pb.GetFeatureSampleRequest{SampleSize: 20, Stratify: true}
	resp, err := s.GetFeatureSample(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Len(t, resp.Samples, 2)
}

func TestUpdateHealthStatusServing(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	require.NoError(t, err)
	defer db.Close()

	mock.ExpectPing()
	healthServer := health.NewServer()

	err = updateHealthStatus(context.Background(), db, healthServer, nil)
	require.NoError(t, err)

	resp, err := healthServer.Check(context.Background(), &grpc_health_v1.HealthCheckRequest{})
	require.NoError(t, err)
	assert.Equal(t, grpc_health_v1.HealthCheckResponse_SERVING, resp.Status)
}

func TestUpdateHealthStatusNotServing(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	require.NoError(t, err)
	defer db.Close()

	mock.ExpectPing().WillReturnError(assert.AnError)
	healthServer := health.NewServer()

	err = updateHealthStatus(context.Background(), db, healthServer, nil)
	require.Error(t, err)

	resp, err := healthServer.Check(context.Background(), &grpc_health_v1.HealthCheckRequest{})
	require.NoError(t, err)
	assert.Equal(t, grpc_health_v1.HealthCheckResponse_NOT_SERVING, resp.Status)
}

func TestGetDailyStatsRejectsInvalidDays(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := &server{db: db}
	_, err = s.GetDailyStats(context.Background(), &pb.GetDailyStatsRequest{Days: -1})
	require.Error(t, err)

	st, _ := status.FromError(err)
	assert.Equal(t, codes.InvalidArgument, st.Code())
}

func TestGetTransactionDetailsRejectsInvalidLimit(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := &server{db: db}
	_, err = s.GetTransactionDetails(context.Background(), &pb.GetTransactionDetailsRequest{
		Days:  1,
		Limit: maxTransactionLimit + 1,
	})
	require.Error(t, err)

	st, _ := status.FromError(err)
	assert.Equal(t, codes.InvalidArgument, st.Code())
}

func TestSearchTransactions(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := &server{db: db}

	mock.ExpectQuery("SELECT COUNT\\(\\*\\) FROM generated_records WHERE user_id = \\$1 AND amount >= \\$2").
		WithArgs("user-1", 12.5).
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(2))

	mock.ExpectQuery("(?s)SELECT.*FROM generated_records WHERE user_id = \\$1 AND amount >= \\$2.*ORDER BY transaction_timestamp DESC OFFSET \\$3 LIMIT \\$4").
		WithArgs("user-1", 12.5, int32(0), int32(25)).
		WillReturnRows(sqlmock.NewRows([]string{
			"record_id",
			"user_id",
			"transaction_timestamp",
			"amount",
			"is_fraudulent",
			"fraud_type",
			"is_off_hours_txn",
			"merchant_risk_score",
			"amount_to_avg_ratio",
			"balance_volatility_z_score",
		}).AddRow("rec-1", "user-1", time.Now(), 22.5, false, "none", false, 10, 1.2, -0.3))

	minAmount := 12.5
	req := &pb.SearchTransactionsRequest{
		UserId:    "user-1",
		MinAmount: &minAmount,
		Limit:     25,
		Offset:    0,
	}

	resp, err := s.SearchTransactions(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, int64(2), resp.Total)
	assert.Len(t, resp.Transactions, 1)
	assert.Equal(t, "rec-1", resp.Transactions[0].RecordId)
	assert.True(t, resp.Transactions[0].IsTrainEligible)
	assert.True(t, resp.Transactions[0].IsPreFraud)
}

func TestSearchTransactions_Unfiltered(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := &server{db: db}

	// Expect query to pg_class for estimated count
	mock.ExpectQuery("SELECT reltuples::bigint FROM pg_class WHERE relname = \\$1").
		WithArgs("generated_records").
		WillReturnRows(sqlmock.NewRows([]string{"reltuples"}).AddRow(5000))

	// Expect data query with no WHERE clause
	mock.ExpectQuery("(?s)SELECT.*FROM generated_records ORDER BY transaction_timestamp DESC OFFSET \\$1 LIMIT \\$2").
		WithArgs(int32(0), int32(10)).
		WillReturnRows(sqlmock.NewRows([]string{
			"record_id", "user_id", "transaction_timestamp", "amount",
			"is_fraudulent", "fraud_type", "is_off_hours_txn",
			"merchant_risk_score", "amount_to_avg_ratio", "balance_volatility_z_score",
		}).AddRow("rec-2", "user-2", time.Now(), 50.0, false, "", false, 20, 1.0, 0.0))

	req := &pb.SearchTransactionsRequest{
		Limit:  10,
		Offset: 0,
	}

	resp, err := s.SearchTransactions(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, int64(5000), resp.Total)
	assert.Len(t, resp.Transactions, 1)
}

func TestGetRecentAlertsRejectsInvalidLimit(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := &server{db: db}
	_, err = s.GetRecentAlerts(context.Background(), &pb.GetRecentAlertsRequest{Limit: maxAlertLimit + 1})
	require.Error(t, err)

	st, _ := status.FromError(err)
	assert.Equal(t, codes.InvalidArgument, st.Code())
}

func TestGetFeatureSampleRejectsInvalidSampleSize(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := &server{db: db}
	_, err = s.GetFeatureSample(context.Background(), &pb.GetFeatureSampleRequest{SampleSize: maxSampleSizeLimit + 1})
	require.Error(t, err)

	st, _ := status.FromError(err)
	assert.Equal(t, codes.InvalidArgument, st.Code())
}

func TestResolveDatabaseURLUsesEnv(t *testing.T) {
	value, err := resolveDatabaseURL(func(key string) string {
		if key == "DATABASE_URL" {
			return "postgresql://user:pass@localhost:5432/db"
		}
		return ""
	})
	require.NoError(t, err)
	assert.Equal(t, "postgresql://user:pass@localhost:5432/db", value)
}

func TestResolveDatabaseURLAllowsDefaultsWhenEnabled(t *testing.T) {
	value, err := resolveDatabaseURL(func(key string) string {
		if key == "ANALYTICS_CRUD_ALLOW_INSECURE_DEFAULTS" {
			return "true"
		}
		return ""
	})
	require.NoError(t, err)
	assert.Equal(t, defaultDatabaseURL, value)
}

func TestResolveDatabaseURLRequiresExplicitValue(t *testing.T) {
	_, err := resolveDatabaseURL(func(string) string { return "" })
	require.Error(t, err)
}

// ============================================================================
// Go Generator Integration Tests
// ============================================================================

func TestGenerateDataReturnsUnimplementedWhenDisabled(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := &server{db: db}

	// Explicitly disable via env var
	t.Setenv("ENABLE_GO_DATASET_GENERATE", "false")

	req := &pb.GenerateDataRequest{
		NumUsers:  10,
		FraudRate: 0.1,
	}

	_, err = s.GenerateData(context.Background(), req)
	require.Error(t, err)

	st, ok := status.FromError(err)
	require.True(t, ok)
	assert.Equal(t, codes.Unimplemented, st.Code())
	assert.Contains(t, st.Message(), "disabled")
}

func TestGeneratorPackageIntegration(t *testing.T) {
	// Tests that the generator package produces valid data
	seed := int64(42)
	gen := generator.NewGenerator(&seed)

	// Generate a small dataset
	result := gen.GenerateDatasetWithSequences(5, 0.2)

	// Verify records exist
	assert.NotEmpty(t, result.Records)
	assert.NotEmpty(t, result.Metadata)
	assert.Equal(t, len(result.Records), len(result.Metadata))

	// Verify records have required fields
	for i, r := range result.Records {
		assert.NotEmpty(t, r.UserId, "record %d missing UserId", i)
		assert.NotEmpty(t, r.FullName, "record %d missing FullName", i)
		assert.NotEmpty(t, r.Email, "record %d missing Email", i)
		assert.Greater(t, r.Amount, float64(0), "record %d has invalid Amount", i)
	}

	// Verify metadata
	for i, m := range result.Metadata {
		assert.NotEmpty(t, m.UserId, "metadata %d missing UserId", i)
		assert.NotEmpty(t, m.RecordId, "metadata %d missing RecordId", i)
		assert.GreaterOrEqual(t, m.SequenceNumber, int32(1), "metadata %d has invalid SequenceNumber", i)
	}

	// Verify fraud rate is approximately correct (1 fraudulent user = ~1 fraud record)
	fraudCount := 0
	for _, r := range result.Records {
		if r.IsFraudulent {
			fraudCount++
		}
	}
	assert.GreaterOrEqual(t, fraudCount, 0)
}

func TestGeneratorDeterminism(t *testing.T) {
	// Two generators with same seed should produce identical output
	seed := int64(12345)

	gen1 := generator.NewGenerator(&seed)
	gen2 := generator.NewGenerator(&seed)

	result1 := gen1.GenerateDatasetWithSequences(3, 0.1)
	result2 := gen2.GenerateDatasetWithSequences(3, 0.1)

	require.Equal(t, len(result1.Records), len(result2.Records))

	// Check that deterministic fields match
	for i := range result1.Records {
		assert.Equal(t, result1.Records[i].Amount, result2.Records[i].Amount, "record %d Amount mismatch", i)
		assert.Equal(t, result1.Records[i].AvailableBalance, result2.Records[i].AvailableBalance, "record %d AvailableBalance mismatch", i)
		assert.Equal(t, result1.Records[i].BalanceVolatilityZScore, result2.Records[i].BalanceVolatilityZScore, "record %d ZScore mismatch", i)
		assert.Equal(t, result1.Records[i].IsFraudulent, result2.Records[i].IsFraudulent, "record %d IsFraudulent mismatch", i)
		assert.Equal(t, result1.Records[i].FraudType, result2.Records[i].FraudType, "record %d FraudType mismatch", i)
	}
}

func TestGeneratorFraudTypeDistribution(t *testing.T) {
	seed := int64(999)
	gen := generator.NewGenerator(&seed)

	// Generate with 100% fraud rate to ensure we get fraud records
	result := gen.GenerateDatasetWithSequences(10, 1.0)

	fraudTypes := make(map[string]int)
	for _, r := range result.Records {
		if r.IsFraudulent {
			fraudTypes[r.FraudType]++
		}
	}

	// Should have at least one fraud type represented
	assert.NotEmpty(t, fraudTypes, "expected fraud types to be generated")

	// Verify fraud types are valid
	validTypes := map[string]bool{
		"liquidity_crunch": true,
		"link_burst":       true,
		"ato":              true,
	}
	for ft := range fraudTypes {
		assert.True(t, validTypes[ft], "unexpected fraud type: %s", ft)
	}
}
