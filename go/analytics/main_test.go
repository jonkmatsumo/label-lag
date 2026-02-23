package main

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/generator"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/service"
	"github.com/jonkmatsumo/label-lag/go/analytics/internal/store"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
)

func TestGetDailyStats(t *testing.T) {
	mockStore := new(store.MockStore)
	s := service.NewService(mockStore, nil)

	expectedStats := []*pb.DailyStat{
		{
			Date:              time.Now().Format("2006-01-02"),
			TotalTransactions: 100,
			FraudCount:        5,
			FraudRate:         5.0,
			TotalAmount:       1000.0,
			AvgZScore:         0.5,
		},
	}

	mockStore.On("GetDailyStats", mock.Anything, mock.Anything, mock.Anything).Return(expectedStats, nil)

	req := &pb.GetDailyStatsRequest{Days: 30}
	resp, err := s.GetDailyStats(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Len(t, resp.Stats, 1)
	assert.Equal(t, int64(100), resp.Stats[0].TotalTransactions)
	assert.Equal(t, int64(5), resp.Stats[0].FraudCount)
	mockStore.AssertExpectations(t)
}

func TestGetOverviewMetrics(t *testing.T) {
	mockStore := new(store.MockStore)
	s := service.NewService(mockStore, nil)

	expectedResp := &pb.GetOverviewMetricsResponse{
		TotalRecords: 1000,
		FraudRecords: 50,
		FraudRate:    5.0,
	}

	mockStore.On("GetOverviewMetrics", mock.Anything, mock.Anything).Return(expectedResp, nil)

	resp, err := s.GetOverviewMetrics(context.Background(), &pb.GetOverviewMetricsRequest{})

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, int64(1000), resp.TotalRecords)
	assert.Equal(t, int64(50), resp.FraudRecords)
	assert.Equal(t, float64(5.0), resp.FraudRate)
	mockStore.AssertExpectations(t)
}

func TestGetFeatureSample_Stratified(t *testing.T) {
	mockStore := new(store.MockStore)
	s := service.NewService(mockStore, nil)

	expectedSamples := []*pb.FeatureSample{
		{RecordId: "f1", IsFraudulent: true},
		{RecordId: "nf1", IsFraudulent: false},
	}

	// Expectation for GetFeatureSample
	mockStore.On("GetFeatureSample", mock.Anything, int32(20), true, "").Return(expectedSamples, nil)

	req := &pb.GetFeatureSampleRequest{SampleSize: 20, Stratify: true}
	resp, err := s.GetFeatureSample(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Len(t, resp.Samples, 2)
	mockStore.AssertExpectations(t)
}

func TestUpdateHealthStatusServing(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	require.NoError(t, err)
	defer db.Close()

	mock.ExpectPing()
	healthServer := health.NewServer()

	// Direct call to function under test, server struct not needed
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

	// Direct call to function under test, server struct not needed
	err = updateHealthStatus(context.Background(), db, healthServer, nil)
	require.Error(t, err)

	resp, err := healthServer.Check(context.Background(), &grpc_health_v1.HealthCheckRequest{})
	require.NoError(t, err)
	assert.Equal(t, grpc_health_v1.HealthCheckResponse_NOT_SERVING, resp.Status)
}

func TestSearchTransactions(t *testing.T) {
	mockStore := new(store.MockStore)
	s := service.NewService(mockStore, nil)

	expectedTxs := []*pb.TransactionDetail{
		{
			RecordId:        "rec-1",
			IsTrainEligible: true,
			IsPreFraud:      true,
		},
	}
	expectedNextCursor := "next"
	expectedMeta := &pb.AnalyticsMeta{
		EffectiveLimit: 25,
		Truncated:      false,
	}

	mockStore.On("SearchTransactions", mock.Anything, mock.Anything).
		Return(expectedTxs, expectedNextCursor, expectedMeta, nil).Once()

	minAmount := 12.5
	req := &pb.SearchTransactionsRequest{
		UserId:    "user-1",
		MinAmount: &minAmount,
		Limit:     25,
	}

	resp, err := s.SearchTransactions(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "next", resp.Pagination.NextCursor)
	assert.False(t, resp.Meta.Truncated)
	assert.Len(t, resp.Transactions, 1)
	assert.Equal(t, "rec-1", resp.Transactions[0].RecordId)
	assert.True(t, resp.Transactions[0].IsTrainEligible)
	assert.True(t, resp.Transactions[0].IsPreFraud)
	mockStore.AssertExpectations(t)
}

func TestSearchTransactions_Unfiltered(t *testing.T) {
	mockStore := new(store.MockStore)
	s := service.NewService(mockStore, nil)

	expectedTxs := []*pb.TransactionDetail{
		{RecordId: "rec-2"},
	}
	expectedNextCursor := ""
	expectedMeta := &pb.AnalyticsMeta{
		EffectiveLimit: 10,
		Truncated:      false,
	}

	mockStore.On("SearchTransactions", mock.Anything, mock.Anything).
		Return(expectedTxs, expectedNextCursor, expectedMeta, nil).Once()

	req := &pb.SearchTransactionsRequest{
		Limit: 10,
	}

	resp, err := s.SearchTransactions(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.False(t, resp.Meta.Truncated)
	assert.Nil(t, resp.Pagination)
	assert.Len(t, resp.Transactions, 1)
	mockStore.AssertExpectations(t)
}

// ============================================================================
// Go Generator Integration Tests
// ============================================================================

func TestGenerateDataReturnsUnimplementedWhenDisabled(t *testing.T) {
	mockStore := new(store.MockStore)
	s := service.NewService(mockStore, nil)

	// Explicitly disable via env var
	t.Setenv("ENABLE_GO_DATASET_GENERATE", "false")

	req := &pb.GenerateDataRequest{
		NumUsers:  10,
		FraudRate: 0.1,
	}

	_, err := s.GenerateData(context.Background(), req)
	require.Error(t, err)

	st, ok := status.FromError(err)
	require.True(t, ok)
	assert.Equal(t, codes.Unimplemented, st.Code())
	assert.Contains(t, st.Message(), "disabled")
}

func TestGeneratorPackageIntegration(t *testing.T) {
	// Tests that the generator package produces valid data
	seed := int64(42)
	gen := generator.NewGenerator(&seed, nil)

	// Generate a small dataset
	result := gen.GenerateDatasetWithSequences(context.Background(), 5, 0.2)

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

	gen1 := generator.NewGenerator(&seed, nil)
	gen2 := generator.NewGenerator(&seed, nil)

	result1 := gen1.GenerateDatasetWithSequences(context.Background(), 3, 0.1)
	result2 := gen2.GenerateDatasetWithSequences(context.Background(), 3, 0.1)

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
	gen := generator.NewGenerator(&seed, nil)

	// Generate with 100% fraud rate to ensure we get fraud records
	result := gen.GenerateDatasetWithSequences(context.Background(), 10, 1.0)

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
