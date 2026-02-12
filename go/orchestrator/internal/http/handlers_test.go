package httpserver

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	forecastv1 "github.com/jonkmatsumo/label-lag/go/forecast/proto/forecastv1"
	grpcclient "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc"
	inferencev1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc/inferencev1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	trainingv1 "github.com/jonkmatsumo/label-lag/go/training/proto/trainingv1"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestHandleEvaluateSignal_RejectsLargeBody(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 32, "", "")

	body := strings.Repeat("a", 64)
	req := httptest.NewRequest(http.MethodPost, "/evaluate/signal", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.handleEvaluateSignal(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected status %d, got %d", http.StatusRequestEntityTooLarge, rec.Code)
	}
}

func TestHandleEvaluateSignal_RejectsUnknownFields(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	payload := `{"user_id":"u1","amount":12.3,"currency":"USD","client_transaction_id":"t1","unknown":"x"}`
	req := httptest.NewRequest(http.MethodPost, "/evaluate/signal", strings.NewReader(payload))
	rec := httptest.NewRecorder()

	handler.handleEvaluateSignal(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
}

func TestHandleReadyReportsHealthy(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, stubInferenceClient{}, nil, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()

	handler.handleReady(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["status"] != "ready" {
		t.Fatalf("expected ready status, got %v", payload["status"])
	}
}

func TestHandleReadyReportsUnhealthy(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, stubInferenceClient{readyErr: errors.New("not ready")}, nil, stubTrainingClient{}, stubForecastClient{}, errProvider{}, 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()

	handler.handleReady(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}
}

func TestHandleSearchTransactions(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		resp: &crudv1.SearchTransactionsResponse{
			Transactions: []*crudv1.TransactionDetail{
				{
					RecordId:                "rec-1",
					UserId:                  "user-1",
					CreatedAt:               timestamppb.New(time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC)),
					IsTrainEligible:         true,
					IsPreFraud:              true,
					Amount:                  10.5,
					IsFraudulent:            false,
					FraudType:               "none",
					IsOffHoursTxn:           false,
					MerchantRiskScore:       12,
					Velocity_24H:            0,
					AmountToAvgRatio_30D:    1.1,
					BalanceVolatilityZScore: -0.2,
				},
			},
			Total: 1,
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodPost, "/analytics/transactions/search", strings.NewReader(`{"user_id":"user-1","limit":10}`))
	rec := httptest.NewRecorder()

	handler.handleSearchTransactions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["total"] != float64(1) {
		t.Fatalf("expected total 1, got %v", payload["total"])
	}
	txs, ok := payload["transactions"].([]any)
	if !ok || len(txs) != 1 {
		t.Fatalf("expected 1 transaction, got %v", payload["transactions"])
	}
	tx := txs[0].(map[string]any)
	if tx["record_id"] != "rec-1" {
		t.Fatalf("expected record_id rec-1, got %v", tx["record_id"])
	}
	if tx["created_at"] == "" {
		t.Fatalf("expected created_at to be set")
	}
}

func TestHandleAnalyticsOverview(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		overviewResp: &crudv1.GetOverviewMetricsResponse{
			TotalRecords:            100,
			FraudRecords:            5,
			FraudRate:               0.05,
			UniqueUsers:             42,
			MinTransactionTimestamp: timestamppb.New(time.Date(2025, 2, 1, 0, 0, 0, 0, time.UTC)),
			MaxTransactionTimestamp: timestamppb.New(time.Date(2025, 2, 2, 0, 0, 0, 0, time.UTC)),
			MinCreatedAt:            timestamppb.New(time.Date(2025, 2, 1, 1, 0, 0, 0, time.UTC)),
			MaxCreatedAt:            timestamppb.New(time.Date(2025, 2, 2, 1, 0, 0, 0, time.UTC)),
			TotalAmount:             123.45,
			FraudAmount:             6.78,
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/analytics/overview", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsOverview(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["total_records"] != float64(100) {
		t.Fatalf("expected total_records 100, got %v", payload["total_records"])
	}
	if payload["fraud_records"] != float64(5) {
		t.Fatalf("expected fraud_records 5, got %v", payload["fraud_records"])
	}
	if payload["min_transaction_timestamp"] == "" {
		t.Fatalf("expected min_transaction_timestamp to be set")
	}
}

func TestHandleAnalyticsOverviewPropagatesErrors(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		err: &grpcclient.RPCError{Code: codes.Unavailable, Message: "downstream unavailable"},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/analytics/overview", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsOverview(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}
}

func TestHandleAnalyticsDailyStats(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		dailyStatsResp: &crudv1.GetDailyStatsResponse{
			Stats: []*crudv1.DailyStat{
				{
					Date:              "2025-02-01",
					TotalTransactions: 10,
					FraudCount:        1,
					FraudRate:         0.1,
					TotalAmount:       100.0,
					AvgZScore:         0.25,
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/analytics/daily-stats?days=7", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsDailyStats(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if stub.lastDailyStatsReq == nil || stub.lastDailyStatsReq.GetDays() != 7 {
		t.Fatalf("expected days 7, got %v", stub.lastDailyStatsReq)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	stats, ok := payload["stats"].([]any)
	if !ok || len(stats) != 1 {
		t.Fatalf("expected 1 stat, got %v", payload["stats"])
	}
	stat := stats[0].(map[string]any)
	if stat["total_transactions"] != float64(10) {
		t.Fatalf("expected total_transactions 10, got %v", stat["total_transactions"])
	}
}

func TestHandleAnalyticsTransactions(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		transactionDetailsResp: &crudv1.GetTransactionDetailsResponse{
			Transactions: []*crudv1.TransactionDetail{
				{
					RecordId:                "rec-1",
					UserId:                  "user-1",
					CreatedAt:               timestamppb.New(time.Date(2025, 2, 1, 2, 3, 4, 0, time.UTC)),
					IsTrainEligible:         true,
					IsPreFraud:              false,
					Amount:                  99.5,
					IsFraudulent:            false,
					FraudType:               "",
					IsOffHoursTxn:           false,
					MerchantRiskScore:       12,
					Velocity_24H:            3,
					AmountToAvgRatio_30D:    1.2,
					BalanceVolatilityZScore: 0.4,
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/analytics/transactions", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsTransactions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if stub.lastTransactionDetailsReq == nil || stub.lastTransactionDetailsReq.GetDays() != 7 || stub.lastTransactionDetailsReq.GetLimit() != 1000 {
		t.Fatalf("expected default days and limit, got %v", stub.lastTransactionDetailsReq)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	txs, ok := payload["transactions"].([]any)
	if !ok || len(txs) != 1 {
		t.Fatalf("expected 1 transaction, got %v", payload["transactions"])
	}
	tx := txs[0].(map[string]any)
	if tx["record_id"] != "rec-1" {
		t.Fatalf("expected record_id rec-1, got %v", tx["record_id"])
	}
}

func TestHandleAnalyticsRecentAlerts(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		recentAlertsResp: &crudv1.GetRecentAlertsResponse{
			Alerts: []*crudv1.Alert{
				{
					RecordId:                "rec-1",
					UserId:                  "user-1",
					CreatedAt:               timestamppb.New(time.Date(2025, 2, 1, 2, 3, 4, 0, time.UTC)),
					Amount:                  250.75,
					IsFraudulent:            true,
					FraudType:               "stolen",
					MerchantRiskScore:       15,
					Velocity_24H:            2,
					AmountToAvgRatio_30D:    1.1,
					BalanceVolatilityZScore: 0.5,
					ComputedRiskScore:       99,
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/analytics/recent-alerts", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsRecentAlerts(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if stub.lastRecentAlertsReq == nil || stub.lastRecentAlertsReq.GetLimit() != 50 {
		t.Fatalf("expected default limit 50, got %v", stub.lastRecentAlertsReq)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	alerts, ok := payload["alerts"].([]any)
	if !ok || len(alerts) != 1 {
		t.Fatalf("expected 1 alert, got %v", payload["alerts"])
	}
	alert := alerts[0].(map[string]any)
	if alert["record_id"] != "rec-1" {
		t.Fatalf("expected record_id rec-1, got %v", alert["record_id"])
	}
}

func TestHandleAnalyticsFeatureSample(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		featureSampleResp: &crudv1.GetFeatureSampleResponse{
			Samples: []*crudv1.FeatureSample{
				{
					RecordId:                "rec-1",
					IsFraudulent:            true,
					Velocity_24H:            2.1,
					AmountToAvgRatio_30D:    1.5,
					BalanceVolatilityZScore: 0.3,
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/analytics/feature-sample?sample_size=5&stratify=false", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsFeatureSample(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if stub.lastFeatureSampleReq == nil || stub.lastFeatureSampleReq.GetSampleSize() != 5 || stub.lastFeatureSampleReq.GetStratify() {
		t.Fatalf("expected sample_size 5 and stratify false, got %v", stub.lastFeatureSampleReq)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	samples := payload["samples"].([]any)
	if len(samples) != 1 {
		t.Fatalf("expected 1 sample, got %v", payload["samples"])
	}
}

func TestHandleMonitoringDrift(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := stubForecastClient{
		driftResp: &forecastv1.GetDriftMonitoringResponse{
			DriftScore:    0.25,
			DriftDetected: true,
		},
	}
	handler := NewHandler(logger, nil, nil, stubTrainingClient{}, stub, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/monitoring/drift?hours=24", nil)
	rec := httptest.NewRecorder()

	handler.handleMonitoringDrift(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["drift_score"] != 0.25 {
		t.Fatalf("expected drift_score 0.25, got %v", payload["drift_score"])
	}
	if payload["drift_detected"] != true {
		t.Fatalf("expected drift_detected true, got %v", payload["drift_detected"])
	}
}

func TestHandleMetricsShadowComparison(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		shadowComparisonResp: &crudv1.GetShadowComparisonResponse{
			Metrics: &crudv1.ShadowModeMetrics{
				TotalEvaluations: 100,
				ActiveScoreMean:  50,
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/metrics/shadow/comparison?hours=24", nil)
	rec := httptest.NewRecorder()

	handler.handleMetricsShadowComparison(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	metrics := payload["metrics"].(map[string]any)
	if metrics["total_evaluations"] != float64(100) {
		t.Fatalf("expected total_evaluations 100, got %v", metrics["total_evaluations"])
	}
}

func TestHandleBacktestResults(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		backtestResultsResp: &crudv1.ListBacktestResultsResponse{
			Results: []*crudv1.BacktestResult{
				{
					JobId:          "job-1",
					RuleId:         "rule-1",
					RulesetVersion: "v1",
					StartDate:      timestamppb.New(time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)),
					EndDate:        timestamppb.New(time.Date(2025, 1, 2, 0, 0, 0, 0, time.UTC)),
					Metrics: &crudv1.BacktestMetrics{
						TotalRecords:      100,
						MatchedCount:      10,
						MatchRate:         0.1,
						ScoreDistribution: map[string]int32{"0-10": 5},
						ScoreMean:         50,
						ScoreStd:          5,
						ScoreMin:          10,
						ScoreMax:          90,
						RejectedCount:     2,
						RejectedRate:      0.02,
					},
					CompletedAt: timestamppb.New(time.Date(2025, 1, 2, 1, 0, 0, 0, time.UTC)),
				},
				{
					JobId:          "job-2",
					RulesetVersion: "v2",
					StartDate:      timestamppb.New(time.Date(2025, 1, 3, 0, 0, 0, 0, time.UTC)),
					EndDate:        timestamppb.New(time.Date(2025, 1, 4, 0, 0, 0, 0, time.UTC)),
					Metrics:        &crudv1.BacktestMetrics{},
					CompletedAt:    timestamppb.New(time.Date(2025, 1, 4, 1, 0, 0, 0, time.UTC)),
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/backtest/results?rule_id=rule-1&start_date=2025-01-01&end_date=2025-01-31&limit=1", nil)
	rec := httptest.NewRecorder()

	handler.handleBacktestResults(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if stub.lastBacktestResultsReq == nil || stub.lastBacktestResultsReq.GetRuleId() != "rule-1" {
		t.Fatalf("expected rule_id rule-1, got %v", stub.lastBacktestResultsReq)
	}
	if stub.lastBacktestResultsReq.GetStartDate() == nil || stub.lastBacktestResultsReq.GetEndDate() == nil {
		t.Fatalf("expected start/end dates to be set")
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	results, ok := payload["results"].([]any)
	if !ok || len(results) != 1 {
		t.Fatalf("expected 1 result, got %v", payload["results"])
	}
	if payload["total"] != float64(1) {
		t.Fatalf("expected total 1, got %v", payload["total"])
	}
}

type stubInferenceClient struct {
	readyErr error
}

func (s stubInferenceClient) Ready(context.Context) error {
	return s.readyErr
}

func (s stubInferenceClient) Score(context.Context, *inferencev1.ScoreRequest) (*inferencev1.ScoreResponse, error) {
	return &inferencev1.ScoreResponse{
		ModelScore:   0.1,
		ModelVersion: "v1",
	}, nil
}

type stubForecastClient struct {
	predictResp *forecastv1.PredictSignalResponse
	driftResp   *forecastv1.GetDriftMonitoringResponse
	err         error
}

func (s stubForecastClient) GetHealth(ctx context.Context, req *forecastv1.GetHealthRequest) (*forecastv1.GetHealthResponse, error) {
	return &forecastv1.GetHealthResponse{Status: "ok"}, s.err
}

func (s stubForecastClient) GetDriftMonitoring(ctx context.Context, req *forecastv1.GetDriftMonitoringRequest) (*forecastv1.GetDriftMonitoringResponse, error) {
	return s.driftResp, s.err
}

func (s stubForecastClient) GetScoreDistribution(ctx context.Context, req *forecastv1.GetScoreDistributionRequest) (*forecastv1.GetScoreDistributionResponse, error) {
	return &forecastv1.GetScoreDistributionResponse{}, s.err
}

func (s stubForecastClient) ReloadModel(ctx context.Context, req *forecastv1.ReloadModelRequest) (*forecastv1.ReloadModelResponse, error) {
	return &forecastv1.ReloadModelResponse{Success: true}, s.err
}

func (s stubForecastClient) DeployModel(ctx context.Context, req *forecastv1.DeployModelRequest) (*forecastv1.DeployModelResponse, error) {
	return &forecastv1.DeployModelResponse{Success: true}, s.err
}

func (s stubForecastClient) PredictSignal(ctx context.Context, req *forecastv1.PredictSignalRequest) (*forecastv1.PredictSignalResponse, error) {
	if s.predictResp != nil {
		return s.predictResp, s.err
	}
	return &forecastv1.PredictSignalResponse{
		Probability:  0.1,
		ModelVersion: "v1",
	}, s.err
}

type stubTrainingClient struct {
	err error
}

func (s stubTrainingClient) Train(ctx context.Context, req *trainingv1.TrainRequest) (*trainingv1.TrainResponse, error) {
	return &trainingv1.TrainResponse{Success: true, RunId: "run-1"}, s.err
}

func (s stubTrainingClient) ClearData(ctx context.Context, req *trainingv1.ClearDataRequest) (*trainingv1.ClearDataResponse, error) {
	return &trainingv1.ClearDataResponse{Success: true}, s.err
}

func (s stubTrainingClient) GetHealth(ctx context.Context, req *trainingv1.GetHealthRequest) (*trainingv1.GetHealthResponse, error) {
	return &trainingv1.GetHealthResponse{Ready: true}, s.err
}

type stubAnalyticsClient struct {
	resp                   *crudv1.SearchTransactionsResponse
	dailyStatsResp         *crudv1.GetDailyStatsResponse
	overviewResp           *crudv1.GetOverviewMetricsResponse
	transactionDetailsResp *crudv1.GetTransactionDetailsResponse
	recentAlertsResp       *crudv1.GetRecentAlertsResponse
	featureSampleResp      *crudv1.GetFeatureSampleResponse
	generateDataResp       *crudv1.GenerateDataResponse

	backtestResultsResp         *crudv1.ListBacktestResultsResponse
	listDecisionsResp           *crudv1.ListDecisionsResponse
	getDecisionResp             *crudv1.GetDecisionResponse
	listJobsResp                *crudv1.ListJobsResponse
	getJobResp                  *crudv1.GetJobResponse
	getJobEventsResp            *crudv1.GetJobEventsResponse
	getJobSummaryResp           *crudv1.GetJobSummaryResponse
	listTrainingRunsResp        *crudv1.ListTrainingRunsResponse
	getTrainingRunResp          *crudv1.GetTrainingRunResponse
	listModelVersionsResp       *crudv1.ListModelVersionsResponse
	getMetricSeriesResp         *crudv1.GetMetricSeriesResponse
	listDatasetProfilesResp     *crudv1.ListDatasetProfilesResponse
	getLatestDatasetProfileResp *crudv1.GetLatestDatasetProfileResponse
	compareDatasetProfilesResp  *crudv1.CompareDatasetProfilesResponse

	err                       error
	lastReq                   *crudv1.SearchTransactionsRequest
	lastDailyStatsReq         *crudv1.GetDailyStatsRequest
	lastOverviewReq           *crudv1.GetOverviewMetricsRequest
	lastTransactionDetailsReq *crudv1.GetTransactionDetailsRequest
	lastRecentAlertsReq       *crudv1.GetRecentAlertsRequest
	lastFeatureSampleReq      *crudv1.GetFeatureSampleRequest
	lastBacktestResultsReq    *crudv1.ListBacktestResultsRequest
	lastClearAllDataReq       *crudv1.ClearAllDataRequest
	shadowComparisonResp      *crudv1.GetShadowComparisonResponse
}

func (s *stubAnalyticsClient) SearchTransactions(ctx context.Context, req *crudv1.SearchTransactionsRequest) (*crudv1.SearchTransactionsResponse, error) {
	s.lastReq = req
	return s.resp, s.err
}

func (s *stubAnalyticsClient) GetDailyStats(ctx context.Context, req *crudv1.GetDailyStatsRequest) (*crudv1.GetDailyStatsResponse, error) {
	s.lastDailyStatsReq = req
	return s.dailyStatsResp, s.err
}

func (s *stubAnalyticsClient) GetOverviewMetrics(ctx context.Context, req *crudv1.GetOverviewMetricsRequest) (*crudv1.GetOverviewMetricsResponse, error) {
	s.lastOverviewReq = req
	return s.overviewResp, s.err
}

func (s *stubAnalyticsClient) GetTransactionDetails(ctx context.Context, req *crudv1.GetTransactionDetailsRequest) (*crudv1.GetTransactionDetailsResponse, error) {
	s.lastTransactionDetailsReq = req
	return s.transactionDetailsResp, s.err
}

func (s *stubAnalyticsClient) GetRecentAlerts(ctx context.Context, req *crudv1.GetRecentAlertsRequest) (*crudv1.GetRecentAlertsResponse, error) {
	s.lastRecentAlertsReq = req
	return s.recentAlertsResp, s.err
}

func (s *stubAnalyticsClient) GetFeatureSample(ctx context.Context, req *crudv1.GetFeatureSampleRequest) (*crudv1.GetFeatureSampleResponse, error) {
	s.lastFeatureSampleReq = req
	return s.featureSampleResp, s.err
}

func (s *stubAnalyticsClient) ListBacktestResults(ctx context.Context, req *crudv1.ListBacktestResultsRequest) (*crudv1.ListBacktestResultsResponse, error) {
	s.lastBacktestResultsReq = req
	return s.backtestResultsResp, s.err
}

func (s *stubAnalyticsClient) ClearAllData(ctx context.Context, req *crudv1.ClearAllDataRequest) (*crudv1.ClearAllDataResponse, error) {
	s.lastClearAllDataReq = req
	return &crudv1.ClearAllDataResponse{Success: true}, s.err
}

func (s *stubAnalyticsClient) GetFeatures(ctx context.Context, userID, tenantID string) (map[string]any, error) {
	if s.err != nil {
		return nil, s.err
	}
	if s.resp != nil && len(s.resp.Transactions) > 0 {
		tx := s.resp.Transactions[0]
		return map[string]any{
			"velocity_24h": float64(tx.Velocity_24H),
		}, nil
	}
	return nil, nil
}

func (s *stubAnalyticsClient) ListRuleVersions(ctx context.Context, req *crudv1.ListRuleVersionsRequest) (*crudv1.ListRuleVersionsResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) GetRuleVersion(ctx context.Context, req *crudv1.GetRuleVersionRequest) (*crudv1.GetRuleVersionResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) PublishRuleVersion(ctx context.Context, req *crudv1.PublishRuleVersionRequest) (*crudv1.PublishRuleVersionResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) GetRuleReadiness(ctx context.Context, req *crudv1.GetRuleReadinessRequest) (*crudv1.GetRuleReadinessResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) DiffRuleVersions(ctx context.Context, req *crudv1.DiffRuleVersionsRequest) (*crudv1.DiffRuleVersionsResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) SaveRule(ctx context.Context, req *crudv1.SaveRuleRequest) (*crudv1.SaveRuleResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) GetRule(ctx context.Context, req *crudv1.GetRuleRequest) (*crudv1.GetRuleResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) ListRules(ctx context.Context, req *crudv1.ListRulesRequest) (*crudv1.ListRulesResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) DeleteRule(ctx context.Context, req *crudv1.DeleteRuleRequest) (*crudv1.DeleteRuleResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) GetRuleStats(ctx context.Context, req *crudv1.GetRuleStatsRequest) (*crudv1.GetRuleStatsResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) GetAttribution(ctx context.Context, req *crudv1.GetAttributionRequest) (*crudv1.GetAttributionResponse, error) {
	return nil, s.err
}

func (s *stubAnalyticsClient) LogInferenceEvent(ctx context.Context, req *crudv1.LogInferenceEventRequest) (*crudv1.LogInferenceEventResponse, error) {
	return &crudv1.LogInferenceEventResponse{Success: true}, s.err
}

func (s *stubAnalyticsClient) CompareBacktests(ctx context.Context, req *crudv1.CompareBacktestsRequest) (*crudv1.CompareBacktestsResponse, error) {
	return &crudv1.CompareBacktestsResponse{}, s.err
}

func (s *stubAnalyticsClient) GetShadowComparison(ctx context.Context, req *crudv1.GetShadowComparisonRequest) (*crudv1.GetShadowComparisonResponse, error) {
	return s.shadowComparisonResp, s.err
}

func (s *stubAnalyticsClient) ListDecisions(ctx context.Context, req *crudv1.ListDecisionsRequest) (*crudv1.ListDecisionsResponse, error) {
	if s.listDecisionsResp != nil {
		return s.listDecisionsResp, s.err
	}
	return &crudv1.ListDecisionsResponse{}, s.err
}

func (s *stubAnalyticsClient) GetDecision(ctx context.Context, req *crudv1.GetDecisionRequest) (*crudv1.GetDecisionResponse, error) {
	if s.getDecisionResp != nil {
		return s.getDecisionResp, s.err
	}
	return &crudv1.GetDecisionResponse{}, s.err
}

func (s *stubAnalyticsClient) GetDecisionTrace(ctx context.Context, req *crudv1.GetDecisionTraceRequest) (*crudv1.GetDecisionTraceResponse, error) {
	return &crudv1.GetDecisionTraceResponse{}, s.err
}

func (s *stubAnalyticsClient) GetRuleImpact(ctx context.Context, req *crudv1.GetRuleImpactRequest) (*crudv1.GetRuleImpactResponse, error) {
	return &crudv1.GetRuleImpactResponse{}, s.err
}

func (s *stubAnalyticsClient) GetKpis(ctx context.Context, req *crudv1.GetKpisRequest) (*crudv1.GetKpisResponse, error) {
	return &crudv1.GetKpisResponse{}, s.err
}

func (s *stubAnalyticsClient) GetVolumeSeries(ctx context.Context, req *crudv1.GetVolumeSeriesRequest) (*crudv1.GetVolumeSeriesResponse, error) {
	return &crudv1.GetVolumeSeriesResponse{}, s.err
}

func (s *stubAnalyticsClient) GetConfusionMatrix(ctx context.Context, req *crudv1.GetConfusionMatrixRequest) (*crudv1.GetConfusionMatrixResponse, error) {
	return &crudv1.GetConfusionMatrixResponse{}, s.err
}

func (s *stubAnalyticsClient) ListJobs(ctx context.Context, req *crudv1.ListJobsRequest) (*crudv1.ListJobsResponse, error) {
	if s.listJobsResp != nil {
		return s.listJobsResp, s.err
	}
	return &crudv1.ListJobsResponse{}, s.err
}

func (s *stubAnalyticsClient) GetJob(ctx context.Context, req *crudv1.GetJobRequest) (*crudv1.GetJobResponse, error) {
	if s.getJobResp != nil {
		return s.getJobResp, s.err
	}
	return &crudv1.GetJobResponse{}, s.err
}

func (s *stubAnalyticsClient) GetJobEvents(ctx context.Context, req *crudv1.GetJobEventsRequest) (*crudv1.GetJobEventsResponse, error) {
	if s.getJobEventsResp != nil {
		return s.getJobEventsResp, s.err
	}
	return &crudv1.GetJobEventsResponse{}, s.err
}

func (s *stubAnalyticsClient) GetDatasetSummary(ctx context.Context, req *crudv1.GetDatasetSummaryRequest) (*crudv1.GetDatasetSummaryResponse, error) {
	return &crudv1.GetDatasetSummaryResponse{}, s.err
}

func (s *stubAnalyticsClient) ListDatasetProfiles(ctx context.Context, req *crudv1.ListDatasetProfilesRequest) (*crudv1.ListDatasetProfilesResponse, error) {
	if s.listDatasetProfilesResp != nil {
		return s.listDatasetProfilesResp, s.err
	}
	return &crudv1.ListDatasetProfilesResponse{}, s.err
}

func (s *stubAnalyticsClient) CompareDatasetProfiles(ctx context.Context, req *crudv1.CompareDatasetProfilesRequest) (*crudv1.CompareDatasetProfilesResponse, error) {
	if s.compareDatasetProfilesResp != nil {
		return s.compareDatasetProfilesResp, s.err
	}
	return &crudv1.CompareDatasetProfilesResponse{}, s.err
}

func (s *stubAnalyticsClient) ListTrainingRuns(ctx context.Context, req *crudv1.ListTrainingRunsRequest) (*crudv1.ListTrainingRunsResponse, error) {
	if s.listTrainingRunsResp != nil {
		return s.listTrainingRunsResp, s.err
	}
	return &crudv1.ListTrainingRunsResponse{}, s.err
}

func (s *stubAnalyticsClient) GetTrainingRun(ctx context.Context, req *crudv1.GetTrainingRunRequest) (*crudv1.GetTrainingRunResponse, error) {
	if s.getTrainingRunResp != nil {
		return s.getTrainingRunResp, s.err
	}
	return &crudv1.GetTrainingRunResponse{}, s.err
}

func (s *stubAnalyticsClient) GetMetricSeries(ctx context.Context, req *crudv1.GetMetricSeriesRequest) (*crudv1.GetMetricSeriesResponse, error) {
	if s.getMetricSeriesResp != nil {
		return s.getMetricSeriesResp, s.err
	}
	return &crudv1.GetMetricSeriesResponse{}, s.err
}

func (s *stubAnalyticsClient) GetJobSummary(ctx context.Context, req *crudv1.GetJobSummaryRequest) (*crudv1.GetJobSummaryResponse, error) {
	if s.getJobSummaryResp != nil {
		return s.getJobSummaryResp, s.err
	}
	return &crudv1.GetJobSummaryResponse{}, s.err
}

func (s *stubAnalyticsClient) ListModelVersions(ctx context.Context, req *crudv1.ListModelVersionsRequest) (*crudv1.ListModelVersionsResponse, error) {
	if s.listModelVersionsResp != nil {
		return s.listModelVersionsResp, s.err
	}
	return &crudv1.ListModelVersionsResponse{}, s.err
}

func (s *stubAnalyticsClient) GetLatestDatasetProfile(ctx context.Context, req *crudv1.GetLatestDatasetProfileRequest) (*crudv1.GetLatestDatasetProfileResponse, error) {
	if s.getLatestDatasetProfileResp != nil {
		return s.getLatestDatasetProfileResp, s.err
	}
	return &crudv1.GetLatestDatasetProfileResponse{}, s.err
}

func (s *stubAnalyticsClient) ReportTrainingRun(ctx context.Context, req *crudv1.ReportTrainingRunRequest) (*crudv1.ReportTrainingRunResponse, error) {
	return &crudv1.ReportTrainingRunResponse{Success: true}, s.err
}

func (s *stubAnalyticsClient) GenerateData(ctx context.Context, req *crudv1.GenerateDataRequest) (*crudv1.GenerateDataResponse, error) {
	return s.generateDataResp, s.err
}

type errProvider struct{}

func (errProvider) GetRules(context.Context) (rules.RuleSet, error) {
	return rules.RuleSet{}, errors.New("rules unavailable")
}

func (errProvider) Reload(context.Context) error {
	return errors.New("reload failed")
}

func TestHandleEvaluateSignal_RulesProviderError(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, stubInferenceClient{}, nil, stubTrainingClient{}, stubForecastClient{}, errProvider{}, 1024, "", "")

	payload := `{"user_id":"u1","amount":100,"currency":"USD","client_transaction_id":"t1"}`
	req := httptest.NewRequest(http.MethodPost, "/evaluate/signal", strings.NewReader(payload))
	rec := httptest.NewRecorder()

	handler.handleEvaluateSignal(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d (fail-open), got %d", http.StatusOK, rec.Code)
	}

	var payloadResp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payloadResp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	t.Logf("Response body: %s", rec.Body.String())
	if rv, ok := payloadResp["rules_version"]; ok && rv != nil {
		t.Errorf("expected no rules_version (or null) in response when provider fails, got %v", rv)
	}
}

func TestHandleEvaluateSignal_MissingFeatures(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	// Setup rules that depend on velocity_24h
	ruleset := rules.RuleSet{
		Version: "v1",
		Rules: []rules.Rule{
			{
				ID:     "rule1",
				Field:  "velocity_24h",
				Op:     ">",
				Value:  5,
				Action: "reject",
				Status: rules.RuleStatusActive,
			},
		},
	}
	provider := rules.NewStaticProvider(ruleset)

	inference := stubInferenceClient{}
	// Analytics returns no features
	analytics := &stubAnalyticsClient{}

	handler := NewHandler(logger, inference, analytics, stubTrainingClient{}, stubForecastClient{}, provider, 1024, "", "")

	// user_id "u1" will result in a simulated velocity
	// hash("u1") % 1000 = 327 (approx)
	// velocity = (327 % 10) + 1 = 8
	// 8 > 5 matches rule1
	payload := `{"user_id":"u1","amount":100,"currency":"USD","client_transaction_id":"t1"}`
	req := httptest.NewRequest(http.MethodPost, "/evaluate/signal", strings.NewReader(payload))
	rec := httptest.NewRecorder()

	handler.handleEvaluateSignal(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var payloadResp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payloadResp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(payloadResp["matched_rules"].([]any)) == 0 {
		t.Errorf("expected rule1 to match via simulated features")
	}
}
