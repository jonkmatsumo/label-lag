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

	crudv1 "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	grpcclient "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/grpc"
	inferencev1 "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/grpc/inferencev1/inference/v1"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestHandleEvaluateSignal_RejectsLargeBody(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, rules.NewEmptyProvider(), 32)

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
	handler := NewHandler(logger, nil, nil, rules.NewEmptyProvider(), 1024)

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
	handler := NewHandler(logger, stubInferenceClient{}, nil, rules.NewEmptyProvider(), 1024)

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
	handler := NewHandler(logger, stubInferenceClient{readyErr: errors.New("not ready")}, nil, errProvider{}, 1024)

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()

	handler.handleReady(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}
}

func TestHandleNotImplemented(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, rules.NewEmptyProvider(), 1024)

	req := httptest.NewRequest(http.MethodGet, "/analytics/overview", nil)
	req = req.WithContext(requestid.WithRequestID(req.Context(), "test-req-id"))
	rec := httptest.NewRecorder()

	handler.handleNotImplemented(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("expected status %d, got %d", http.StatusNotImplemented, rec.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["error"] != "not_implemented" {
		t.Fatalf("expected not_implemented error, got %v", payload["error"])
	}
	if payload["request_id"] != "test-req-id" {
		t.Fatalf("expected request_id test-req-id, got %v", payload["request_id"])
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
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

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
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

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
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

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
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

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
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

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
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

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

func TestHandleAnalyticsFingerprint(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		fingerprintResp: &crudv1.GetDatasetFingerprintResponse{
			GeneratedRecords: &crudv1.TableFingerprint{
				Count:        10,
				MaxCreatedAt: timestamppb.New(time.Date(2025, 2, 1, 0, 0, 0, 0, time.UTC)),
				MaxTimestamp: timestamppb.New(time.Date(2025, 2, 1, 1, 0, 0, 0, time.UTC)),
				MaxId:        99,
			},
			FeatureSnapshots: &crudv1.TableFingerprint{
				Count:        5,
				MaxCreatedAt: timestamppb.New(time.Date(2025, 2, 2, 0, 0, 0, 0, time.UTC)),
				MaxTimestamp: timestamppb.New(time.Date(2025, 2, 2, 1, 0, 0, 0, time.UTC)),
				MaxId:        42,
			},
		},
	}
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

	req := httptest.NewRequest(http.MethodGet, "/analytics/fingerprint", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsFingerprint(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	generated := payload["generated_records"].(map[string]any)
	if generated["count"] != float64(10) {
		t.Fatalf("expected generated_records count 10, got %v", generated["count"])
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
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

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

func TestHandleAnalyticsSchema(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		schemaSummaryResp: &crudv1.GetSchemaSummaryResponse{
			Columns: []*crudv1.ColumnInfo{
				{
					TableName:       "generated_records",
					ColumnName:      "record_id",
					DataType:        "text",
					IsNullable:      "NO",
					OrdinalPosition: 1,
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

	req := httptest.NewRequest(http.MethodGet, "/analytics/schema?table_names=generated_records&table_names=feature_snapshots", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsSchema(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if stub.lastSchemaSummaryReq == nil || len(stub.lastSchemaSummaryReq.GetTableNames()) != 2 {
		t.Fatalf("expected 2 table_names, got %v", stub.lastSchemaSummaryReq)
	}
}

func TestHandleMonitoringDriftProxies(t *testing.T) {
	var gotRequestID string
	var gotPath string
	var gotQuery string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRequestID = r.Header.Get("X-Request-Id")
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer upstream.Close()

	t.Setenv("INFERENCE_GATEWAY_API_URL", upstream.URL)

	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, rules.NewEmptyProvider(), 1024)

	req := httptest.NewRequest(http.MethodGet, "/monitoring/drift?hours=24&threshold=0.25&force_refresh=false", nil)
	req = req.WithContext(requestid.WithRequestID(req.Context(), "req-1"))
	rec := httptest.NewRecorder()

	handler.handleMonitoringDrift(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if gotRequestID != "req-1" {
		t.Fatalf("expected request id req-1, got %v", gotRequestID)
	}
	if gotPath != "/monitoring/drift" {
		t.Fatalf("expected path /monitoring/drift, got %v", gotPath)
	}
	if gotQuery != "hours=24&threshold=0.25&force_refresh=false" {
		t.Fatalf("expected query forwarded, got %v", gotQuery)
	}
}

func TestHandleMetricsShadowComparisonProxies(t *testing.T) {
	var gotRequestID string
	var gotPath string
	var gotQuery string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRequestID = r.Header.Get("X-Request-Id")
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"total_requests":0}`))
	}))
	defer upstream.Close()

	t.Setenv("INFERENCE_GATEWAY_API_URL", upstream.URL)

	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, rules.NewEmptyProvider(), 1024)

	req := httptest.NewRequest(http.MethodGet, "/metrics/shadow/comparison?start_date=2025-01-01&end_date=2025-01-31&rule_ids=r1,r2", nil)
	req = req.WithContext(requestid.WithRequestID(req.Context(), "req-2"))
	rec := httptest.NewRecorder()

	handler.handleMetricsShadowComparison(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if gotRequestID != "req-2" {
		t.Fatalf("expected request id req-2, got %v", gotRequestID)
	}
	if gotPath != "/metrics/shadow/comparison" {
		t.Fatalf("expected path /metrics/shadow/comparison, got %v", gotPath)
	}
	if gotQuery != "start_date=2025-01-01&end_date=2025-01-31&rule_ids=r1,r2" {
		t.Fatalf("expected query forwarded, got %v", gotQuery)
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
	handler := NewHandler(logger, nil, stub, rules.NewEmptyProvider(), 1024)

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
	return &inferencev1.ScoreResponse{}, nil
}

type stubAnalyticsClient struct {
	resp                      *crudv1.SearchTransactionsResponse
	dailyStatsResp            *crudv1.GetDailyStatsResponse
	overviewResp              *crudv1.GetOverviewMetricsResponse
	transactionDetailsResp    *crudv1.GetTransactionDetailsResponse
	recentAlertsResp          *crudv1.GetRecentAlertsResponse
	fingerprintResp           *crudv1.GetDatasetFingerprintResponse
	featureSampleResp         *crudv1.GetFeatureSampleResponse
	schemaSummaryResp         *crudv1.GetSchemaSummaryResponse
	backtestResultsResp       *crudv1.ListBacktestResultsResponse
	err                       error
	lastReq                   *crudv1.SearchTransactionsRequest
	lastDailyStatsReq         *crudv1.GetDailyStatsRequest
	lastOverviewReq           *crudv1.GetOverviewMetricsRequest
	lastTransactionDetailsReq *crudv1.GetTransactionDetailsRequest
	lastRecentAlertsReq       *crudv1.GetRecentAlertsRequest
	lastFingerprintReq        *crudv1.GetDatasetFingerprintRequest
	lastFeatureSampleReq      *crudv1.GetFeatureSampleRequest
	lastSchemaSummaryReq      *crudv1.GetSchemaSummaryRequest
	lastBacktestResultsReq    *crudv1.ListBacktestResultsRequest
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

func (s *stubAnalyticsClient) GetDatasetFingerprint(ctx context.Context, req *crudv1.GetDatasetFingerprintRequest) (*crudv1.GetDatasetFingerprintResponse, error) {
	s.lastFingerprintReq = req
	return s.fingerprintResp, s.err
}

func (s *stubAnalyticsClient) GetFeatureSample(ctx context.Context, req *crudv1.GetFeatureSampleRequest) (*crudv1.GetFeatureSampleResponse, error) {
	s.lastFeatureSampleReq = req
	return s.featureSampleResp, s.err
}

func (s *stubAnalyticsClient) GetSchemaSummary(ctx context.Context, req *crudv1.GetSchemaSummaryRequest) (*crudv1.GetSchemaSummaryResponse, error) {
	s.lastSchemaSummaryReq = req
	return s.schemaSummaryResp, s.err
}

func (s *stubAnalyticsClient) ListBacktestResults(ctx context.Context, req *crudv1.ListBacktestResultsRequest) (*crudv1.ListBacktestResultsResponse, error) {
	s.lastBacktestResultsReq = req
	return s.backtestResultsResp, s.err
}

type errProvider struct{}

func (errProvider) GetRules(context.Context) (rules.RuleSet, error) {
	return rules.RuleSet{}, errors.New("rules unavailable")
}
