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
	resp              *crudv1.SearchTransactionsResponse
	dailyStatsResp    *crudv1.GetDailyStatsResponse
	overviewResp      *crudv1.GetOverviewMetricsResponse
	err               error
	lastReq           *crudv1.SearchTransactionsRequest
	lastDailyStatsReq *crudv1.GetDailyStatsRequest
	lastOverviewReq   *crudv1.GetOverviewMetricsRequest
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

type errProvider struct{}

func (errProvider) GetRules(context.Context) (rules.RuleSet, error) {
	return rules.RuleSet{}, errors.New("rules unavailable")
}
