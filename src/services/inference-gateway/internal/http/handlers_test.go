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
	inferencev1 "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/grpc/inferencev1/inference/v1"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
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
					Velocity24H:             0,
					AmountToAvgRatio30D:     1.1,
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
	resp    *crudv1.SearchTransactionsResponse
	err     error
	lastReq *crudv1.SearchTransactionsRequest
}

func (s *stubAnalyticsClient) SearchTransactions(ctx context.Context, req *crudv1.SearchTransactionsRequest) (*crudv1.SearchTransactionsResponse, error) {
	s.lastReq = req
	return s.resp, s.err
}

type errProvider struct{}

func (errProvider) GetRules(context.Context) (rules.RuleSet, error) {
	return rules.RuleSet{}, errors.New("rules unavailable")
}
