package httpserver

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	grpcclient "github.com/jonkmatsumo/label-lag/go/inference/internal/grpc"
	"github.com/jonkmatsumo/label-lag/go/inference/internal/requestid"
	"github.com/jonkmatsumo/label-lag/go/inference/internal/rules"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestMonitoringDriftContract(t *testing.T) {
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
	handler := NewHandler(logger, nil, nil, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/monitoring/drift?hours=24&threshold=0.25&force_refresh=false", nil)
	req = req.WithContext(requestid.WithRequestID(req.Context(), "req-55"))
	rec := httptest.NewRecorder()

	handler.handleMonitoringDrift(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if gotRequestID != "req-55" {
		t.Fatalf("expected request id req-55, got %v", gotRequestID)
	}
	if gotPath != "/monitoring/drift" {
		t.Fatalf("expected path /monitoring/drift, got %v", gotPath)
	}
	if gotQuery != "hours=24&threshold=0.25&force_refresh=false" {
		t.Fatalf("expected query forwarded, got %v", gotQuery)
	}
}

func TestMonitoringDriftContractError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"detail":"boom"}`))
	}))
	defer upstream.Close()

	t.Setenv("INFERENCE_GATEWAY_API_URL", upstream.URL)

	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/monitoring/drift", nil)
	rec := httptest.NewRecorder()

	handler.handleMonitoringDrift(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected status %d, got %d", http.StatusInternalServerError, rec.Code)
	}
	if !bytes.Contains(rec.Body.Bytes(), []byte(`"detail":"boom"`)) {
		t.Fatalf("expected response body to pass through, got %s", rec.Body.String())
	}
}

func TestShadowComparisonContract(t *testing.T) {
	var gotRequestID string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRequestID = r.Header.Get("X-Request-Id")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"total_requests":0}`))
	}))
	defer upstream.Close()

	t.Setenv("INFERENCE_GATEWAY_API_URL", upstream.URL)

	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/metrics/shadow/comparison?start_date=2025-01-01&end_date=2025-01-31", nil)
	req = req.WithContext(requestid.WithRequestID(req.Context(), "req-66"))
	rec := httptest.NewRecorder()

	handler.handleMetricsShadowComparison(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if gotRequestID != "req-66" {
		t.Fatalf("expected request id req-66, got %v", gotRequestID)
	}
}

func TestShadowComparisonContractError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"detail":"upstream down"}`))
	}))
	defer upstream.Close()

	t.Setenv("INFERENCE_GATEWAY_API_URL", upstream.URL)

	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, nil, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/metrics/shadow/comparison?start_date=2025-01-01&end_date=2025-01-31", nil)
	rec := httptest.NewRecorder()

	handler.handleMetricsShadowComparison(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected status %d, got %d", http.StatusBadGateway, rec.Code)
	}
}

func TestBacktestResultsContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		backtestResultsResp: &crudv1.ListBacktestResultsResponse{
			Results: []*crudv1.BacktestResult{
				{
					JobId:          "job-1",
					RulesetVersion: "v1",
					StartDate:      timestamppb.New(time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)),
					EndDate:        timestamppb.New(time.Date(2025, 1, 2, 0, 0, 0, 0, time.UTC)),
					Metrics:        &crudv1.BacktestMetrics{},
					CompletedAt:    timestamppb.New(time.Date(2025, 1, 2, 1, 0, 0, 0, time.UTC)),
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/backtest/results?limit=1", nil)
	rec := httptest.NewRecorder()

	handler.handleBacktestResults(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
}

func TestBacktestResultsContractError(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		err: &grpcclient.RPCError{Code: codes.InvalidArgument, Message: "bad request"},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/backtest/results", nil)
	rec := httptest.NewRecorder()

	handler.handleBacktestResults(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
}

func TestBacktestResultsRejectsInvalidDate(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, &stubAnalyticsClient{}, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	req := httptest.NewRequest(http.MethodGet, "/backtest/results?start_date=not-a-date", nil)
	rec := httptest.NewRecorder()

	handler.handleBacktestResults(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
}

func TestBacktestResultsPreserveRequestIDHeader(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, &stubAnalyticsClient{
		backtestResultsResp: &crudv1.ListBacktestResultsResponse{},
	}, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	mux := http.NewServeMux()
	handler.Register(mux)
	server := requestIDMiddleware(logger, mux)

	req := httptest.NewRequest(http.MethodGet, "/backtest/results", nil)
	req.Header.Set("X-Request-Id", "req-77")
	req = req.WithContext(requestid.WithRequestID(req.Context(), "req-77"))
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Header().Get("X-Request-Id") != "req-77" {
		t.Fatalf("expected request id header to be preserved")
	}
}
