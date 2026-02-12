package httpserver

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	forecastv1 "github.com/jonkmatsumo/label-lag/go/forecast/proto/forecastv1"
	grpcclient "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestMonitoringDriftContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := stubForecastClient{
		driftResp: &forecastv1.GetDriftMonitoringResponse{
			DriftScore:    0.1,
			DriftDetected: false,
		},
	}
	handler := NewHandler(HandlerOptions{
		Logger:         logger,
		TrainingClient: stubTrainingClient{},
		ForecastClient: stub,
		RulesProvider:  rules.NewEmptyProvider(),
		MaxBodyBytes:   1024,
	})

	req := httptest.NewRequest(http.MethodGet, "/monitoring/drift?hours=24&threshold=0.25&force_refresh=false", nil)
	rec := httptest.NewRecorder()

	handler.handleMonitoringDrift(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["drift_score"] != 0.1 {
		t.Fatalf("expected drift_score 0.1, got %v", payload["drift_score"])
	}
}

func TestMonitoringDriftContractError(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := stubForecastClient{
		err: &grpcclient.RPCError{Code: codes.Unavailable, Message: "service down"},
	}
	handler := NewHandler(HandlerOptions{
		Logger:         logger,
		TrainingClient: stubTrainingClient{},
		ForecastClient: stub,
		RulesProvider:  rules.NewEmptyProvider(),
		MaxBodyBytes:   1024,
	})

	req := httptest.NewRequest(http.MethodGet, "/monitoring/drift", nil)
	rec := httptest.NewRecorder()

	handler.handleMonitoringDrift(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}
}

func TestShadowComparisonContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		shadowComparisonResp: &crudv1.GetShadowComparisonResponse{
			Metrics: &crudv1.ShadowModeMetrics{
				TotalEvaluations: 100,
			},
		},
	}
	handler := NewHandler(HandlerOptions{
		Logger:          logger,
		AnalyticsClient: stub,
		TrainingClient:  stubTrainingClient{},
		ForecastClient:  stubForecastClient{},
		RulesProvider:   rules.NewEmptyProvider(),
		MaxBodyBytes:    1024,
	})

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

func TestShadowComparisonContractError(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		err: &grpcclient.RPCError{Code: codes.Unavailable, Message: "upstream down"},
	}
	handler := NewHandler(HandlerOptions{
		Logger:          logger,
		AnalyticsClient: stub,
		TrainingClient:  stubTrainingClient{},
		ForecastClient:  stubForecastClient{},
		RulesProvider:   rules.NewEmptyProvider(),
		MaxBodyBytes:    1024,
	})

	req := httptest.NewRequest(http.MethodGet, "/metrics/shadow/comparison", nil)
	rec := httptest.NewRecorder()

	handler.handleMetricsShadowComparison(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
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
	handler := NewHandler(HandlerOptions{
		Logger:          logger,
		AnalyticsClient: stub,
		TrainingClient:  stubTrainingClient{},
		ForecastClient:  stubForecastClient{},
		RulesProvider:   rules.NewEmptyProvider(),
		MaxBodyBytes:    1024,
	})

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
	handler := NewHandler(HandlerOptions{
		Logger:          logger,
		AnalyticsClient: stub,
		TrainingClient:  stubTrainingClient{},
		ForecastClient:  stubForecastClient{},
		RulesProvider:   rules.NewEmptyProvider(),
		MaxBodyBytes:    1024,
	})

	req := httptest.NewRequest(http.MethodGet, "/backtest/results", nil)
	rec := httptest.NewRecorder()

	handler.handleBacktestResults(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
}

func TestBacktestResultsRejectsInvalidDate(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(HandlerOptions{
		Logger:          logger,
		AnalyticsClient: &stubAnalyticsClient{},
		TrainingClient:  stubTrainingClient{},
		ForecastClient:  stubForecastClient{},
		RulesProvider:   rules.NewEmptyProvider(),
		MaxBodyBytes:    1024,
	})

	req := httptest.NewRequest(http.MethodGet, "/backtest/results?start_date=not-a-date", nil)
	rec := httptest.NewRecorder()

	handler.handleBacktestResults(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
}

func TestBacktestResultsPreserveRequestIDHeader(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(HandlerOptions{
		Logger: logger,
		AnalyticsClient: &stubAnalyticsClient{
			backtestResultsResp: &crudv1.ListBacktestResultsResponse{},
		},
		TrainingClient: stubTrainingClient{},
		ForecastClient: stubForecastClient{},
		RulesProvider:  rules.NewEmptyProvider(),
		MaxBodyBytes:   1024,
	})

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
