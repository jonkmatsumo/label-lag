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
	grpcclient "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestAnalyticsOverviewContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		overviewResp: &crudv1.GetOverviewMetricsResponse{
			TotalRecords:            10,
			FraudRecords:            2,
			FraudRate:               0.2,
			UniqueUsers:             3,
			MinTransactionTimestamp: timestamppb.New(time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)),
			MaxTransactionTimestamp: timestamppb.New(time.Date(2025, 1, 2, 0, 0, 0, 0, time.UTC)),
			TotalAmount:             123.45,
			FraudAmount:             12.34,
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)

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
	if payload["total_records"] != float64(10) {
		t.Fatalf("expected total_records 10, got %v", payload["total_records"])
	}
	if payload["fraud_records"] != float64(2) {
		t.Fatalf("expected fraud_records 2, got %v", payload["fraud_records"])
	}
	if payload["min_transaction_timestamp"] == "" {
		t.Fatalf("expected min_transaction_timestamp to be set")
	}
}

func TestAnalyticsDailyStatsContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		dailyStatsResp: &crudv1.GetDailyStatsResponse{
			Stats: []*crudv1.DailyStat{
				{
					Date:              "2025-01-01",
					TotalTransactions: 10,
					FraudCount:        1,
					FraudRate:         0.1,
					TotalAmount:       100,
					AvgZScore:         0.2,
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)

	req := httptest.NewRequest(http.MethodGet, "/analytics/daily-stats", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsDailyStats(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	stats := payload["stats"].([]any)
	if len(stats) != 1 {
		t.Fatalf("expected 1 stat, got %v", payload["stats"])
	}
}

func TestAnalyticsTransactionsContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		transactionDetailsResp: &crudv1.GetTransactionDetailsResponse{
			Transactions: []*crudv1.TransactionDetail{
				{
					RecordId:  "rec-1",
					UserId:    "user-1",
					CreatedAt: timestamppb.New(time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)),
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)

	req := httptest.NewRequest(http.MethodGet, "/analytics/transactions", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsTransactions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	transactions := payload["transactions"].([]any)
	if len(transactions) != 1 {
		t.Fatalf("expected 1 transaction, got %v", payload["transactions"])
	}
}

func TestAnalyticsRecentAlertsContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		recentAlertsResp: &crudv1.GetRecentAlertsResponse{
			Alerts: []*crudv1.Alert{
				{
					RecordId:  "rec-1",
					UserId:    "user-1",
					CreatedAt: timestamppb.New(time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)),
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)

	req := httptest.NewRequest(http.MethodGet, "/analytics/recent-alerts", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsRecentAlerts(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	alerts := payload["alerts"].([]any)
	if len(alerts) != 1 {
		t.Fatalf("expected 1 alert, got %v", payload["alerts"])
	}
}

func TestAnalyticsFingerprintContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		fingerprintResp: &crudv1.GetDatasetFingerprintResponse{
			GeneratedRecords: &crudv1.TableFingerprint{Count: 10},
			FeatureSnapshots: &crudv1.TableFingerprint{Count: 5},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)

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
	if payload["generated_records"] == nil || payload["feature_snapshots"] == nil {
		t.Fatalf("expected fingerprint sections to be present")
	}
}

func TestAnalyticsFeatureSampleContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		featureSampleResp: &crudv1.GetFeatureSampleResponse{
			Samples: []*crudv1.FeatureSample{
				{
					RecordId: "rec-1",
				},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)

	req := httptest.NewRequest(http.MethodGet, "/analytics/feature-sample", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsFeatureSample(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
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

func TestAnalyticsSchemaContract(t *testing.T) {
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
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)

	req := httptest.NewRequest(http.MethodGet, "/analytics/schema", nil)
	rec := httptest.NewRecorder()

	handler.handleAnalyticsSchema(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	columns := payload["columns"].([]any)
	if len(columns) != 1 {
		t.Fatalf("expected 1 column, got %v", payload["columns"])
	}
}

func TestAnalyticsHandlersMapDownstreamErrors(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	paths := []struct {
		name    string
		path    string
		handler func(*Handler, http.ResponseWriter, *http.Request)
	}{
		{"overview", "/analytics/overview", (*Handler).handleAnalyticsOverview},
		{"daily-stats", "/analytics/daily-stats", (*Handler).handleAnalyticsDailyStats},
		{"transactions", "/analytics/transactions", (*Handler).handleAnalyticsTransactions},
		{"recent-alerts", "/analytics/recent-alerts", (*Handler).handleAnalyticsRecentAlerts},
		{"fingerprint", "/analytics/fingerprint", (*Handler).handleAnalyticsFingerprint},
		{"feature-sample", "/analytics/feature-sample", (*Handler).handleAnalyticsFeatureSample},
		{"schema", "/analytics/schema", (*Handler).handleAnalyticsSchema},
	}

	for _, tc := range paths {
		t.Run(tc.name, func(t *testing.T) {
			stub := &stubAnalyticsClient{
				err: &grpcclient.RPCError{Code: codes.InvalidArgument, Message: "bad request"},
			}
			handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)
			req := httptest.NewRequest(http.MethodGet, tc.path, nil)
			rec := httptest.NewRecorder()

			tc.handler(handler, rec, req)

			if rec.Code != http.StatusBadRequest {
				t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
			}
		})
	}
}

func TestAnalyticsHandlersPreserveRequestIDHeader(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, &stubAnalyticsClient{
		overviewResp:           &crudv1.GetOverviewMetricsResponse{},
		dailyStatsResp:         &crudv1.GetDailyStatsResponse{},
		transactionDetailsResp: &crudv1.GetTransactionDetailsResponse{},
		recentAlertsResp:       &crudv1.GetRecentAlertsResponse{},
		fingerprintResp:        &crudv1.GetDatasetFingerprintResponse{},
		featureSampleResp:      &crudv1.GetFeatureSampleResponse{},
		schemaSummaryResp:      &crudv1.GetSchemaSummaryResponse{},
	}, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "", false, false)

	mux := http.NewServeMux()
	handler.Register(mux)
	server := requestIDMiddleware(logger, mux)

	paths := []string{
		"/analytics/overview",
		"/analytics/daily-stats",
		"/analytics/transactions",
		"/analytics/recent-alerts",
		"/analytics/fingerprint",
		"/analytics/feature-sample",
		"/analytics/schema",
	}

	for _, path := range paths {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		req.Header.Set("X-Request-Id", "req-123")
		req = req.WithContext(requestid.WithRequestID(req.Context(), "req-123"))
		rec := httptest.NewRecorder()

		server.ServeHTTP(rec, req)

		if rec.Header().Get("X-Request-Id") != "req-123" {
			t.Fatalf("expected request id header to be preserved for %s", path)
		}
	}
}
