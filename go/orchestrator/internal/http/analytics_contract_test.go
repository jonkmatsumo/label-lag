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
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

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
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

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
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

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
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

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
		{"feature-sample", "/analytics/feature-sample", (*Handler).handleAnalyticsFeatureSample},
	}

	for _, tc := range paths {
		t.Run(tc.name, func(t *testing.T) {
			stub := &stubAnalyticsClient{
				err: &grpcclient.RPCError{Code: codes.InvalidArgument, Message: "bad request"},
			}
			handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")
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
		featureSampleResp:      &crudv1.GetFeatureSampleResponse{},
	}, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	mux := http.NewServeMux()
	handler.Register(mux)
	server := requestIDMiddleware(logger, mux)

	paths := []string{
		"/analytics/overview",
		"/analytics/daily-stats",
		"/analytics/transactions",
		"/analytics/recent-alerts",
		"/analytics/feature-sample",
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

func TestAnalyticsDecisionsContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		listDecisionsResp: &crudv1.ListDecisionsResponse{
			Decisions: []*crudv1.DecisionSummary{
				{
					RequestId:      "req-1",
					UserId:         "user-1",
					FinalScore:     45,
					Decision:       "APPROVE",
					DecisionReason: "score_below_threshold",
					Thresholds: &crudv1.DecisionThresholds{
						ApproveMax: 50,
						ReviewMax:  80,
					},
					CreatedAt: timestamppb.Now(),
				},
			},
			Total: 1,
		},
		getDecisionResp: &crudv1.GetDecisionResponse{
			Decision: &crudv1.InferenceEvent{
				RequestId:      "req-1",
				FinalScore:     45,
				Decision:       "APPROVE",
				DecisionReason: "score_below_threshold",
				Thresholds: &crudv1.DecisionThresholds{
					ApproveMax: 50,
					ReviewMax:  80,
				},
				Timestamp: timestamppb.Now(),
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	t.Run("list_decisions", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/decisions?limit=10", nil)
		rec := httptest.NewRecorder()
		handler.handleListDecisions(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
		}
		var payload map[string]any
		err := json.Unmarshal(rec.Body.Bytes(), &payload)
		if err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}
		decisions := payload["decisions"].([]any)
		d := decisions[0].(map[string]any)
		if d["decision_reason"] != "score_below_threshold" {
			t.Fatalf("expected decision_reason score_below_threshold, got %v", d["decision_reason"])
		}
		thresholds := d["thresholds"].(map[string]any)
		if thresholds["approve_max"] != float64(50) {
			t.Fatalf("expected approve_max 50, got %v", thresholds["approve_max"])
		}
	})

	t.Run("get_decision", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/decisions/req-1", nil)
		// No req.SetPathValue in standard net/http/httptest, we depend on manual setting if using standard mux?
		// Handler uses r.PathValue("request_id")
		req.SetPathValue("request_id", "req-1")
		rec := httptest.NewRecorder()
		handler.handleGetDecision(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
		}
		var payload map[string]any
		err := json.Unmarshal(rec.Body.Bytes(), &payload)
		if err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}
		decision := payload["decision"].(map[string]any)
		if decision["decision_reason"] != "score_below_threshold" {
			t.Fatalf("expected decision_reason score_below_threshold, got %v", decision["decision_reason"])
		}
	})
}

func TestAnalyticsJobsContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		listJobsResp: &crudv1.ListJobsResponse{
			Jobs: []*crudv1.Job{
				{JobId: "job-1", JobType: "training", Status: "completed"},
			},
			Total: 1,
		},
		getJobSummaryResp: &crudv1.GetJobSummaryResponse{
			Summaries: []*crudv1.JobSummaryBucket{
				{BucketTime: timestamppb.Now(), TotalJobs: 5, CompletedJobs: 4, FailedJobs: 1},
			},
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	t.Run("list_jobs", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/jobs", nil)
		rec := httptest.NewRecorder()
		handler.handleListJobs(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
		}
		var payload map[string]any
		json.Unmarshal(rec.Body.Bytes(), &payload)
		jobs := payload["jobs"].([]any)
		if len(jobs) != 1 {
			t.Fatalf("expected 1 job, got %v", len(jobs))
		}
	})

	t.Run("get_job_summary", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/jobs/summary", nil)
		rec := httptest.NewRecorder()
		handler.handleGetJobSummary(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
		}
		var payload map[string]any
		json.Unmarshal(rec.Body.Bytes(), &payload)
		summaries := payload["summaries"].([]any)
		if len(summaries) != 1 {
			t.Fatalf("expected 1 summary, got %v", len(summaries))
		}
	})
}

func TestAnalyticsTrainingRunsContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		listTrainingRunsResp: &crudv1.ListTrainingRunsResponse{
			Runs: []*crudv1.TrainingRun{
				{RunId: "run-1", ModelName: "xgboost", Status: "completed"},
			},
			Total: 1,
		},
		listModelVersionsResp: &crudv1.ListModelVersionsResponse{
			Versions: []*crudv1.TrainingRun{
				{RunId: "run-1", ModelName: "xgboost", Status: "completed"},
			},
			Total: 1,
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	t.Run("list_training_runs", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/training-runs", nil)
		rec := httptest.NewRecorder()
		handler.handleListTrainingRuns(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
		}
		var payload map[string]any
		json.Unmarshal(rec.Body.Bytes(), &payload)
		runs := payload["runs"].([]any)
		if len(runs) != 1 {
			t.Fatalf("expected 1 run, got %v", len(runs))
		}
	})

	t.Run("list_model_versions", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/models/versions?model_name=xgboost", nil)
		rec := httptest.NewRecorder()
		handler.handleListModelVersions(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
		}
		var payload map[string]any
		json.Unmarshal(rec.Body.Bytes(), &payload)
		versions := payload["versions"].([]any)
		if len(versions) != 1 {
			t.Fatalf("expected 1 version, got %v", len(versions))
		}
	})
}

func TestAnalyticsDatasetProfilesContract(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{
		listDatasetProfilesResp: &crudv1.ListDatasetProfilesResponse{
			Profiles: []*crudv1.DatasetProfile{
				{ProfileId: "prof-1", RecordCount: 1000},
			},
			Total: 1,
		},
		getLatestDatasetProfileResp: &crudv1.GetLatestDatasetProfileResponse{
			ProfileId:   "prof-1",
			RecordCount: 1000,
			ComputedAt:  timestamppb.Now(),
		},
	}
	handler := NewHandler(logger, nil, stub, stubTrainingClient{}, stubForecastClient{}, rules.NewEmptyProvider(), 1024, "", "")

	t.Run("list_dataset_profiles", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/dataset/profiles", nil)
		rec := httptest.NewRecorder()
		handler.handleListDatasetProfiles(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
		}
		var payload map[string]any
		json.Unmarshal(rec.Body.Bytes(), &payload)
		profiles := payload["profiles"].([]any)
		if len(profiles) != 1 {
			t.Fatalf("expected 1 profile, got %v", len(profiles))
		}
	})

	t.Run("get_latest_dataset_profile", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/dataset/latest", nil)
		rec := httptest.NewRecorder()
		handler.handleGetLatestDatasetProfile(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
		}
		var payload map[string]any
		json.Unmarshal(rec.Body.Bytes(), &payload)
		if payload["profile_id"] != "prof-1" {
			t.Fatalf("expected profile_id prof-1, got %v", payload["profile_id"])
		}
	})
}
