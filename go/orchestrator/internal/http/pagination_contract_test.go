package httpserver

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
)

func TestPaginationMutualExclusivity(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{}
	handler := NewHandler(HandlerOptions{
		Logger:          logger,
		AnalyticsClient: stub,
		TrainingClient:  stubTrainingClient{},
		ForecastClient:  stubForecastClient{},
		RulesProvider:   rules.NewEmptyProvider(),
	})

	endpoints := []struct {
		name    string
		path    string
		handler func(*Handler, http.ResponseWriter, *http.Request)
	}{
		{"decisions", "/decisions", (*Handler).handleListDecisions},
		{"jobs", "/jobs", (*Handler).handleListJobs},
		{"profiles", "/dataset/profiles", (*Handler).handleListDatasetProfiles},
		{"training_runs", "/training-runs", (*Handler).handleListTrainingRuns},
	}

	for _, tc := range endpoints {
		t.Run(tc.name, func(t *testing.T) {
			// Both cursor and offset -> 400
			req := httptest.NewRequest(http.MethodGet, tc.path+"?cursor=abc&offset=10", nil)
			req = withTenantRequest(req)
			rec := httptest.NewRecorder()

			tc.handler(handler, rec, req)

			if rec.Code != http.StatusBadRequest {
				t.Errorf("%s: expected status 400 when both cursor and offset are provided, got %d", tc.name, rec.Code)
			}
		})
	}
}

func TestPaginationPrecedence(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	stub := &stubAnalyticsClient{}
	handler := NewHandler(HandlerOptions{
		Logger:          logger,
		AnalyticsClient: stub,
		TrainingClient:  stubTrainingClient{},
		ForecastClient:  stubForecastClient{},
		RulesProvider:   rules.NewEmptyProvider(),
	})

	t.Run("cursor_passthrough", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/decisions?cursor=c123&limit=25", nil)
		req = withTenantRequest(req)
		rec := httptest.NewRecorder()

		handler.handleListDecisions(rec, req)

		if stub.lastListDecisionsReq == nil {
			t.Fatal("expected request to be captured by stub")
		}
		if stub.lastListDecisionsReq.Pagination == nil || stub.lastListDecisionsReq.Pagination.Cursor != "c123" {
			t.Errorf("expected cursor c123, got %v", stub.lastListDecisionsReq.Pagination)
		}
		if stub.lastListDecisionsReq.Pagination.Limit != 25 {
			t.Errorf("expected limit 25, got %d", stub.lastListDecisionsReq.Pagination.Limit)
		}
	})
}
