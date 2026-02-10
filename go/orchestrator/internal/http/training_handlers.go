package httpserver

import (
	"net/http"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func (h *Handler) handleListTrainingRuns(w http.ResponseWriter, r *http.Request) {
	limit, err := parseIntQuery(r, "limit", 20, 1, 100)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	offset, err := parseIntQuery(r, "offset", 0, 0, 10000)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	req := &crudv1.ListTrainingRunsRequest{
		ModelName: r.URL.Query().Get("model_name"),
		Status:    r.URL.Query().Get("status"),
		Limit:     limit,
		Offset:    offset,
	}

	resp, err := h.analyticsClient.ListTrainingRuns(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleGetTrainingRun(w http.ResponseWriter, r *http.Request) {
	runID := r.PathValue("id")
	if runID == "" {
		writeJSONError(w, http.StatusBadRequest, "run_id required")
		return
	}

	resp, err := h.analyticsClient.GetTrainingRun(r.Context(), &crudv1.GetTrainingRunRequest{RunId: runID})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleGetMetricSeries(w http.ResponseWriter, r *http.Request) {
	req := &crudv1.GetMetricSeriesRequest{
		ModelName:  r.URL.Query().Get("model_name"),
		MetricName: r.URL.Query().Get("metric_name"),
	}

	if startStr := r.URL.Query().Get("start_date"); startStr != "" {
		if t, err := time.Parse(time.RFC3339, startStr); err == nil {
			req.StartDate = timestamppb.New(t)
		}
	}
	if endStr := r.URL.Query().Get("end_date"); endStr != "" {
		if t, err := time.Parse(time.RFC3339, endStr); err == nil {
			req.EndDate = timestamppb.New(t)
		}
	}

	if req.ModelName == "" || req.MetricName == "" {
		writeJSONError(w, http.StatusBadRequest, "model_name and metric_name required")
		return
	}

	resp, err := h.analyticsClient.GetMetricSeries(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}
