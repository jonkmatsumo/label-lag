package httpserver

import (
	"net/http"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	commonv1 "github.com/jonkmatsumo/label-lag/go/common/proto/v1"
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

	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}

	if err := h.validatePaginationParams(w, r); err != nil {
		return
	}
	cursor := r.URL.Query().Get("cursor")

	req := &crudv1.ListTrainingRunsRequest{
		ModelName: r.URL.Query().Get("model_name"),
		Status:    r.URL.Query().Get("status"),
		Limit:     limit,
		Offset:    offset,
		TenantId:  tenantID,
	}

	if cursor != "" {
		req.Pagination = &commonv1.CursorPageRequest{
			Cursor: cursor,
			Limit:  limit,
		}
	}

	if startStr := r.URL.Query().Get("start_date"); startStr != "" {
		if t, err := time.Parse(time.RFC3339, startStr); err == nil {
			req.StartDate = timestamppb.New(t)
		} else {
			writeJSONError(w, http.StatusBadRequest, "invalid start_date format (RFC3339 required)")
			return
		}
	}
	if endStr := r.URL.Query().Get("end_date"); endStr != "" {
		if t, err := time.Parse(time.RFC3339, endStr); err == nil {
			req.EndDate = timestamppb.New(t)
		} else {
			writeJSONError(w, http.StatusBadRequest, "invalid end_date format (RFC3339 required)")
			return
		}
	}

	resp, err := h.analyticsClient.ListTrainingRuns(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleListModelVersions(w http.ResponseWriter, r *http.Request) {
	modelName := r.URL.Query().Get("model_name")
	if modelName == "" {
		writeJSONError(w, http.StatusBadRequest, "model_name required")
		return
	}

	limit, err := parseIntQuery(r, "limit", 50, 1, 100)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	offset, err := parseIntQuery(r, "offset", 0, 0, 10000)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}

	if err := h.validatePaginationParams(w, r); err != nil {
		return
	}
	cursor := r.URL.Query().Get("cursor")

	req := &crudv1.ListModelVersionsRequest{
		ModelName: modelName,
		Limit:     limit,
		Offset:    offset,
		TenantId:  tenantID,
	}

	if cursor != "" {
		req.Pagination = &commonv1.CursorPageRequest{
			Cursor: cursor,
			Limit:  limit,
		}
	}

	resp, err := h.analyticsClient.ListModelVersions(r.Context(), req)
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

	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}

	resp, err := h.analyticsClient.GetTrainingRun(r.Context(), &crudv1.GetTrainingRunRequest{
		RunId:    runID,
		TenantId: tenantID,
	})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleGetMetricSeries(w http.ResponseWriter, r *http.Request) {
	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}

	req := &crudv1.GetMetricSeriesRequest{
		ModelName:  r.URL.Query().Get("model_name"),
		MetricName: r.URL.Query().Get("metric_name"),
		TenantId:   tenantID,
	}

	if startStr := r.URL.Query().Get("start_date"); startStr != "" {
		t, err := time.Parse(time.RFC3339, startStr)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid start_date format (RFC3339 required)")
			return
		}
		req.StartDate = timestamppb.New(t)
	}
	if endStr := r.URL.Query().Get("end_date"); endStr != "" {
		t, err := time.Parse(time.RFC3339, endStr)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid end_date format (RFC3339 required)")
			return
		}
		req.EndDate = timestamppb.New(t)
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
