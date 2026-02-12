package httpserver

import (
	"net/http"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/tenant"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// handleListJobs lists jobs with filters.
// Query params: limit (1-100), offset, job_type, status, start_date (RFC3339), end_date (RFC3339).
func (h *Handler) handleListJobs(w http.ResponseWriter, r *http.Request) {
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

	req := &crudv1.ListJobsRequest{
		Limit:   limit,
		Offset:  offset,
		JobType: r.URL.Query().Get("job_type"),
		Status:  r.URL.Query().Get("status"),
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

	if req.StartDate != nil && req.EndDate != nil && req.StartDate.AsTime().After(req.EndDate.AsTime()) {
		writeJSONError(w, http.StatusBadRequest, "start_date must be <= end_date")
		return
	}

	resp, err := h.analyticsClient.ListJobs(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleGetJob(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	if jobID == "" {
		writeJSONError(w, http.StatusBadRequest, "job_id required")
		return
	}

	resp, err := h.analyticsClient.GetJob(r.Context(), &crudv1.GetJobRequest{
		JobId:    jobID,
		TenantId: tenant.FromContext(r.Context()),
	})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleGetJobEvents(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	if jobID == "" {
		writeJSONError(w, http.StatusBadRequest, "job_id required")
		return
	}

	limit, err := parseIntQuery(r, "limit", 100, 1, 1000)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	offset, err := parseIntQuery(r, "offset", 0, 0, 10000)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	resp, err := h.analyticsClient.GetJobEvents(r.Context(), &crudv1.GetJobEventsRequest{
		JobId:    jobID,
		Limit:    limit,
		Offset:   offset,
		TenantId: tenant.FromContext(r.Context()),
	})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleGetJobSummary(w http.ResponseWriter, r *http.Request) {
	req := &crudv1.GetJobSummaryRequest{
		TenantId: tenant.FromContext(r.Context()),
	}

	if startStr := r.URL.Query().Get("start_time"); startStr != "" {
		if t, err := time.Parse(time.RFC3339, startStr); err == nil {
			req.StartTime = timestamppb.New(t)
		} else {
			writeJSONError(w, http.StatusBadRequest, "invalid start_time format (RFC3339 required)")
			return
		}
	}
	if endStr := r.URL.Query().Get("end_time"); endStr != "" {
		if t, err := time.Parse(time.RFC3339, endStr); err == nil {
			req.EndTime = timestamppb.New(t)
		} else {
			writeJSONError(w, http.StatusBadRequest, "invalid end_time format (RFC3339 required)")
			return
		}
	}

	resp, err := h.analyticsClient.GetJobSummary(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}
