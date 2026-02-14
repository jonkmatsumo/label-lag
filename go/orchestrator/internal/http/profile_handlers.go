package httpserver

import (
	"net/http"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	commonv1 "github.com/jonkmatsumo/label-lag/go/common/proto/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func (h *Handler) handleGetDatasetProfile(w http.ResponseWriter, r *http.Request) {
	profileID := r.PathValue("id")
	if profileID == "" {
		profileID = r.URL.Query().Get("profile_id")
	}
	if profileID == "" {
		profileID = "latest"
	}

	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}

	req := &crudv1.GetDatasetSummaryRequest{
		ProfileId: profileID,
		TenantId:  tenantID,
	}

	resp, err := h.analyticsClient.GetDatasetSummary(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleListDatasetProfiles(w http.ResponseWriter, r *http.Request) {
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

	cursor := r.URL.Query().Get("cursor")
	if cursor != "" && r.URL.Query().Get("offset") != "" {
		writeJSONError(w, http.StatusBadRequest, "cannot provide both cursor and offset")
		return
	}

	req := &crudv1.ListDatasetProfilesRequest{
		Limit:    limit,
		Offset:   offset,
		TenantId: tenantID,
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

	resp, err := h.analyticsClient.ListDatasetProfiles(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleGetLatestDatasetProfile(w http.ResponseWriter, r *http.Request) {
	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}

	resp, err := h.analyticsClient.GetLatestDatasetProfile(r.Context(), &crudv1.GetLatestDatasetProfileRequest{
		TenantId: tenantID,
	})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleCompareDatasetProfiles(w http.ResponseWriter, r *http.Request) {
	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}

	req := &crudv1.CompareDatasetProfilesRequest{
		BaselineProfileId:  r.URL.Query().Get("baseline_id"),
		CandidateProfileId: r.URL.Query().Get("candidate_id"),
		TenantId:           tenantID,
	}

	if req.BaselineProfileId == "" || req.CandidateProfileId == "" {
		writeJSONError(w, http.StatusBadRequest, "baseline_id and candidate_id required")
		return
	}

	resp, err := h.analyticsClient.CompareDatasetProfiles(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}
