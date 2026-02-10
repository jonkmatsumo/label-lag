package httpserver

import (
	"net/http"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
)

func (h *Handler) handleGetDatasetProfile(w http.ResponseWriter, r *http.Request) {
	profileID := r.PathValue("id")
	if profileID == "" {
		writeJSONError(w, http.StatusBadRequest, "profile_id required")
		return
	}

	req := &crudv1.GetDatasetSummaryRequest{
		ProfileId: profileID,
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

	req := &crudv1.ListDatasetProfilesRequest{
		Limit:  limit,
		Offset: offset,
	}

	resp, err := h.analyticsClient.ListDatasetProfiles(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, resp)
}

func (h *Handler) handleCompareDatasetProfiles(w http.ResponseWriter, r *http.Request) {
	req := &crudv1.CompareDatasetProfilesRequest{
		BaselineProfileId:  r.URL.Query().Get("baseline_id"),
		CandidateProfileId: r.URL.Query().Get("candidate_id"),
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
