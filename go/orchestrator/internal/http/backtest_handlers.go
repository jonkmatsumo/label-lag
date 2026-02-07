package httpserver

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// TODO(phase4): Map core backtest read/compare routes.
// - GET /backtest/results
// - POST /backtest/compare

type backtestCompareRequest struct {
	BaselineJobID  string `json:"baseline_job_id"`
	CandidateJobID string `json:"candidate_job_id"`
}

type backtestCompareResponse struct {
	Baseline  backtestResultResponse   `json:"baseline"`
	Candidate backtestResultResponse   `json:"candidate"`
	Delta     backtestMetricsDeltaJSON `json:"delta"`
}

type backtestMetricsDeltaJSON struct {
	MatchRateDelta    float64 `json:"match_rate_delta"`
	ScoreMeanDelta    float64 `json:"score_mean_delta"`
	ScoreStdDelta     float64 `json:"score_std_delta"`
	RejectedRateDelta float64 `json:"rejected_rate_delta"`
	TotalRecordsDelta int64   `json:"total_records_delta"`
	MatchedCountDelta int64   `json:"matched_count_delta"`
}

func (h *Handler) handleBacktestCompare(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	var req backtestCompareRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid JSON payload")
		return
	}

	if req.BaselineJobID == "" || req.CandidateJobID == "" {
		writeJSONError(w, http.StatusBadRequest, "baseline_job_id and candidate_job_id are required")
		return
	}

	resp, err := h.analyticsClient.CompareBacktests(r.Context(), &crudv1.CompareBacktestsRequest{
		BaselineJobId:  req.BaselineJobID,
		CandidateJobId: req.CandidateJobID,
	})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	writeAnalyticsJSON(w, backtestCompareResponse{
		Baseline:  mapBacktestResult(resp.Baseline),
		Candidate: mapBacktestResult(resp.Candidate),
		Delta: backtestMetricsDeltaJSON{
			MatchRateDelta:    resp.Delta.GetMatchRateDelta(),
			ScoreMeanDelta:    resp.Delta.GetScoreMeanDelta(),
			ScoreStdDelta:     resp.Delta.GetScoreStdDelta(),
			RejectedRateDelta: resp.Delta.GetRejectedRateDelta(),
			TotalRecordsDelta: resp.Delta.GetTotalRecordsDelta(),
			MatchedCountDelta: resp.Delta.GetMatchedCountDelta(),
		},
	})
}

func (h *Handler) handleBacktestResults(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	limit, err := parseIntQueryParam(r, "limit", 50, 1, 100)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	ruleID := r.URL.Query().Get("rule_id")
	startDate, err := parseOptionalTime(r.URL.Query().Get("start_date"))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	endDate, err := parseOptionalTime(r.URL.Query().Get("end_date"))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	req := &crudv1.ListBacktestResultsRequest{RuleId: ruleID}
	if startDate != nil {
		req.StartDate = timestamppb.New(*startDate)
	}
	if endDate != nil {
		req.EndDate = timestamppb.New(*endDate)
	}

	resp, err := h.analyticsClient.ListBacktestResults(r.Context(), req)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	results := make([]backtestResultResponse, 0, len(resp.GetResults()))
	for _, res := range resp.GetResults() {
		results = append(results, mapBacktestResult(res))
	}

	if int(limit) < len(results) {
		results = results[:limit]
	}

	writeAnalyticsJSON(w, backtestResultsListResponse{
		Results: results,
		Total:   len(results),
	})
}

func parseOptionalTime(raw string) (*time.Time, error) {
	if raw == "" {
		return nil, nil
	}

	parsed, err := parseISODate(raw)
	if err != nil {
		return nil, err
	}
	return &parsed, nil
}

func parseISODate(raw string) (time.Time, error) {
	layouts := []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02",
	}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, raw); err == nil {
			return t, nil
		}
	}
	return time.Time{}, errors.New("Invalid date format. Use ISO format (YYYY-MM-DD)")
}

func parseIntQueryParam(r *http.Request, name string, defaultValue, minValue, maxValue int32) (int32, error) {
	raw := r.URL.Query().Get(name)
	if raw == "" {
		return defaultValue, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return 0, errors.New("invalid query parameter")
	}
	if value < int(minValue) || value > int(maxValue) {
		return 0, errors.New("query parameter out of range")
	}
	return int32(value), nil
}

type backtestMetricsResponse struct {
	TotalRecords      int64            `json:"total_records"`
	MatchedCount      int64            `json:"matched_count"`
	MatchRate         float64          `json:"match_rate"`
	ScoreDistribution map[string]int32 `json:"score_distribution"`
	ScoreMean         float64          `json:"score_mean"`
	ScoreStd          float64          `json:"score_std"`
	ScoreMin          int32            `json:"score_min"`
	ScoreMax          int32            `json:"score_max"`
	RejectedCount     int64            `json:"rejected_count"`
	RejectedRate      float64          `json:"rejected_rate"`
}

type backtestResultResponse struct {
	JobID          string                  `json:"job_id"`
	RuleID         *string                 `json:"rule_id"`
	RulesetVersion string                  `json:"ruleset_version"`
	StartDate      string                  `json:"start_date"`
	EndDate        string                  `json:"end_date"`
	Metrics        backtestMetricsResponse `json:"metrics"`
	CompletedAt    string                  `json:"completed_at"`
	Error          *string                 `json:"error"`
}

type backtestResultsListResponse struct {
	Results []backtestResultResponse `json:"results"`
	Total   int                      `json:"total"`
}

func mapBacktestResult(res *crudv1.BacktestResult) backtestResultResponse {
	if res == nil {
		return backtestResultResponse{}
	}
	var ruleID *string
	if value := res.GetRuleId(); value != "" {
		ruleID = &value
	}
	var errText *string
	if value := res.GetError(); value != "" {
		errText = &value
	}

	metrics := res.GetMetrics()
	scoreDist := map[string]int32{}
	if metrics != nil && metrics.ScoreDistribution != nil {
		scoreDist = metrics.GetScoreDistribution()
	}

	startDate := ""
	if res.GetStartDate() != nil {
		startDate = res.GetStartDate().AsTime().UTC().Format(time.RFC3339)
	}
	endDate := ""
	if res.GetEndDate() != nil {
		endDate = res.GetEndDate().AsTime().UTC().Format(time.RFC3339)
	}
	completedAt := ""
	if res.GetCompletedAt() != nil {
		completedAt = res.GetCompletedAt().AsTime().UTC().Format(time.RFC3339)
	}

	return backtestResultResponse{
		JobID:          res.GetJobId(),
		RuleID:         ruleID,
		RulesetVersion: res.GetRulesetVersion(),
		StartDate:      startDate,
		EndDate:        endDate,
		Metrics: backtestMetricsResponse{
			TotalRecords:      metrics.GetTotalRecords(),
			MatchedCount:      metrics.GetMatchedCount(),
			MatchRate:         metrics.GetMatchRate(),
			ScoreDistribution: scoreDist,
			ScoreMean:         metrics.GetScoreMean(),
			ScoreStd:          metrics.GetScoreStd(),
			ScoreMin:          metrics.GetScoreMin(),
			ScoreMax:          metrics.GetScoreMax(),
			RejectedCount:     metrics.GetRejectedCount(),
			RejectedRate:      metrics.GetRejectedRate(),
		},
		CompletedAt: completedAt,
		Error:       errText,
	}
}
