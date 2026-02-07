package httpserver

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	forecastv1 "github.com/jonkmatsumo/label-lag/go/forecast/proto/forecastv1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
)

// TODO(phase4): Map core monitoring read routes.
// - GET /monitoring/drift
// - GET /metrics/shadow/comparison

func (h *Handler) handleMonitoringDrift(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.forecastClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "forecast backend unavailable")
		return
	}

	hours, _ := parseIntQuery(r, "hours", 24, 1, 168)
	thresholdRaw := r.URL.Query().Get("threshold")
	var threshold float64
	if thresholdRaw != "" {
		threshold, _ = strconv.ParseFloat(thresholdRaw, 64)
	}
	forceRefresh := r.URL.Query().Get("force_refresh") == "true"

	resp, err := h.forecastClient.GetDriftMonitoring(r.Context(), &forecastv1.GetDriftMonitoringRequest{
		Hours:        hours,
		Threshold:    threshold,
		ForceRefresh: forceRefresh,
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *Handler) handleMetricsShadowComparison(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	hours, _ := parseIntQuery(r, "hours", 24, 1, 720)

	resp, err := h.analyticsClient.GetShadowComparison(r.Context(), &crudv1.GetShadowComparisonRequest{
		Hours: hours,
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

var apiHTTPClient = &http.Client{Timeout: 10 * time.Second}

func proxyAPIGet(w http.ResponseWriter, r *http.Request) {
	apiBaseURL := strings.TrimRight(getAPIBaseURL(), "/")
	if apiBaseURL == "" {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	targetURL := apiBaseURL + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, targetURL, nil)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "analytics backend error")
		return
	}
	if reqID := requestid.FromContext(r.Context()); reqID != "" {
		req.Header.Set("X-Request-Id", reqID)
	}

	resp, err := apiHTTPClient.Do(req)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "analytics backend error")
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "analytics backend error")
		return
	}

	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)
}

func getAPIBaseURL() string {
	if value := strings.TrimSpace(os.Getenv("INFERENCE_GATEWAY_API_URL")); value != "" {
		return value
	}
	return "http://training-server:8000"
}
