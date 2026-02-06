package httpserver

import (
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
)

// TODO(phase4): Map core monitoring read routes.
// - GET /monitoring/drift
// - GET /metrics/shadow/comparison

func (h *Handler) handleMonitoringDrift(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	proxyAPIGet(w, r)
}

func (h *Handler) handleMetricsShadowComparison(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	proxyAPIGet(w, r)
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
	return "http://api:8000"
}
