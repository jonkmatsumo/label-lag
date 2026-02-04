package httpserver

import (
	"encoding/json"
	"net/http"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
)

var notImplementedRoutes = []string{
	"/analytics/attribution",
	"/analytics/relationships",
	"/analytics/correlations",
	"/analytics/rules/",
	"/monitoring/drift",
	"/metrics/shadow/comparison",
	"/backtest/results",
	"/backtest/compare",
	"/train",
	"/models/deploy",
	"/rules",
	"/rules/",
	"/suggestions/heuristic",
	"/suggestions/accept",
	"/data/generate",
	"/data/clear",
	"/mlflow/experiments/search",
	"/mlflow/runs/search",
	"/mlflow/model-versions/search",
	"/mlflow/model-versions/transition-stage",
	"/mlflow/runs/",
}

type notImplementedResponse struct {
	Error     string `json:"error"`
	Path      string `json:"path"`
	Method    string `json:"method"`
	RequestID string `json:"request_id"`
}

func (h *Handler) handleNotImplemented(w http.ResponseWriter, r *http.Request) {
	writeNotImplemented(w, r)
}

func writeNotImplemented(w http.ResponseWriter, r *http.Request) {
	reqID := requestid.FromContext(r.Context())
	if reqID == "" {
		reqID = requestid.Generate()
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotImplemented)
	_ = json.NewEncoder(w).Encode(notImplementedResponse{
		Error:     "not_implemented",
		Path:      r.URL.Path,
		Method:    r.Method,
		RequestID: reqID,
	})
}
