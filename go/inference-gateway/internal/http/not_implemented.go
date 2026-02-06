package httpserver

import (
	"encoding/json"
	"net/http"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
)

var notImplementedRoutes = []string{}

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
