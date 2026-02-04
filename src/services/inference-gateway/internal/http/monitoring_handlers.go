package httpserver

import "net/http"

// TODO(phase4): Map core monitoring read routes.
// - GET /monitoring/drift
// - GET /metrics/shadow/comparison

func (h *Handler) handleMonitoringDrift(w http.ResponseWriter, r *http.Request) {}

func (h *Handler) handleMetricsShadowComparison(w http.ResponseWriter, r *http.Request) {}
