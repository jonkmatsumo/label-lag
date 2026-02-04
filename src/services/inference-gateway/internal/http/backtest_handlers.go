package httpserver

import "net/http"

// TODO(phase4): Map core backtest read/compare routes.
// - GET /backtest/results
// - POST /backtest/compare

func (h *Handler) handleBacktestResults(w http.ResponseWriter, r *http.Request) {}

func (h *Handler) handleBacktestCompare(w http.ResponseWriter, r *http.Request) {}
