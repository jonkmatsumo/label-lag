package httpserver

import (
	"log/slog"
	"net/http"
	"time"
)

func NewServer(addr string, logger *slog.Logger) *http.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)

	h := requestIDMiddleware(logger, mux)

	return &http.Server{
		Addr:              addr,
		Handler:           h,
		ReadHeaderTimeout: 5 * time.Second,
	}
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}
