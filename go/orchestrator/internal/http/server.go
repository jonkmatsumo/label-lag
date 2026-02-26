package httpserver

import (
	"log/slog"
	"net/http"
	"time"
)

func NewServer(addr string, logger *slog.Logger, handler *Handler, readTimeout, writeTimeout, idleTimeout time.Duration) *http.Server {
	// Resolve env-driven global rate limit values at startup.
	resetGlobalLimiterFromEnv()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	if handler != nil {
		handler.Register(mux)
	}

	h := http.Handler(mux)
	h = rateLimitMiddleware(h)
	h = tenancyMiddleware(h)
	h = metricsMiddleware(logger, h)
	h = requestIDMiddleware(logger, h)

	return &http.Server{
		Addr:              addr,
		Handler:           h,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       idleTimeout,
		ReadHeaderTimeout: 5 * time.Second,
	}
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}
