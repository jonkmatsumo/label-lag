package proxy

import (
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
)

// NewHandler creates a reverse proxy handler for the given target URL.
func NewHandler(target string, logger *slog.Logger) http.HandlerFunc {
	targetURL, err := url.Parse(target)
	if err != nil {
		logger.Error("failed to parse proxy target", "target", target, "error", err)
		return func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "invalid proxy configuration", http.StatusInternalServerError)
		}
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	// Custom error handler
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		logger.Error("proxy error", "target", target, "path", r.URL.Path, "error", err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"detail":"upstream service unavailable"}`))
	}

	return func(w http.ResponseWriter, r *http.Request) {
		// Update the headers to allow for SSL redirection
		r.URL.Host = targetURL.Host
		r.URL.Scheme = targetURL.Scheme
		r.Header.Set("X-Forwarded-Host", r.Header.Get("Host"))
		r.Host = targetURL.Host

		proxy.ServeHTTP(w, r)
	}
}
