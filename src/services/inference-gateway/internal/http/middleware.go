package httpserver

import (
	"log/slog"
	"net/http"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

func requestIDMiddleware(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := r.Header.Get("X-Request-Id")
		if requestID == "" {
			requestID = requestid.Generate()
		}

		w.Header().Set("X-Request-Id", requestID)

		ctx := requestid.WithRequestID(r.Context(), requestID)
		tracer := otel.Tracer("inference-gateway")
		ctx, span := tracer.Start(ctx, "HTTP "+r.Method+" "+r.URL.Path, trace.WithAttributes(
			attribute.String("http.method", r.Method),
			attribute.String("http.target", r.URL.Path),
			attribute.String("request_id", requestID),
		))
		defer span.End()

		next.ServeHTTP(w, r.WithContext(ctx))
		logger.Info("request completed", "method", r.Method, "path", r.URL.Path, "request_id", requestID)
	})
}
