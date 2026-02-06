package httpserver

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

func requestIDMiddleware(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
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

		rec := &statusResponseWriter{ResponseWriter: w}
		next.ServeHTTP(rec, r.WithContext(ctx))

		status := rec.status
		if status == 0 {
			status = http.StatusOK
		}

		span.SetAttributes(
			attribute.Int("http.status_code", status),
			attribute.Int("http.response_content_length", rec.bytes),
		)
		if status >= http.StatusBadRequest {
			span.SetStatus(codes.Error, http.StatusText(status))
		}

		logger.Info("request completed",
			"method", r.Method,
			"path", r.URL.Path,
			"request_id", requestID,
			"status", status,
			"duration", time.Since(start),
			"bytes", rec.bytes,
		)
	})
}

type statusResponseWriter struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (w *statusResponseWriter) WriteHeader(statusCode int) {
	w.status = statusCode
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *statusResponseWriter) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	n, err := w.ResponseWriter.Write(data)
	w.bytes += n
	return n, err
}
