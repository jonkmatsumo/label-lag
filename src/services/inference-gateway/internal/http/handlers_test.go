package httpserver

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
)

func TestHandleEvaluateSignal_RejectsLargeBody(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	handler := NewHandler(logger, nil, rules.NewEmptyProvider(), 32)

	body := strings.Repeat("a", 64)
	req := httptest.NewRequest(http.MethodPost, "/evaluate/signal", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.handleEvaluateSignal(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected status %d, got %d", http.StatusRequestEntityTooLarge, rec.Code)
	}
}
