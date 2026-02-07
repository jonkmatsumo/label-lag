package httpserver

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestStatusResponseWriterTracksStatusAndBytes(t *testing.T) {
	rec := httptest.NewRecorder()
	writer := &statusResponseWriter{ResponseWriter: rec}

	writer.WriteHeader(http.StatusTeapot)
	_, _ = writer.Write([]byte("hello"))

	if writer.status != http.StatusTeapot {
		t.Fatalf("expected status %d, got %d", http.StatusTeapot, writer.status)
	}
	if writer.bytes != 5 {
		t.Fatalf("expected 5 bytes, got %d", writer.bytes)
	}

	rec2 := httptest.NewRecorder()
	writer2 := &statusResponseWriter{ResponseWriter: rec2}
	_, _ = writer2.Write([]byte("ok"))

	if writer2.status != http.StatusOK {
		t.Fatalf("expected default status %d, got %d", http.StatusOK, writer2.status)
	}
	if writer2.bytes != 2 {
		t.Fatalf("expected 2 bytes, got %d", writer2.bytes)
	}
}
