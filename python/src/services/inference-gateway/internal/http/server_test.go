package httpserver

import (
	"io"
	"log/slog"
	"testing"
	"time"
)

func TestNewServerAppliesTimeouts(t *testing.T) {
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	server := NewServer("127.0.0.1:0", logger, nil, 2*time.Second, 3*time.Second, 4*time.Second)

	if server.ReadTimeout != 2*time.Second {
		t.Fatalf("expected ReadTimeout 2s, got %s", server.ReadTimeout)
	}
	if server.WriteTimeout != 3*time.Second {
		t.Fatalf("expected WriteTimeout 3s, got %s", server.WriteTimeout)
	}
	if server.IdleTimeout != 4*time.Second {
		t.Fatalf("expected IdleTimeout 4s, got %s", server.IdleTimeout)
	}
}
