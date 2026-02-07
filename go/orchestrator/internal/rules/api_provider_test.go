package rules

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestAPIProviderBackoffSkipsRapidRetries(t *testing.T) {
	var calls int32
	provider := NewAPIProvider("http://rules.local", time.Minute)
	provider.httpClient = &http.Client{
		Transport: roundTripperFunc(func(_ *http.Request) (*http.Response, error) {
			atomic.AddInt32(&calls, 1)
			return &http.Response{
				StatusCode: http.StatusInternalServerError,
				Body:       io.NopCloser(strings.NewReader("boom")),
				Header:     make(http.Header),
			}, nil
		}),
	}
	provider.baseBackoff = 50 * time.Millisecond
	provider.maxBackoff = 100 * time.Millisecond
	provider.jitterFn = func(time.Duration) time.Duration { return 0 }

	if _, err := provider.GetRules(context.Background()); err == nil {
		t.Fatal("expected error on first fetch")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected 1 fetch, got %d", got)
	}

	if _, err := provider.GetRules(context.Background()); err == nil {
		t.Fatal("expected error during backoff")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected no additional fetch during backoff, got %d", got)
	}

	time.Sleep(60 * time.Millisecond)
	if _, err := provider.GetRules(context.Background()); err == nil {
		t.Fatal("expected error after backoff with failing server")
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("expected second fetch after backoff, got %d", got)
	}
}

func TestAPIProviderSingleflight(t *testing.T) {
	var calls int32
	release := make(chan struct{})
	provider := NewAPIProvider("http://rules.local", 0)
	provider.httpClient = &http.Client{
		Transport: roundTripperFunc(func(_ *http.Request) (*http.Response, error) {
			atomic.AddInt32(&calls, 1)
			<-release
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"version":"v1","rules":[]}`)),
				Header:     http.Header{"Content-Type": []string{"application/json"}},
			}, nil
		}),
	}
	provider.jitterFn = func(time.Duration) time.Duration { return 0 }

	var wg sync.WaitGroup
	errCh := make(chan error, 5)
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := provider.GetRules(context.Background())
			errCh <- err
		}()
	}

	time.Sleep(20 * time.Millisecond)
	close(release)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected 1 fetch, got %d", got)
	}
}
