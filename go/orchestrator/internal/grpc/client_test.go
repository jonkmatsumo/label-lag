package grpc

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestCircuitBreaker(t *testing.T) {
	cb := &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: 2,
		resetTimeout:     100 * time.Millisecond,
	}

	// 1. First failure
	cb.RecordResult(status.Error(codes.Unavailable, "fail"))
	if cb.state != StateClosed {
		t.Errorf("expected state Closed, got %v", cb.state)
	}

	// 2. Second failure -> Open
	cb.RecordResult(status.Error(codes.Unavailable, "fail"))
	if cb.state != StateOpen {
		t.Errorf("expected state Open, got %v", cb.state)
	}

	// 3. Allow should fail
	if err := cb.Allow(); err == nil {
		t.Error("expected Allow to return error when open")
	}

	// 4. Wait for timeout -> Half-Open
	time.Sleep(110 * time.Millisecond)
	if err := cb.Allow(); err != nil {
		t.Errorf("expected Allow to succeed after timeout, got %v", err)
	}
	if cb.state != StateHalfOpen {
		t.Errorf("expected state HalfOpen, got %v", cb.state)
	}

	// 5. Success -> Closed
	cb.RecordResult(nil)
	if cb.state != StateClosed {
		t.Errorf("expected state Closed after success, got %v", cb.state)
	}
	if cb.failureCount != 0 {
		t.Errorf("expected failureCount 0, got %d", cb.failureCount)
	}
}

func TestCircuitBreaker_HalfOpenFailure(t *testing.T) {
	cb := &CircuitBreaker{
		state:            StateOpen,
		failureThreshold: 2,
		resetTimeout:     10 * time.Millisecond,
		lastFailureTime:  time.Now().Add(-20 * time.Millisecond),
	}

	// Should be half-open now
	if err := cb.Allow(); err != nil {
		t.Fatalf("expected Allow to succeed, got %v", err)
	}

	// Failure in half-open -> Open immediately
	cb.RecordResult(status.Error(codes.Unavailable, "fail"))
	if cb.state != StateOpen {
		t.Errorf("expected state Open after half-open failure, got %v", cb.state)
	}
}

func TestNewCircuitBreakerWithPrefix(t *testing.T) {
	// Default breaker should have INFERENCE name
	cb := NewCircuitBreaker()
	if cb.name != "INFERENCE" {
		t.Errorf("expected name INFERENCE, got %q", cb.name)
	}

	// Custom prefix
	cb2 := NewCircuitBreakerWithPrefix("TRAINING")
	if cb2.name != "TRAINING" {
		t.Errorf("expected name TRAINING, got %q", cb2.name)
	}
	if cb2.failureThreshold != defaultBreakerThreshold {
		t.Errorf("expected default threshold %d, got %d", defaultBreakerThreshold, cb2.failureThreshold)
	}

	// Env var override
	t.Setenv("FORECAST_BREAKER_FAILURE_THRESHOLD", "10")
	t.Setenv("FORECAST_BREAKER_RESET_TIMEOUT_MS", "5000")
	cb3 := NewCircuitBreakerWithPrefix("FORECAST")
	if cb3.failureThreshold != 10 {
		t.Errorf("expected threshold 10, got %d", cb3.failureThreshold)
	}
	if cb3.resetTimeout != 5000*time.Millisecond {
		t.Errorf("expected timeout 5s, got %v", cb3.resetTimeout)
	}
}

func TestCircuitBreaker_IgnoredErrors(t *testing.T) {
	cb := &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: 2,
		resetTimeout:     10 * time.Millisecond,
	}

	// 1. Record NotFound (should be ignored)
	errNotFound := status.Error(codes.NotFound, "not found")
	cb.RecordResult(errNotFound)
	if cb.state != StateClosed {
		t.Errorf("expected state Closed after NotFound, got %v", cb.state)
	}
	if cb.failureCount != 0 {
		t.Errorf("expected failureCount 0 after NotFound, got %d", cb.failureCount)
	}

	// 2. Record InvalidArgument (should be ignored)
	errInvalid := status.Error(codes.InvalidArgument, "invalid")
	cb.RecordResult(errInvalid)
	if cb.failureCount != 0 {
		t.Errorf("expected failureCount 0 after InvalidArgument, got %d", cb.failureCount)
	}

	// 3. Record Unavailable (should count as failure)
	errUnavailable := status.Error(codes.Unavailable, "unavailable")
	cb.RecordResult(errUnavailable)
	if cb.failureCount != 1 {
		t.Errorf("expected failureCount 1 after Unavailable, got %d", cb.failureCount)
	}
}

func TestCircuitBreaker_PrometheusMetrics(t *testing.T) {
	cb := NewCircuitBreakerWithPrefix("TEST_METRICS")
	cb.failureThreshold = 2
	cb.resetTimeout = 10 * time.Millisecond

	// Trip the breaker: closed -> open
	cb.RecordResult(status.Error(codes.Unavailable, "fail"))
	cb.RecordResult(status.Error(codes.Unavailable, "fail"))

	stateVal := testutil.ToFloat64(circuitBreakerState.WithLabelValues("TEST_METRICS"))
	if stateVal != float64(StateOpen) {
		t.Errorf("expected state gauge %v (open), got %v", float64(StateOpen), stateVal)
	}

	transVal := testutil.ToFloat64(circuitBreakerTransitions.WithLabelValues("TEST_METRICS", "closed", "open"))
	if transVal != 1 {
		t.Errorf("expected 1 closed->open transition, got %v", transVal)
	}

	// Wait for cooldown, then Allow -> half-open
	time.Sleep(15 * time.Millisecond)
	if err := cb.Allow(); err != nil {
		t.Fatalf("expected Allow to succeed, got %v", err)
	}

	stateVal = testutil.ToFloat64(circuitBreakerState.WithLabelValues("TEST_METRICS"))
	if stateVal != float64(StateHalfOpen) {
		t.Errorf("expected state gauge %v (half-open), got %v", float64(StateHalfOpen), stateVal)
	}

	transVal = testutil.ToFloat64(circuitBreakerTransitions.WithLabelValues("TEST_METRICS", "open", "half-open"))
	if transVal != 1 {
		t.Errorf("expected 1 open->half-open transition, got %v", transVal)
	}

	// Success -> closed
	cb.RecordResult(nil)

	stateVal = testutil.ToFloat64(circuitBreakerState.WithLabelValues("TEST_METRICS"))
	if stateVal != float64(StateClosed) {
		t.Errorf("expected state gauge %v (closed), got %v", float64(StateClosed), stateVal)
	}

	transVal = testutil.ToFloat64(circuitBreakerTransitions.WithLabelValues("TEST_METRICS", "half-open", "closed"))
	if transVal != 1 {
		t.Errorf("expected 1 half-open->closed transition, got %v", transVal)
	}
}
