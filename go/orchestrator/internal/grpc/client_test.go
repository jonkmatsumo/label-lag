package grpc

import (
	"testing"
	"time"

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
