package grpc

import (
	"errors"
	"testing"
	"time"
)

func TestCircuitBreaker(t *testing.T) {
	cb := &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: 2,
		resetTimeout:     100 * time.Millisecond,
	}

	// 1. First failure
	cb.RecordResult(errors.New("fail"))
	if cb.state != StateClosed {
		t.Errorf("expected state Closed, got %v", cb.state)
	}

	// 2. Second failure -> Open
	cb.RecordResult(errors.New("fail"))
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
	cb.RecordResult(errors.New("fail"))
	if cb.state != StateOpen {
		t.Errorf("expected state Open after half-open failure, got %v", cb.state)
	}
}
