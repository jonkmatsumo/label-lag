package grpc

import (
	"context"
	"testing"
	"time"
)

func TestRemainingBudget_NoDeadline(t *testing.T) {
	ctx := context.Background()
	got := RemainingBudget(ctx, 3*time.Second)
	if got != 3*time.Second {
		t.Errorf("expected 3s default, got %v", got)
	}
}

func TestRemainingBudget_NoDeadline_ClampedToMax(t *testing.T) {
	ctx := context.Background()
	got := RemainingBudget(ctx, 30*time.Second)
	if got != budgetMax {
		t.Errorf("expected budgetMax %v, got %v", budgetMax, got)
	}
}

func TestRemainingBudget_DeadlinePresent(t *testing.T) {
	// 2s from now → remaining = 2s - 200ms safety = 1800ms
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	got := RemainingBudget(ctx, 5*time.Second)
	// Should be roughly 1800ms, allow some slack for test execution
	if got < 1500*time.Millisecond || got > 1900*time.Millisecond {
		t.Errorf("expected ~1800ms, got %v", got)
	}
}

func TestRemainingBudget_NearExpiry(t *testing.T) {
	// 300ms from now → remaining = 300ms - 200ms = 100ms → clamped to budgetMin
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	got := RemainingBudget(ctx, 5*time.Second)
	if got != budgetMin {
		t.Errorf("expected budgetMin %v for near-expiry, got %v", budgetMin, got)
	}
}

func TestRemainingBudget_PastDeadline(t *testing.T) {
	// 100ms from now → remaining = 100ms - 200ms = negative → clamped to budgetMin
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	got := RemainingBudget(ctx, 5*time.Second)
	if got != budgetMin {
		t.Errorf("expected budgetMin %v for past deadline, got %v", budgetMin, got)
	}
}

func TestRemainingBudget_DefaultClampedBelowMin(t *testing.T) {
	ctx := context.Background()
	got := RemainingBudget(ctx, 50*time.Millisecond)
	if got != budgetMin {
		t.Errorf("expected budgetMin %v for tiny default, got %v", budgetMin, got)
	}
}

func TestRemainingBudget_LargeRemaining_ClampedToDefault(t *testing.T) {
	// 10s from now → remaining = 10s - 200ms = 9800ms → clamped to default (3s)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	got := RemainingBudget(ctx, 3*time.Second)
	if got != 3*time.Second {
		t.Errorf("expected 3s (default), got %v", got)
	}
}

func TestClampDuration(t *testing.T) {
	tests := []struct {
		name     string
		d, min   time.Duration
		max      time.Duration
		expected time.Duration
	}{
		{"below min", 50 * time.Millisecond, 200 * time.Millisecond, 5 * time.Second, 200 * time.Millisecond},
		{"above max", 10 * time.Second, 200 * time.Millisecond, 5 * time.Second, 5 * time.Second},
		{"in range", 2 * time.Second, 200 * time.Millisecond, 5 * time.Second, 2 * time.Second},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := clampDuration(tt.d, tt.min, tt.max)
			if got != tt.expected {
				t.Errorf("clampDuration(%v, %v, %v) = %v, want %v", tt.d, tt.min, tt.max, got, tt.expected)
			}
		})
	}
}
