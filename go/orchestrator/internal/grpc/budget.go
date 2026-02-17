package grpc

import (
	"context"
	"time"
)

const (
	// budgetSafetyMargin is subtracted from the remaining deadline
	// before propagating to downstream hops. This allows the caller
	// to handle the response (serialization, logging) without racing
	// against its own deadline.
	budgetSafetyMargin = 200 * time.Millisecond

	// budgetMin prevents issuing a downstream call with an
	// unreasonably short timeout that is almost certain to fail.
	budgetMin = 200 * time.Millisecond

	// budgetMax prevents a missing upstream deadline from producing
	// an infinite downstream timeout. It falls back to the client's
	// configured default.
	budgetMax = 5 * time.Second
)

// RemainingBudget computes the timeout to use for a downstream gRPC call
// given the current request context and a default timeout.
//
// If the context already carries a deadline the returned duration is:
//
//	clamp(deadline - now - safetyMargin, budgetMin, defaultTimeout)
//
// If the context has no deadline the default timeout is returned unchanged.
func RemainingBudget(ctx context.Context, defaultTimeout time.Duration) time.Duration {
	deadline, ok := ctx.Deadline()
	if !ok {
		return clampDuration(defaultTimeout, budgetMin, budgetMax)
	}

	remaining := time.Until(deadline) - budgetSafetyMargin
	if remaining < budgetMin {
		return budgetMin
	}

	max := defaultTimeout
	if max > budgetMax {
		max = budgetMax
	}
	if remaining > max {
		return max
	}
	return remaining
}

func clampDuration(d, min, max time.Duration) time.Duration {
	if d < min {
		return min
	}
	if d > max {
		return max
	}
	return d
}
