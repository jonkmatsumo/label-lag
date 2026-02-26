package db

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestMapDBError_DeadlineExceededIncrementsTimeoutCounter(t *testing.T) {
	before := testutil.ToFloat64(dbTimeoutTotal)

	err := MapDBError(context.DeadlineExceeded)
	if status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("expected DeadlineExceeded status, got %v", status.Code(err))
	}

	after := testutil.ToFloat64(dbTimeoutTotal)
	if after-before != 1 {
		t.Fatalf("expected timeout counter delta 1, got %v", after-before)
	}
}

func TestMapDBError_NonTimeoutDoesNotIncrementTimeoutCounter(t *testing.T) {
	before := testutil.ToFloat64(dbTimeoutTotal)

	err := MapDBError(context.Canceled)
	if status.Code(err) != codes.Canceled {
		t.Fatalf("expected Canceled status, got %v", status.Code(err))
	}

	after := testutil.ToFloat64(dbTimeoutTotal)
	if after-before != 0 {
		t.Fatalf("expected timeout counter delta 0, got %v", after-before)
	}
}
