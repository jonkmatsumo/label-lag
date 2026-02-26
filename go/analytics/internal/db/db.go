package db

import (
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"

	"github.com/lib/pq"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var dbTimeoutTotal = promauto.NewCounter(prometheus.CounterOpts{
	Name: "analytics_db_timeout_total",
	Help: "Total number of database operations that timed out before completion.",
})

func MapDBError(err error) error {
	if err == nil {
		return nil
	}

	var pqErr *pq.Error
	if errors.As(err, &pqErr) {
		switch pqErr.Code {
		case "23505": // unique_violation
			return status.Error(codes.AlreadyExists, "resource already exists")
		case "23503": // foreign_key_violation
			return status.Error(codes.FailedPrecondition, "foreign key violation")
		case "40P01": // deadlock_detected
			return status.Error(codes.Unavailable, "database deadlock detected")
		}
	}

	if errors.Is(err, sql.ErrNoRows) {
		return status.Error(codes.NotFound, "resource not found")
	}
	if errors.Is(err, context.DeadlineExceeded) {
		dbTimeoutTotal.Inc()
		return status.Error(codes.DeadlineExceeded, "database query timed out")
	}
	if errors.Is(err, context.Canceled) {
		return status.Error(codes.Canceled, "request canceled")
	}
	return status.Errorf(codes.Internal, "database error: %v", err)
}

// EncodeCursor takes a variable number of keys and encodes them as a base64 JSON array string
func EncodeCursor(keys ...interface{}) string {
	b, err := json.Marshal(keys)
	if err != nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(b)
}

// DecodeCursor decodes a base64 JSON array string into an array of interfaces
func DecodeCursor(cursor string) ([]interface{}, error) {
	b, err := base64.StdEncoding.DecodeString(cursor)
	if err != nil {
		return nil, err
	}
	var keys []interface{}
	if err := json.Unmarshal(b, &keys); err != nil {
		return nil, err
	}
	return keys, nil
}
