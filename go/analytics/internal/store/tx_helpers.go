package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	"google.golang.org/grpc/status"
)

func beginRequestBoundTx(ctx context.Context, sqlDB *sql.DB) (context.Context, context.CancelFunc, *sql.Tx, error) {
	txCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	tx, err := sqlDB.BeginTx(txCtx, nil)
	if err != nil {
		cancel()
		return nil, nil, nil, db.MapDBError(fmt.Errorf("failed to begin transaction: %w", err))
	}
	return txCtx, cancel, tx, nil
}

func rollbackTxOnError(tx *sql.Tx, cause error) error {
	if rollbackErr := tx.Rollback(); rollbackErr != nil && !errors.Is(rollbackErr, sql.ErrTxDone) {
		return mapOrPassDBError(fmt.Errorf("%v (rollback failed: %w)", cause, rollbackErr))
	}
	return mapOrPassDBError(cause)
}

func mapOrPassDBError(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := status.FromError(err); ok {
		return err
	}
	return db.MapDBError(err)
}
