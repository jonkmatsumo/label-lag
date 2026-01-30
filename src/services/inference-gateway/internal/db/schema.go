package db

import (
	"context"
	"database/sql"
	"fmt"
)

type TableSchema struct {
	Name    string
	Columns []string
}

func ValidateSchema(ctx context.Context, db *sql.DB, expected []TableSchema) error {
	for _, ts := range expected {
		// Check table existence
		var exists bool
		err := db.QueryRowContext(ctx, `
			SELECT EXISTS (
				SELECT FROM information_schema.tables 
				WHERE table_schema = 'public' 
				AND table_name = $1
			)
		`, ts.Name).Scan(&exists)
		if err != nil {
			return fmt.Errorf("check table %s existence: %w", ts.Name, err)
		}
		if !exists {
			return fmt.Errorf("required table %s is missing", ts.Name)
		}

		// Check columns
		for _, col := range ts.Columns {
			err := db.QueryRowContext(ctx, `
				SELECT EXISTS (
					SELECT FROM information_schema.columns 
					WHERE table_schema = 'public' 
					AND table_name = $1 
					AND column_name = $2
				)
			`, ts.Name, col).Scan(&exists)
			if err != nil {
				return fmt.Errorf("check column %s.%s existence: %w", ts.Name, col, err)
			}
			if !exists {
				return fmt.Errorf("required column %s.%s is missing", ts.Name, col)
			}
		}
	}
	return nil
}
