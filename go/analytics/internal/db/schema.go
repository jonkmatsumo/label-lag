package db

import (
	"database/sql"
)

func InitDB(db *sql.DB) error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS backtest_results (
			job_id TEXT PRIMARY KEY,
			rule_id TEXT,
			ruleset_version TEXT NOT NULL,
			start_date TIMESTAMP NOT NULL,
			end_date TIMESTAMP NOT NULL,
			metrics JSONB NOT NULL,
			completed_at TIMESTAMP NOT NULL,
			error TEXT
		)`,
		`CREATE TABLE IF NOT EXISTS rules (
			rule_id TEXT PRIMARY KEY,
			field TEXT NOT NULL,
			op TEXT NOT NULL,
			value TEXT NOT NULL,
			action TEXT NOT NULL,
			score INTEGER,
			severity TEXT NOT NULL,
			reason TEXT,
			status TEXT NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS inference_events (
			id SERIAL PRIMARY KEY,
			ts TIMESTAMP NOT NULL DEFAULT NOW(),
			request_id TEXT NOT NULL,
			model_version TEXT NOT NULL,
			rules_version TEXT NOT NULL,
			model_score INTEGER NOT NULL,
			final_score INTEGER NOT NULL,
			rule_impacts JSONB NOT NULL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_backtest_results_rule_id ON backtest_results(rule_id)`,
		`CREATE INDEX IF NOT EXISTS idx_backtest_results_completed_at ON backtest_results(completed_at)`,
		`CREATE INDEX IF NOT EXISTS idx_rules_status ON rules(status)`,
		`CREATE INDEX IF NOT EXISTS idx_inference_events_ts ON inference_events(ts)`,
		// Rule Versioning
		`CREATE TABLE IF NOT EXISTS rule_versions (
			version_id TEXT PRIMARY KEY,
			rule_id TEXT NOT NULL,
			rule_json JSONB NOT NULL,
			created_at TIMESTAMP NOT NULL,
			created_by TEXT,
			change_description TEXT,
			status TEXT NOT NULL,
			is_active BOOLEAN DEFAULT FALSE
		)`,
		`CREATE INDEX IF NOT EXISTS idx_rule_versions_rule_id ON rule_versions(rule_id)`,
		`CREATE INDEX IF NOT EXISTS idx_rule_versions_created_at ON rule_versions(created_at)`,
		`ALTER TABLE rules ADD COLUMN IF NOT EXISTS active_version_id TEXT`,
	}

	for _, q := range queries {
		if _, err := db.Exec(q); err != nil {
			return err
		}
	}
	return nil
}
