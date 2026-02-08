package db

import (
	"database/sql"
)

func InitDB(db *sql.DB) error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS generated_records (
			id SERIAL PRIMARY KEY,
			record_id TEXT UNIQUE NOT NULL,
			user_id TEXT NOT NULL,
			full_name TEXT,
			email TEXT,
			phone TEXT,
			transaction_timestamp TIMESTAMP NOT NULL,
			is_off_hours_txn BOOLEAN,
			available_balance FLOAT,
			balance_to_transaction_ratio FLOAT,
			avg_available_balance_30d FLOAT,
			balance_volatility_z_score FLOAT,
			bank_connections_count_24h INTEGER,
			bank_connections_count_7d INTEGER,
			bank_connections_avg_30d FLOAT,
			amount FLOAT,
			amount_to_avg_ratio FLOAT,
			merchant_risk_score INTEGER,
			is_returned BOOLEAN,
			email_changed_at TIMESTAMP,
			phone_changed_at TIMESTAMP,
			is_fraudulent BOOLEAN,
			fraud_type TEXT,
			numerical_features JSONB,
			categorical_features JSONB,
			created_at TIMESTAMP NOT NULL DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS evaluation_metadata (
			id SERIAL PRIMARY KEY,
			user_id TEXT NOT NULL,
			record_id TEXT UNIQUE NOT NULL,
			sequence_number INTEGER,
			fraud_confirmed_at TIMESTAMP,
			is_pre_fraud BOOLEAN,
			days_to_fraud INTEGER,
			is_train_eligible BOOLEAN,
			created_at TIMESTAMP NOT NULL DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS feature_snapshots (
			id SERIAL PRIMARY KEY,
			record_id TEXT UNIQUE NOT NULL,
			user_id TEXT NOT NULL,
			velocity_24h INTEGER,
			amount_to_avg_ratio_30d FLOAT,
			balance_volatility_z_score FLOAT,
			experimental_signals JSONB,
			computed_at TIMESTAMP NOT NULL DEFAULT NOW()
		)`,
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
		`CREATE TABLE IF NOT EXISTS rule_impacts (
			id SERIAL PRIMARY KEY,
			request_id TEXT NOT NULL,
			rule_id TEXT NOT NULL,
			is_shadow BOOLEAN,
			score_delta INTEGER
		)`,
		`CREATE INDEX IF NOT EXISTS idx_generated_records_record_id ON generated_records(record_id)`,
		`CREATE INDEX IF NOT EXISTS idx_evaluation_metadata_record_id ON evaluation_metadata(record_id)`,
		`CREATE INDEX IF NOT EXISTS idx_feature_snapshots_record_id ON feature_snapshots(record_id)`,
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
		// Dynamic features support (redundant if tables created above but good for safety)
		`ALTER TABLE generated_records ADD COLUMN IF NOT EXISTS numerical_features JSONB`,
		`ALTER TABLE generated_records ADD COLUMN IF NOT EXISTS categorical_features JSONB`,
	}

	for _, q := range queries {
		if _, err := db.Exec(q); err != nil {
			return err
		}
	}
	return nil
}
