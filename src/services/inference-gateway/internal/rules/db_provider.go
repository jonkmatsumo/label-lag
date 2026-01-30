package rules

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	_ "github.com/lib/pq"
)

type DBProvider struct {
	db     *sql.DB
	logger *slog.Logger
}

func NewDBProvider(dsn string, logger *slog.Logger) (*DBProvider, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}

	// Ping to ensure connection is valid
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("ping db: %w", err)
	}

	return &DBProvider{
		db:     db,
		logger: logger,
	}, nil
}

func (p *DBProvider) Close() error {
	return p.db.Close()
}

func (p *DBProvider) GetRules(ctx context.Context) (RuleSet, error) {
	// 1. Get latest published ruleset
	var rulesetID int
	var versionName string
	err := p.db.QueryRowContext(ctx, `
		SELECT id, version_name 
		FROM published_rulesets 
		ORDER BY published_at DESC 
		LIMIT 1
	`).Scan(&rulesetID, &versionName)

	if err != nil {
		if err == sql.ErrNoRows {
			return RuleSet{}, fmt.Errorf("no published ruleset found")
		}
		return RuleSet{}, fmt.Errorf("query latest ruleset: %w", err)
	}

	// 2. Get rules for this ruleset
	rows, err := p.db.QueryContext(ctx, `
		SELECT rv.rule_id, rv.field, rv.op, rv.value, rv.action, rv.score, rv.severity, rv.reason
		FROM rule_versions rv
		JOIN published_ruleset_versions prv ON rv.id = prv.version_id
		WHERE prv.ruleset_id = $1
	`, rulesetID)
	if err != nil {
		return RuleSet{}, fmt.Errorf("query rules: %w", err)
	}
	defer rows.Close()

	var rules []Rule
	for rows.Next() {
		var r Rule
		var valJSON []byte
		var status string = "active" // Default to active since they are from published_rulesets

		err := rows.Scan(&r.ID, &r.Field, &r.Op, &valJSON, &r.Action, &r.Score, &r.Severity, &r.Reason)
		if err != nil {
			return RuleSet{}, fmt.Errorf("scan rule: %w", err)
		}

		// Unpack value from {"v": actual_value}
		var valWrapper struct {
			V any `json:"v"`
		}
		if err := json.Unmarshal(valJSON, &valWrapper); err != nil {
			// Fallback if not wrapped (should not happen with new schema)
			if err := json.Unmarshal(valJSON, &r.Value); err != nil {
				return RuleSet{}, fmt.Errorf("unmarshal rule value: %w", err)
			}
		} else {
			r.Value = valWrapper.V
		}

		r.Status = RuleStatus(status)
		rules = append(rules, r)
	}

	if err := rows.Err(); err != nil {
		return RuleSet{}, fmt.Errorf("rows error: %w", err)
	}

	return RuleSet{
		Version: versionName,
		Rules:   rules,
	}, nil
}
