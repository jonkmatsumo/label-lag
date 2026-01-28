package rules

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
)

type FileProvider struct {
	path string
}

type ruleSetJSON struct {
	Version string     `json:"version"`
	Rules   []ruleJSON `json:"rules"`
}

type ruleJSON struct {
	ID       string `json:"id"`
	Field    string `json:"field"`
	Op       string `json:"op"`
	Value    any    `json:"value"`
	Action   string `json:"action"`
	Score    *int   `json:"score"`
	Severity string `json:"severity"`
	Reason   string `json:"reason"`
	Status   string `json:"status"`
}

func NewFileProvider(path string) (*FileProvider, error) {
	if path == "" {
		return nil, fmt.Errorf("rules path is required")
	}
	return &FileProvider{path: path}, nil
}

func (p *FileProvider) GetRules(_ context.Context) (RuleSet, error) {
	data, err := os.ReadFile(p.path)
	if err != nil {
		return RuleSet{}, fmt.Errorf("read rules file: %w", err)
	}

	var payload ruleSetJSON
	if err := json.Unmarshal(data, &payload); err != nil {
		return RuleSet{}, fmt.Errorf("parse rules file: %w", err)
	}

	rules := make([]Rule, 0, len(payload.Rules))
	for _, r := range payload.Rules {
		status := RuleStatus(r.Status)
		if status == "" {
			status = RuleStatusActive
		}
		rules = append(rules, Rule{
			ID:       r.ID,
			Field:    r.Field,
			Op:       r.Op,
			Value:    r.Value,
			Action:   r.Action,
			Score:    r.Score,
			Severity: r.Severity,
			Reason:   r.Reason,
			Status:   status,
		})
	}

	return RuleSet{
		Version: payload.Version,
		Rules:   rules,
	}, nil
}
