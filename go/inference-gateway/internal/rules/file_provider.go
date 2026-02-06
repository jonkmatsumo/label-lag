package rules

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sync/atomic"

	"github.com/fsnotify/fsnotify"
)

type FileProvider struct {
	path   string
	rs     atomic.Pointer[RuleSet]
	logger *slog.Logger
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

func NewFileProvider(path string, logger *slog.Logger) (*FileProvider, error) {
	if path == "" {
		return nil, fmt.Errorf("rules path is required")
	}
	p := &FileProvider{
		path:   path,
		logger: logger,
	}

	// Initial load
	rs, err := p.loadRules()
	if err != nil {
		return nil, fmt.Errorf("initial load: %w", err)
	}
	p.rs.Store(&rs)

	if os.Getenv("INFERENCE_GATEWAY_RULES_WATCH") == "true" {
		if err := p.startWatcher(); err != nil {
			logger.Warn("failed to start rules watcher", "error", err)
		}
	}

	return p, nil
}

func (p *FileProvider) GetRules(_ context.Context) (RuleSet, error) {
	rs := p.rs.Load()
	if rs == nil {
		return RuleSet{}, fmt.Errorf("no rules loaded")
	}
	return *rs, nil
}

func (p *FileProvider) loadRules() (RuleSet, error) {
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

	rules, _ = FilterValidRules(rules)

	return RuleSet{
		Version: payload.Version,
		Rules:   rules,
	}, nil
}

func (p *FileProvider) startWatcher() error {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}

	go func() {
		defer watcher.Close()
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				if event.Has(fsnotify.Write) {
					p.logger.Info("reloading rules due to file change", "path", p.path)
					rs, err := p.loadRules()
					if err != nil {
						p.logger.Error("failed to reload rules", "error", err, "path", p.path)
						// Keep last known good rules
						continue
					}
					p.rs.Store(&rs)
					p.logger.Info("successfully reloaded rules", "version", rs.Version, "path", p.path)
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				p.logger.Error("watcher error", "error", err)
			}
		}
	}()

	return watcher.Add(p.path)
}