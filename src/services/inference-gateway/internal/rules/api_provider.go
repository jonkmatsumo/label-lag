package rules

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

type APIProvider struct {
	apiURL     string
	httpClient *http.Client
	ttl        time.Duration

	mu           sync.RWMutex
	cachedRules  RuleSet
	lastFetched  time.Time
}

func NewAPIProvider(apiURL string, ttl time.Duration) *APIProvider {
	return &APIProvider{
		apiURL:     apiURL,
		httpClient: &http.Client{Timeout: 10 * time.Second},
		ttl:        ttl,
	}
}

func (p *APIProvider) GetRules(ctx context.Context) (RuleSet, error) {
	p.mu.RLock()
	if time.Since(p.lastFetched) < p.ttl && p.cachedRules.Version != "" {
		rules := p.cachedRules
		p.mu.RUnlock()
		return rules, nil
	}
	p.mu.RUnlock()

	return p.fetchAndCache(ctx)
}

func (p *APIProvider) fetchAndCache(ctx context.Context) (RuleSet, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Double check if another goroutine fetched while we were waiting for lock
	if time.Since(p.lastFetched) < p.ttl && p.cachedRules.Version != "" {
		return p.cachedRules, nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.apiURL, nil)
	if err != nil {
		return p.cachedRules, fmt.Errorf("create request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		if p.cachedRules.Version != "" {
			return p.cachedRules, nil // Fallback to stale cache
		}
		return RuleSet{}, fmt.Errorf("fetch rules from API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		if p.cachedRules.Version != "" {
			return p.cachedRules, nil // Fallback to stale cache
		}
		return RuleSet{}, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var payload ruleSetJSON
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		if p.cachedRules.Version != "" {
			return p.cachedRules, nil // Fallback to stale cache
		}
		return RuleSet{}, fmt.Errorf("decode API response: %w", err)
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

	p.cachedRules = RuleSet{
		Version: payload.Version,
		Rules:   rules,
	}
	p.lastFetched = time.Now()

	return p.cachedRules, nil
}
