package rules

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
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
	lastAttempt  time.Time
	nextAttempt  time.Time
	backoff      time.Duration
	baseBackoff  time.Duration
	maxBackoff   time.Duration
	jitterFn     func(time.Duration) time.Duration
	fetching     bool
	cond         *sync.Cond
	lastErr      error
}

func NewAPIProvider(apiURL string, ttl time.Duration) *APIProvider {
	provider := &APIProvider{
		apiURL:     apiURL,
		httpClient: &http.Client{Timeout: 10 * time.Second},
		ttl:        ttl,
		baseBackoff: 500 * time.Millisecond,
		maxBackoff:  30 * time.Second,
		jitterFn:    defaultJitter,
	}
	provider.cond = sync.NewCond(&provider.mu)
	return provider
}

func (p *APIProvider) GetRules(ctx context.Context) (RuleSet, error) {
	for {
		p.mu.Lock()
		if time.Since(p.lastFetched) < p.ttl && p.cachedRules.Version != "" {
			rules := p.cachedRules
			p.mu.Unlock()
			return rules, nil
		}
		if !p.nextAttempt.IsZero() && time.Now().Before(p.nextAttempt) {
			cached := p.cachedRules
			nextAttempt := p.nextAttempt
			p.mu.Unlock()
			if cached.Version != "" {
				return cached, nil
			}
			return RuleSet{}, fmt.Errorf("rules provider in backoff until %s", nextAttempt.Format(time.RFC3339))
		}
		if p.fetching {
			for p.fetching {
				p.cond.Wait()
			}
			cached := p.cachedRules
			lastErr := p.lastErr
			p.mu.Unlock()
			if cached.Version != "" {
				return cached, nil
			}
			if lastErr != nil {
				return RuleSet{}, lastErr
			}
			return RuleSet{}, fmt.Errorf("rules unavailable")
		}
		p.fetching = true
		p.mu.Unlock()

		rules, err := p.fetchAndCache(ctx)

		p.mu.Lock()
		p.fetching = false
		p.cond.Broadcast()
		p.mu.Unlock()

		return rules, err
	}
}

func (p *APIProvider) fetchAndCache(ctx context.Context) (RuleSet, error) {
	now := time.Now()
	p.mu.Lock()
	p.lastAttempt = now
	p.mu.Unlock()

	rules, err := p.fetchRules(ctx)
	if err != nil {
		p.mu.Lock()
		p.noteFailure(now)
		p.lastErr = err
		cached := p.cachedRules
		p.mu.Unlock()
		if cached.Version != "" {
			return cached, nil
		}
		return RuleSet{}, err
	}

	p.mu.Lock()
	p.cachedRules = rules
	p.lastFetched = time.Now()
	p.backoff = 0
	p.nextAttempt = time.Time{}
	p.lastErr = nil
	p.mu.Unlock()

	return rules, nil
}

func (p *APIProvider) fetchRules(ctx context.Context) (RuleSet, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.apiURL, nil)
	if err != nil {
		return RuleSet{}, fmt.Errorf("create request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return RuleSet{}, fmt.Errorf("fetch rules from API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return RuleSet{}, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var payload ruleSetJSON
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
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

	rules, _ = FilterValidRules(rules)

	return RuleSet{
		Version: payload.Version,
		Rules:   rules,
	}, nil
}

func (p *APIProvider) noteFailure(now time.Time) {
	if p.backoff <= 0 {
		p.backoff = p.baseBackoff
	} else {
		p.backoff *= 2
		if p.backoff > p.maxBackoff {
			p.backoff = p.maxBackoff
		}
	}
	p.nextAttempt = now.Add(p.backoff + p.jitterFn(p.backoff))
}

func defaultJitter(backoff time.Duration) time.Duration {
	if backoff <= 0 {
		return 0
	}
	return time.Duration(rand.Int63n(int64(backoff/2) + 1))
}
