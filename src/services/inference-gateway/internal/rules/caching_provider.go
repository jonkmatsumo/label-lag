package rules

import (
	"context"
	"log/slog"
	"sync"
	"time"
)

type CachingProvider struct {
	underlying Provider
	interval   time.Duration
	logger     *slog.Logger

	mu      sync.RWMutex
	ruleset RuleSet

	stop chan struct{}
}

func NewCachingProvider(underlying Provider, interval time.Duration, logger *slog.Logger) (*CachingProvider, error) {
	p := &CachingProvider{
		underlying: underlying,
		interval:   interval,
		logger:     logger,
		stop:       make(chan struct{}),
	}

	// Initial load
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	ruleset, err := underlying.GetRules(ctx)
	if err != nil {
		return nil, err
	}
	p.ruleset = ruleset

	go p.refreshLoop()

	return p, nil
}

func (p *CachingProvider) GetRules(_ context.Context) (RuleSet, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.ruleset, nil
}

func (p *CachingProvider) Stop() {
	close(p.stop)
}

func (p *CachingProvider) refreshLoop() {
	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			ruleset, err := p.underlying.GetRules(ctx)
			cancel()

			if err != nil {
				p.logger.Error("failed to refresh ruleset", "error", err)
				continue
			}

			p.mu.Lock()
			p.ruleset = ruleset
			p.mu.Unlock()
			p.logger.Info("ruleset refreshed", "version", ruleset.Version, "rules_count", len(ruleset.Rules))
		case <-p.stop:
			return
		}
	}
}
