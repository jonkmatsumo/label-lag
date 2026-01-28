package rules

import "context"

type Provider interface {
	GetRules(ctx context.Context) (RuleSet, error)
}

type StaticProvider struct {
	ruleset RuleSet
}

func NewStaticProvider(ruleset RuleSet) *StaticProvider {
	return &StaticProvider{ruleset: ruleset}
}

func (p *StaticProvider) GetRules(_ context.Context) (RuleSet, error) {
	return p.ruleset, nil
}

func NewEmptyProvider() *StaticProvider {
	return &StaticProvider{ruleset: RuleSet{Version: "", Rules: []Rule{}}}
}
