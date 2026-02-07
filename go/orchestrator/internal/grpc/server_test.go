package grpc

import (
	"context"
	"testing"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	gatewayv1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/http/gatewayv1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	"google.golang.org/protobuf/types/known/structpb"
)

type mockRulesProvider struct {
	ruleset rules.RuleSet
	err     error
}

func (m *mockRulesProvider) GetRules(ctx context.Context) (rules.RuleSet, error) {
	return m.ruleset, m.err
}

func (m *mockRulesProvider) Reload(ctx context.Context) error {
	return m.err
}

func TestEvaluateRules(t *testing.T) {
	mockRules := rules.RuleSet{
		Version: "v1",
		Rules: []rules.Rule{
			{ID: "r1", Field: "f1", Op: ">", Value: 10, Action: "reject", Status: rules.RuleStatusActive},
		},
	}
	provider := &mockRulesProvider{ruleset: mockRules}
	server := NewGatewayServer(provider, false)

	features, _ := structpb.NewStruct(map[string]any{"f1": 20})
	req := &gatewayv1.EvaluateRulesRequest{
		Features:  features,
		BaseScore: 50,
	}

	resp, err := server.EvaluateRules(context.Background(), req)
	if err != nil {
		t.Fatalf("EvaluateRules failed: %v", err)
	}

	if resp.FinalScore != 99 {
		t.Errorf("expected score 99, got %d", resp.FinalScore)
	}
	if !resp.Rejected {
		t.Errorf("expected rejected")
	}
	if len(resp.MatchedRules) != 1 || resp.MatchedRules[0] != "r1" {
		t.Errorf("expected match r1")
	}
}

func TestEvaluateRulesDiff(t *testing.T) {
	server := NewGatewayServer(&mockRulesProvider{}, false)

	features, _ := structpb.NewStruct(map[string]any{"f1": 20})

	// Ruleset A: no rules
	rsA := &gatewayv1.RuleSet{}

	// Ruleset B: r1 rejects
	rsB := &gatewayv1.RuleSet{
		Rules: []*crudv1.Rule{
			{Id: "r1", Field: "f1", Op: ">", ValueJson: "10", Action: "reject", Status: "active"},
		},
	}

	req := &gatewayv1.EvaluateRulesDiffRequest{
		Features:  features,
		BaseScore: 50,
		RulesetA:  rsA,
		RulesetB:  rsB,
	}

	resp, err := server.EvaluateRulesDiff(context.Background(), req)
	if err != nil {
		t.Fatalf("EvaluateRulesDiff failed: %v", err)
	}

	// B should reject, so score 99
	// A should accept, so score 50
	// Diff score delta = 99 - 50 = 49
	if resp.A.FinalScore != 50 {
		t.Errorf("expected A score 50, got %d", resp.A.FinalScore)
	}
	if resp.B.FinalScore != 99 {
		t.Errorf("expected B score 99, got %d", resp.B.FinalScore)
	}
	if resp.Diff.ScoreDelta != 49 {
		t.Errorf("expected score delta 49, got %d", resp.Diff.ScoreDelta)
	}
	if len(resp.Diff.MatchedRulesAdded) != 1 || resp.Diff.MatchedRulesAdded[0] != "r1" {
		t.Errorf("expected r1 added")
	}
}
