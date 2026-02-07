package grpc

import (
	"context"
	"errors"
	"testing"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	gatewayv1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/http/gatewayv1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestReloadRules_Disabled(t *testing.T) {
	server := NewGatewayServer(&mockRulesProvider{}, false)
	req := &gatewayv1.ReloadRulesRequest{Source: gatewayv1.ReloadRulesRequest_FILE_PROVIDER}

	_, err := server.ReloadRules(context.Background(), req)
	if status.Code(err) != codes.PermissionDenied {
		t.Errorf("expected PermissionDenied, got %v", err)
	}
}

func TestReloadRules_Success(t *testing.T) {
	mockRules := rules.RuleSet{Version: "v2"}
	provider := &mockRulesProvider{ruleset: mockRules}
	server := NewGatewayServer(provider, true)

	req := &gatewayv1.ReloadRulesRequest{Source: gatewayv1.ReloadRulesRequest_FILE_PROVIDER}
	resp, err := server.ReloadRules(context.Background(), req)
	if err != nil {
		t.Fatalf("ReloadRules failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("expected success")
	}
	if resp.RulesVersion != "v2" {
		t.Errorf("expected version v2, got %s", resp.RulesVersion)
	}
}

func TestReloadRules_ProviderError(t *testing.T) {
	provider := &mockRulesProvider{err: errors.New("io error")}
	server := NewGatewayServer(provider, true)

	req := &gatewayv1.ReloadRulesRequest{Source: gatewayv1.ReloadRulesRequest_API_PROVIDER}
	_, err := server.ReloadRules(context.Background(), req)
	if status.Code(err) != codes.Internal {
		t.Errorf("expected Internal error, got %v", err)
	}
}

func TestReloadRules_InvalidSource(t *testing.T) {
	server := NewGatewayServer(&mockRulesProvider{}, true)

	req := &gatewayv1.ReloadRulesRequest{Source: gatewayv1.ReloadRulesRequest_SOURCE_UNSPECIFIED}
	_, err := server.ReloadRules(context.Background(), req)
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", err)
	}
}

func TestExplainEvaluation_Disabled(t *testing.T) {
	server := NewGatewayServer(&mockRulesProvider{}, false)
	req := &gatewayv1.ExplainEvaluationRequest{}

	_, err := server.ExplainEvaluation(context.Background(), req)
	if status.Code(err) != codes.PermissionDenied {
		t.Errorf("expected PermissionDenied, got %v", err)
	}
}

func TestExplainEvaluation_Success(t *testing.T) {
	mockRules := rules.RuleSet{
		Version: "v1",
		Rules: []rules.Rule{
			{ID: "r1", Field: "f1", Op: ">", Value: 10, Action: "reject", Status: rules.RuleStatusActive},
		},
	}
	provider := &mockRulesProvider{ruleset: mockRules}
	server := NewGatewayServer(provider, true)

	features, _ := structpb.NewStruct(map[string]any{"f1": 20.0})
	req := &gatewayv1.ExplainEvaluationRequest{
		Features: features,
	}

	resp, err := server.ExplainEvaluation(context.Background(), req)
	if err != nil {
		t.Fatalf("ExplainEvaluation failed: %v", err)
	}

	if resp.MatchedCount != 1 {
		t.Errorf("expected matched count 1, got %d", resp.MatchedCount)
	}
	if len(resp.Traces) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(resp.Traces))
	}
	trace := resp.Traces[0]
	if trace.RuleId != "r1" {
		t.Errorf("expected rule id r1, got %s", trace.RuleId)
	}
	if !trace.Matched {
		t.Errorf("expected matched true")
	}
	if len(trace.Conditions) != 1 {
		t.Fatalf("expected 1 condition trace")
	}
	cond := trace.Conditions[0]
	if cond.Field != "f1" {
		t.Errorf("expected field f1, got %s", cond.Field)
	}
	if !cond.Result {
		t.Errorf("expected condition result true")
	}
}

func TestExplainEvaluation_WithProvidedRuleset(t *testing.T) {
	server := NewGatewayServer(&mockRulesProvider{}, true)

	features, _ := structpb.NewStruct(map[string]any{"f1": 5.0})
	// Rule r1: f1 > 10 (should not match)
	customRules := &gatewayv1.RuleSet{
		Rules: []*crudv1.Rule{
			{Id: "r1", Field: "f1", Op: ">", ValueJson: "10", Action: "reject", Status: "active"},
		},
	}

	req := &gatewayv1.ExplainEvaluationRequest{
		Features: features,
		Ruleset:  customRules,
	}

	resp, err := server.ExplainEvaluation(context.Background(), req)
	if err != nil {
		t.Fatalf("ExplainEvaluation failed: %v", err)
	}

	if resp.MatchedCount != 0 {
		t.Errorf("expected matched count 0, got %d", resp.MatchedCount)
	}
	if len(resp.Traces) != 1 {
		t.Fatalf("expected 1 trace")
	}
	if resp.Traces[0].Matched {
		t.Errorf("expected matched false")
	}
}
