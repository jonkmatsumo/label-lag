package grpc

import (
	"context"
	"errors"
	"testing"

	gatewayv1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/http/gatewayv1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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

func TestExplainEvaluation_NotImplemented(t *testing.T) {
	server := NewGatewayServer(&mockRulesProvider{}, true)
	req := &gatewayv1.ExplainEvaluationRequest{}

	_, err := server.ExplainEvaluation(context.Background(), req)
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("expected Unimplemented, got %v", err)
	}
}
