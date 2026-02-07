package grpc

import (
	"context"
	"fmt"
	"time"

	gatewayv1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/http/gatewayv1"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ReloadRules implements the ReloadRules RPC.
func (s *GatewayServer) ReloadRules(ctx context.Context, req *gatewayv1.ReloadRulesRequest) (*gatewayv1.ReloadRulesResponse, error) {
	if !s.enableAdminRPCs {
		return nil, status.Error(codes.PermissionDenied, "admin RPCs are disabled")
	}

	start := time.Now()
	span := sdktrace.SpanFromContext(ctx)
	span.SetAttributes(
		attribute.String("rules.reload.source", req.Source.String()),
	)

	// Validate source
	if req.Source == gatewayv1.ReloadRulesRequest_SOURCE_UNSPECIFIED {
		return nil, status.Error(codes.InvalidArgument, "source must be specified")
	}

	// Trigger reload on provider
	err := s.rulesProvider.Reload(ctx)
	duration := time.Since(start).Milliseconds()

	span.SetAttributes(
		attribute.Int64("rules.reload.duration_ms", duration),
		attribute.Bool("rules.reload.success", err == nil),
	)

	if err != nil {
		return &gatewayv1.ReloadRulesResponse{
			Success:    false,
			DurationMs: duration,
			Message:    fmt.Sprintf("failed to reload rules: %v", err),
		}, status.Errorf(codes.Internal, "failed to reload rules: %v", err)
	}

	// Get new version
	ruleset, _ := s.rulesProvider.GetRules(ctx)

	reloadedSources := []gatewayv1.ReloadRulesRequest_Source{}
	// For now we don't distinguish detailed sources in response, assuming provider handles all
	if req.Source == gatewayv1.ReloadRulesRequest_FILE_PROVIDER || req.Source == gatewayv1.ReloadRulesRequest_BOTH {
		reloadedSources = append(reloadedSources, gatewayv1.ReloadRulesRequest_FILE_PROVIDER)
	}
	if req.Source == gatewayv1.ReloadRulesRequest_API_PROVIDER || req.Source == gatewayv1.ReloadRulesRequest_BOTH {
		reloadedSources = append(reloadedSources, gatewayv1.ReloadRulesRequest_API_PROVIDER)
	}

	return &gatewayv1.ReloadRulesResponse{
		Success:         true,
		RulesVersion:    ruleset.Version,
		ReloadedSources: reloadedSources,
		DurationMs:      duration,
		Message:         "rules reloaded successfully",
	}, nil
}

// ExplainEvaluation implements the ExplainEvaluation RPC.
func (s *GatewayServer) ExplainEvaluation(ctx context.Context, req *gatewayv1.ExplainEvaluationRequest) (*gatewayv1.ExplainEvaluationResponse, error) {
	if !s.enableAdminRPCs {
		return nil, status.Error(codes.PermissionDenied, "admin RPCs are disabled")
	}

	// Placeholder for Phase 4
	return nil, status.Error(codes.Unimplemented, "ExplainEvaluation not yet implemented")
}
