package grpc

import (
	"context"
	"fmt"
	"time"

	gatewayv1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/http/gatewayv1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
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

	start := time.Now()
	features := protoToFeatures(req.Features)

	var ruleset rules.RuleSet
	var err error

	if req.Ruleset != nil {
		rs, err := protoToRuleSet(req.Ruleset)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "invalid ruleset: %v", err)
		}
		// Filter and validate
		validRules, errs := rules.FilterValidRules(rs.Rules)
		if len(errs) > 0 {
			return nil, status.Errorf(codes.InvalidArgument, "invalid rules: %v", errs)
		}
		ruleset = rs
		ruleset.Rules = validRules
	} else {
		ruleset, err = s.rulesProvider.GetRules(ctx)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to load rules: %v", err)
		}
	}

	// Limit max rules to explain to avoid explosion
	maxRules := 50
	if req.MaxRules > 0 {
		maxRules = int(req.MaxRules)
	}
	if len(ruleset.Rules) > maxRules {
		// Just truncate for explanation? Or fail?
		// Truncating is safer for debug tool.
		ruleset.Rules = ruleset.Rules[:maxRules]
	}

	result, err := rules.EvaluateRules(features, 0, &ruleset, rules.EvalOptions{Explain: true})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "evaluation failed: %v", err)
	}

	traces := make([]*gatewayv1.RuleTrace, 0, len(result.RuleTraces))
	for _, t := range result.RuleTraces {
		conds := make([]*gatewayv1.ConditionTrace, 0, len(t.Conditions))
		for _, c := range t.Conditions {
			actualValue := ""
			if req.IncludeValues {
				// We didn't store actual value in rules.go yet to be safe/simple
				// but we have ActualType.
				// If we want actual value, we'd need to update rules.go to store it.
				// For now, let's leave it empty or "REDACTED"
				actualValue = "REDACTED"
			}
			conds = append(conds, &gatewayv1.ConditionTrace{
				Field:               c.Field,
				Operator:            c.Operator,
				ExpectedType:        c.ExpectedType,
				ActualType:          c.ActualType,
				Result:              c.Result,
				ActualValueRedacted: actualValue,
			})
		}
		traces = append(traces, &gatewayv1.RuleTrace{
			RuleId:     t.RuleID,
			Matched:    t.Matched,
			EvalTimeMs: t.EvalTimeMS,
			Conditions: conds,
		})
	}

	return &gatewayv1.ExplainEvaluationResponse{
		EvalTimeMs:   float64(time.Since(start).Milliseconds()), // or result.EvaluationTimeMS
		MatchedCount: int32(len(result.MatchedRules)),
		RulesVersion: result.RulesVersion,
		Traces:       traces,
	}, nil
}
