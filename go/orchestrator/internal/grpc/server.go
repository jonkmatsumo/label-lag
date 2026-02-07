package grpc

import (
	"context"

	gatewayv1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/http/gatewayv1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GatewayServer implements the GatewayService gRPC interface.
type GatewayServer struct {
	gatewayv1.UnimplementedGatewayServiceServer
	rulesProvider rules.Provider
}

// NewGatewayServer creates a new instance of GatewayServer.
func NewGatewayServer(rp rules.Provider) *GatewayServer {
	return &GatewayServer{
		rulesProvider: rp,
	}
}

// Register registers the GatewayServer with the gRPC server.
func (s *GatewayServer) Register(server *grpc.Server) {
	gatewayv1.RegisterGatewayServiceServer(server, s)
}

// EvaluateRules implements the EvaluateRules RPC.
func (s *GatewayServer) EvaluateRules(ctx context.Context, req *gatewayv1.EvaluateRulesRequest) (*gatewayv1.EvaluateRulesResponse, error) {
	features := protoToFeatures(req.Features)
	baseScore := int(req.BaseScore)

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
			// Log warning but proceed with empty ruleset? Or return error?
			// HTTP handler logs warning.
			// for gRPC, maybe we should return error if we can't load default rules?
			// HTTP handler:
			// 		ruleset, err = h.rulesProvider.GetRules(r.Context())
			// 		if err != nil { h.logger.Warn(...) }
			// It proceeds with empty ruleset (zero value) effectively.
		}
	}

	result, err := rules.EvaluateRules(features, baseScore, &ruleset, rules.EvalOptions{Debug: false}) // TODO: Pass debug flag from metadata?
	if err != nil {
		return nil, status.Errorf(codes.Internal, "evaluation failed: %v", err)
	}

	// Add telemetry
	span := sdktrace.SpanFromContext(ctx)
	span.SetAttributes(
		attribute.String("rules.version", result.RulesVersion),
		attribute.Float64("rules.eval_time_ms", result.EvaluationTimeMS),
		attribute.Int("rules.matched_count", len(result.MatchedRules)),
		attribute.Bool("rules.rejected", result.Rejected),
	)

	resp := ruleResultToProto(result, baseScore, req.ShadowMode)
	return resp, nil
}

// EvaluateRulesDiff implements the EvaluateRulesDiff RPC.
func (s *GatewayServer) EvaluateRulesDiff(ctx context.Context, req *gatewayv1.EvaluateRulesDiffRequest) (*gatewayv1.EvaluateRulesDiffResponse, error) {
	features := protoToFeatures(req.Features)
	baseScore := int(req.BaseScore)

	// Helper to evaluate
	eval := func(pbRS *gatewayv1.RuleSet) (rules.RuleResult, error) {
		var ruleset rules.RuleSet
		var err error

		if pbRS != nil {
			rs, err := protoToRuleSet(pbRS)
			if err != nil {
				return rules.RuleResult{}, status.Errorf(codes.InvalidArgument, "invalid ruleset: %v", err)
			}
			validRules, errs := rules.FilterValidRules(rs.Rules)
			if len(errs) > 0 {
				return rules.RuleResult{}, status.Errorf(codes.InvalidArgument, "invalid rules: %v", errs)
			}
			ruleset = rs
			ruleset.Rules = validRules
		} else {
			ruleset, err = s.rulesProvider.GetRules(ctx)
			if err != nil {
				// Log warning
			}
		}
		return rules.EvaluateRules(features, baseScore, &ruleset, rules.EvalOptions{Debug: false})
	}

	resA, err := eval(req.RulesetA)
	if err != nil {
		return nil, err
	}
	resB, err := eval(req.RulesetB)
	if err != nil {
		return nil, err
	}

	diff := rules.ComputeDiff(resA, resB, req.ShadowMode)

	return &gatewayv1.EvaluateRulesDiffResponse{
		A:                     ruleResultToProto(resA, baseScore, req.ShadowMode),
		B:                     ruleResultToProto(resB, baseScore, req.ShadowMode),
		Diff:                  diffSummaryToProto(diff),
		TotalEvaluationTimeMs: resA.EvaluationTimeMS + resB.EvaluationTimeMS,
	}, nil
}
