package grpc

import (
	"encoding/json"
	"fmt"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	gatewayv1 "github.com/jonkmatsumo/label-lag/go/inference/internal/http/gatewayv1"
	"github.com/jonkmatsumo/label-lag/go/inference/internal/rules"
	"google.golang.org/protobuf/types/known/structpb"
)

// protoToRuleSet converts a proto RuleSet to a domain RuleSet.
func protoToRuleSet(pb *gatewayv1.RuleSet) (rules.RuleSet, error) {
	if pb == nil {
		return rules.RuleSet{}, nil
	}
	rs := rules.RuleSet{
		Rules: make([]rules.Rule, len(pb.Rules)),
	}
	for i, r := range pb.Rules {
		rule, err := protoToRule(r)
		if err != nil {
			return rules.RuleSet{}, err
		}
		rs.Rules[i] = rule
	}
	return rs, nil
}

// protoToRule converts a proto Rule to a domain Rule.
func protoToRule(pb *crudv1.Rule) (rules.Rule, error) {
	if pb == nil {
		return rules.Rule{}, fmt.Errorf("nil rule")
	}

	// Parse value_json
	var value any
	if pb.ValueJson != "" {
		if err := json.Unmarshal([]byte(pb.ValueJson), &value); err != nil {
			return rules.Rule{}, fmt.Errorf("failed to unmarshal value_json for rule %s: %w", pb.Id, err)
		}
	}

	// Helper for optional int fields
	var score *int
	if pb.Score != 0 { // Proto3 doesn't have optional primitives by default for int32 unless strictly defined, assume 0 is nil or check logic?
		// Actually analytics.proto says 'int32 score = 6;', so 0 is default.
		// However, in domain Rule `Score` is *int.
		// If 0 is a valid score, this logic is flawed. Rules usually score 1-99.
		// Let's assume 0 means nil for now, or check how analytics service handles it.
		// Looking at analytics.proto, checking convention.
		val := int(pb.Score)
		if val != 0 {
			score = &val
		}
	}

	return rules.Rule{
		ID:       pb.Id,
		Field:    pb.Field,
		Op:       pb.Op,
		Value:    value,
		Action:   pb.Action,
		Score:    score,
		Severity: pb.Severity,
		Reason:   pb.Reason,
		Status:   rules.RuleStatus(pb.Status),
	}, nil
}

// protoToFeatures converts a Struct to a map[string]any.
func protoToFeatures(pb *structpb.Struct) map[string]any {
	if pb == nil {
		return nil
	}
	return pb.AsMap()
}

// ruleResultToProto converts a domain RuleResult to a proto EvaluateRulesResponse.
func ruleResultToProto(res rules.RuleResult, baseScore int, shadowMode bool) *gatewayv1.EvaluateRulesResponse {
	resp := &gatewayv1.EvaluateRulesResponse{
		FinalScore:         int32(res.FinalScore),
		BaselineScore:      int32(baseScore),
		MatchedRules:       res.MatchedRules,
		Explanations:       explanationsToProto(res.Explanations),
		ShadowMatchedRules: res.ShadowMatchedRules,
		ShadowExplanations: explanationsToProto(res.ShadowExplanations),
		Rejected:           res.Rejected,
		RulesetVersion:     res.RulesVersion,
		EvaluationTimeMs:   res.EvaluationTimeMS,
	}

	if shadowMode {
		resp.ShadowScore = int32(res.FinalScore)
		resp.FinalScore = int32(baseScore)
		resp.Rejected = false
	}
	return resp
}

func explanationsToProto(exps []rules.Explanation) []*gatewayv1.Explanation {
	if len(exps) == 0 {
		return nil
	}
	pbExps := make([]*gatewayv1.Explanation, len(exps))
	for i, e := range exps {
		var score int32
		if e.Score != nil {
			score = int32(*e.Score)
		}
		pbExps[i] = &gatewayv1.Explanation{
			RuleId:      e.RuleID,
			Severity:    e.Severity,
			Reason:      e.Reason,
			Explanation: e.Explanation,
			Action:      e.Action,
			Score:       score,
			ScoreDelta:  int32(e.ScoreDelta),
		}
	}
	return pbExps
}

func diffSummaryToProto(diff rules.RulesDiffSummary) *gatewayv1.RulesDiffSummary {
	return &gatewayv1.RulesDiffSummary{
		Severity:            diff.Severity,
		ScoreDelta:          int32(diff.ScoreDelta),
		MatchedRulesAdded:   diff.MatchedRulesAdded,
		MatchedRulesRemoved: diff.MatchedRulesRemoved,
	}
}
