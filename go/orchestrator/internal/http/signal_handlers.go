package httpserver

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	forecastv1 "github.com/jonkmatsumo/label-lag/go/forecast/proto/forecastv1"
	grpcclient "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc"
	inferencev1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc/inferencev1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/timestamppb"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

const defaultHopTimeout = 5 * time.Second

func (h *Handler) handleEvaluateSignal(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	startTime := time.Now()
	span := trace.SpanFromContext(r.Context())

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, r, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, r, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req inferencev1.SignalRequest
	if err := (protojson.UnmarshalOptions{}).Unmarshal(body, &req); err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "invalid json payload")
		return
	}

	normalizeSignalRequest(&req)
	if err := validateSignalRequest(&req); err != nil {
		writeJSONError(w, r, http.StatusBadRequest, err.Error())
		return
	}

	if h.forecastClient == nil {
		writeJSONError(w, r, http.StatusServiceUnavailable, "forecast backend unavailable")
		return
	}

	// Compute per-hop timeout budgets from the request deadline so
	// downstream calls don't outlive the caller.
	hopTimeout := grpcclient.RemainingBudget(r.Context(), defaultHopTimeout)
	hopCtx, hopCancel := context.WithTimeout(r.Context(), hopTimeout)
	defer hopCancel()

	inferenceResp, err := h.forecastClient.PredictSignal(hopCtx, &forecastv1.PredictSignalRequest{
		UserId:              req.UserId,
		Amount:              req.Amount,
		Currency:            req.Currency,
		ClientTransactionId: req.ClientTransactionId,
		TenantId:            tenantIDFromRequest(r),
	})
	if err != nil {
		writeRPCError(w, r, err)
		return
	}

	requestID := requestid.FromContext(r.Context())

	// Feature Hydration: Move ownership to Go
	features := map[string]any{}

	// 1. Try to fetch from Analytics (using budget-aware context)
	if h.analyticsClient != nil {
		featureTimeout := grpcclient.RemainingBudget(r.Context(), defaultHopTimeout)
		featureCtx, featureCancel := context.WithTimeout(r.Context(), featureTimeout)
		hydrated, err := h.analyticsClient.GetFeatures(featureCtx, req.UserId, tenantIDFromRequest(r))
		featureCancel()
		if err != nil {
			h.logger.Warn("failed to hydrate features from analytics", "error", err, "user_id", req.UserId)
			// Ensure empty features on analytics failure
			features = map[string]any{}
		} else if hydrated != nil {
			features = hydrated
		} else {
			// 2. Fallback to simulation for unknown users (matching Python behavior)
			features = simulateFeatures(req.UserId, req.Amount)
		}
	}

	// Add transaction specific features
	features["transaction_amount"] = req.Amount

	ruleset, err := h.rulesProvider.GetRules(r.Context())
	if err != nil {
		h.logger.Warn("failed to load ruleset", "error", err)
		rulesFallbackTotal.Inc()
		if cached := h.getLastGoodRuleset(); cached != nil {
			ruleset = *cached
			h.logger.Info("using cached ruleset fallback", "version", ruleset.Version)
		} else {
			writeJSONError(w, r, http.StatusServiceUnavailable, "rules provider unavailable")
			return
		}
	} else if len(ruleset.Rules) > 0 || ruleset.Version != "" {
		h.cacheRuleset(&ruleset)
	}

	rawScore := int32(math.Round(inferenceResp.GetProbability() * 100))

	ruleResult, err := rules.EvaluateRules(features, int(rawScore), &ruleset, rules.EvalOptions{Debug: false})
	if err != nil {
		writeJSONError(w, r, http.StatusInternalServerError, "rule evaluation failed")
		return
	}

	// Calculate rule impacts for logging
	impacts := make(map[string]float64)
	ruleImpacts := []map[string]any{}

	if int(rawScore) != ruleResult.FinalScore {
		totalDelta := math.Abs(float64(ruleResult.FinalScore - int(rawScore)))
		if len(ruleResult.MatchedRules) > 0 {
			perRuleDelta := totalDelta / float64(len(ruleResult.MatchedRules))
			for _, rid := range ruleResult.MatchedRules {
				impacts[rid] = perRuleDelta
			}
		}
	}

	for _, rid := range ruleResult.MatchedRules {
		ruleImpacts = append(ruleImpacts, map[string]any{
			"rule_id":     rid,
			"is_shadow":   false,
			"score_delta": impacts[rid],
		})
	}
	for _, rid := range ruleResult.ShadowMatchedRules {
		ruleImpacts = append(ruleImpacts, map[string]any{
			"rule_id":     rid,
			"is_shadow":   true,
			"score_delta": 0.0,
		})
	}

	// Structured inference event logging
	event := map[string]any{
		"request_id":    requestID,
		"timestamp":     time.Now().Format(time.RFC3339),
		"model_version": inferenceResp.GetModelVersion(),
		"rules_version": ruleResult.RulesVersion,
		"model_score":   rawScore,
		"final_score":   ruleResult.FinalScore,
		"rule_impacts":  ruleImpacts,
	}
	h.logger.Info("InferenceEvent", "event", event)

	// Persist inference event to CRUD (fire-and-forget, bounded)
	if h.analyticsClient != nil {
		pbImpacts := make([]*crudv1.RuleImpact, 0, len(ruleImpacts))
		for _, ri := range ruleImpacts {
			pbImpacts = append(pbImpacts, &crudv1.RuleImpact{
				RuleId:     ri["rule_id"].(string),
				IsShadow:   ri["is_shadow"].(bool),
				ScoreDelta: ri["score_delta"].(float64),
			})
		}

		tenantID := tenantIDFromRequest(r)
		evt := inferenceLogEvent{
			ctx:         context.Background(),
			requestID:   requestID,
			tenantID:    tenantID,
			spanContext: span.SpanContext(),
			req: &crudv1.LogInferenceEventRequest{
				Event: &crudv1.InferenceEvent{
					RequestId:    requestID,
					Timestamp:    timestamppb.Now(),
					ModelVersion: inferenceResp.GetModelVersion(),
					RulesVersion: ruleResult.RulesVersion,
					ModelScore:   rawScore,
					FinalScore:   int32(ruleResult.FinalScore),
					RuleImpacts:  pbImpacts,
					TenantId:     tenantID,
				},
			},
		}

		select {
		case h.logQueue <- evt:
			// Enqueued successfully
		default:
			dropped := h.droppedLogs.Add(1)
			if dropped%100 == 1 {
				h.logger.Warn("dropping inference log event", "total_dropped", dropped)
			}
		}
	}

	// OTEL attributes
	span.SetAttributes(
		attribute.String("app.request_id", requestID),
		attribute.String("app.model_version", inferenceResp.GetModelVersion()),
		attribute.String("app.rules_version", ruleResult.RulesVersion),
		attribute.Int("app.model_score", int(rawScore)),
		attribute.Int("app.final_score", ruleResult.FinalScore),
		attribute.Int("app.rule_matches", len(ruleResult.MatchedRules)),
	)

	riskComponents := buildRiskComponents(features)
	for _, explanation := range ruleResult.Explanations {
		riskComponents = append(riskComponents, &inferencev1.RiskComponent{
			Key:   "rule_" + explanation.RuleID,
			Label: explanation.Reason,
		})
	}

	latencyMs := float64(time.Since(startTime).Microseconds()) / 1000.0

	riskLabel := "LOW"
	if ruleResult.FinalScore >= 80 {
		riskLabel = "HIGH"
	} else if ruleResult.FinalScore >= 30 {
		riskLabel = "MEDIUM"
	}

	response := &inferencev1.SignalResponse{
		RequestId:          requestID,
		Score:              int32(ruleResult.FinalScore),
		RiskLabel:          riskLabel,
		LatencyMs:          latencyMs,
		RiskComponents:     riskComponents,
		ModelVersion:       inferenceResp.GetModelVersion(),
		MatchedRules:       buildMatchedRules(ruleResult.Explanations),
		ShadowMatchedRules: buildMatchedRules(ruleResult.ShadowExplanations),
	}

	if len(ruleResult.MatchedRules) > 0 {
		response.ModelScore = wrapperspb.Int32(rawScore)
	}
	if ruleResult.RulesVersion != "" {
		response.RulesVersion = wrapperspb.String(ruleResult.RulesVersion)
	}

	writeProtoJSON(w, r, response)
}

func normalizeSignalRequest(req *inferencev1.SignalRequest) {
	if req == nil {
		return
	}
	req.UserId = strings.TrimSpace(req.UserId)
	req.Currency = strings.ToUpper(strings.TrimSpace(req.Currency))
	req.ClientTransactionId = strings.TrimSpace(req.ClientTransactionId)
	if req.Currency == "" {
		req.Currency = "USD"
	}
}

func validateSignalRequest(req *inferencev1.SignalRequest) error {
	if req.UserId == "" {
		return errors.New("user_id is required")
	}
	if len(req.UserId) > 64 {
		return errors.New("user_id is too long")
	}
	if req.Amount <= 0 {
		return errors.New("amount must be greater than 0")
	}
	if req.Amount > 10000000 {
		return errors.New("amount exceeds maximum limit")
	}
	if len(req.Currency) != 3 {
		return errors.New("currency must be a 3-letter ISO code")
	}
	if req.ClientTransactionId == "" {
		return errors.New("client_transaction_id is required")
	}
	if len(req.ClientTransactionId) > 128 {
		return errors.New("client_transaction_id is too long")
	}
	return nil
}

func buildMatchedRules(explanations []rules.Explanation) []*inferencev1.MatchedRule {
	matched := make([]*inferencev1.MatchedRule, 0, len(explanations))
	for _, exp := range explanations {
		matched = append(matched, &inferencev1.MatchedRule{
			RuleId:      exp.RuleID,
			Severity:    exp.Severity,
			Reason:      exp.Reason,
			Explanation: exp.Explanation,
		})
	}
	return matched
}

func writeProtoJSON(w http.ResponseWriter, r *http.Request, msg *inferencev1.SignalResponse) {
	w.Header().Set("Content-Type", "application/json")
	payload, err := protojson.MarshalOptions{
		EmitUnpopulated: true,
		UseProtoNames:   true,
	}.Marshal(msg)
	if err != nil {
		writeJSONError(w, r, http.StatusInternalServerError, "failed to serialize response")
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(payload)
}
