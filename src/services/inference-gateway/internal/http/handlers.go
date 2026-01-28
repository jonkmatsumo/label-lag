package httpserver

import (
	"errors"
	"io"
	"log/slog"
	"math"
	"net/http"

	grpcclient "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/grpc"
	inferencev1 "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/grpc/inferencev1/inference/v1"
	gatewayv1 "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/http/gatewayv1/gateway/v1"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/rules"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

type Handler struct {
	logger          *slog.Logger
	inferenceClient *grpcclient.InferenceClient
	rulesProvider   rules.Provider
}

func NewHandler(logger *slog.Logger, client *grpcclient.InferenceClient, provider rules.Provider) *Handler {
	return &Handler{
		logger:          logger,
		inferenceClient: client,
		rulesProvider:   provider,
	}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("/evaluate/signal", h.handleEvaluateSignal)
}

func (h *Handler) handleEvaluateSignal(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req gatewayv1.SignalRequest
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(body, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	normalizeSignalRequest(&req)
	if err := validateSignalRequest(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	if h.inferenceClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "inference backend unavailable")
		return
	}

	inferenceResp, err := h.inferenceClient.Score(r.Context(), &inferencev1.ScoreRequest{
		UserId:              req.UserId,
		Amount:              req.Amount,
		Currency:            req.Currency,
		ClientTransactionId: req.ClientTransactionId,
		RequestId:           requestid.FromContext(r.Context()),
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	requestID := inferenceResp.GetRequestId()
	if requestID == "" {
		requestID = requestid.FromContext(r.Context())
	}

	features := map[string]any{}
	if inferenceResp.FeaturesUsed != nil {
		features = inferenceResp.FeaturesUsed.AsMap()
	}

	ruleset, err := h.rulesProvider.GetRules(r.Context())
	if err != nil {
		h.logger.Warn("failed to load ruleset", "error", err)
		ruleset = rules.RuleSet{}
	}

	rawScore := int32(math.Round(inferenceResp.GetModelScore()))

	ruleResult, err := rules.EvaluateRules(features, int(rawScore), ruleset)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "rule evaluation failed")
		return
	}

	riskComponents := buildRiskComponents(features)
	for _, explanation := range ruleResult.Explanations {
		riskComponents = append(riskComponents, &gatewayv1.RiskComponent{
			Key:   "rule_" + explanation.RuleID,
			Label: explanation.Reason,
		})
	}

	response := &gatewayv1.SignalResponse{
		RequestId:          requestID,
		Score:              int32(ruleResult.FinalScore),
		RiskComponents:     riskComponents,
		ModelVersion:       inferenceResp.GetModelVersion(),
		MatchedRules:       buildMatchedRules(ruleResult.Explanations),
		ShadowMatchedRules: buildMatchedRules(ruleResult.ShadowExplanations),
	}

	if len(ruleResult.MatchedRules) > 0 {
		response.ModelScore = wrapperspb.Int32(rawScore)
	}
	if ruleset.Version != "" {
		response.RulesVersion = wrapperspb.String(ruleset.Version)
	}

	writeProtoJSON(w, response)
}

func normalizeSignalRequest(req *gatewayv1.SignalRequest) {
	if req.Currency == "" {
		req.Currency = "USD"
	}
}

func validateSignalRequest(req *gatewayv1.SignalRequest) error {
	if req.UserId == "" {
		return errors.New("user_id is required")
	}
	if req.Amount <= 0 {
		return errors.New("amount must be greater than 0")
	}
	if req.ClientTransactionId == "" {
		return errors.New("client_transaction_id is required")
	}
	return nil
}

func buildMatchedRules(explanations []rules.Explanation) []*gatewayv1.MatchedRule {
	matched := make([]*gatewayv1.MatchedRule, 0, len(explanations))
	for _, exp := range explanations {
		matched = append(matched, &gatewayv1.MatchedRule{
			RuleId:   exp.RuleID,
			Severity: exp.Severity,
			Reason:   exp.Reason,
		})
	}
	return matched
}

func writeProtoJSON(w http.ResponseWriter, msg *gatewayv1.SignalResponse) {
	w.Header().Set("Content-Type", "application/json")
	payload, err := protojson.MarshalOptions{
		EmitUnpopulated: true,
		UseProtoNames:   true,
	}.Marshal(msg)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "failed to serialize response")
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(payload)
}

func writeJSONError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(`{"detail":"` + message + `"}`))
}

func writeRPCError(w http.ResponseWriter, err error) {
	var rpcErr *grpcclient.RPCError
	if errors.As(err, &rpcErr) {
		switch rpcErr.Code {
		case codes.InvalidArgument:
			writeJSONError(w, http.StatusBadRequest, rpcErr.Message)
		case codes.DeadlineExceeded, codes.Unavailable:
			writeJSONError(w, http.StatusServiceUnavailable, "inference backend timeout")
		default:
			writeJSONError(w, http.StatusBadGateway, rpcErr.Message)
		}
		return
	}
	writeJSONError(w, http.StatusBadGateway, "inference backend error")
}

func buildRiskComponents(features map[string]any) []*gatewayv1.RiskComponent {
	components := []*gatewayv1.RiskComponent{}

	if toFloat(features["velocity_24h"]) > 5 {
		components = append(components, &gatewayv1.RiskComponent{Key: "velocity", Label: "high_transaction_velocity"})
	}
	if toFloat(features["amount_to_avg_ratio_30d"]) > 3.0 {
		components = append(components, &gatewayv1.RiskComponent{Key: "amount_ratio", Label: "unusual_transaction_amount"})
	}
	if toFloat(features["balance_volatility_z_score"]) < -2.0 {
		components = append(components, &gatewayv1.RiskComponent{Key: "balance", Label: "low_balance_volatility"})
	}
	if toFloat(features["bank_connections_24h"]) > 4 {
		components = append(components, &gatewayv1.RiskComponent{Key: "connections", Label: "connection_burst_detected"})
	}
	if toFloat(features["merchant_risk_score"]) > 70 {
		components = append(components, &gatewayv1.RiskComponent{Key: "merchant", Label: "high_risk_merchant"})
	}
	if hasHistory, ok := features["has_history"].(bool); ok && !hasHistory {
		components = append(components, &gatewayv1.RiskComponent{Key: "history", Label: "insufficient_history"})
	}

	return components
}

func toFloat(value any) float64 {
	switch v := value.(type) {
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case int64:
		return float64(v)
	case float32:
		return float64(v)
	case float64:
		return v
	default:
		return 0
	}
}
