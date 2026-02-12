package httpserver

import (
	"context"
	"encoding/json"
	"errors"
	"hash/fnv"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	grpcclient "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc"
	inferencev1 "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc/inferencev1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/http/proxy"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/rules"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/timestamppb"
	"google.golang.org/protobuf/types/known/wrapperspb"

	forecastv1 "github.com/jonkmatsumo/label-lag/go/forecast/proto/forecastv1"
	trainingv1 "github.com/jonkmatsumo/label-lag/go/training/proto/trainingv1"
)

type InferenceClient interface {
	Score(ctx context.Context, req *inferencev1.ScoreRequest) (*inferencev1.ScoreResponse, error)
	Ready(ctx context.Context) error
}

type TrainingClient interface {
	Train(ctx context.Context, req *trainingv1.TrainRequest) (*trainingv1.TrainResponse, error)
	ClearData(ctx context.Context, req *trainingv1.ClearDataRequest) (*trainingv1.ClearDataResponse, error)
}

type ForecastClient interface {
	GetHealth(ctx context.Context, req *forecastv1.GetHealthRequest) (*forecastv1.GetHealthResponse, error)
	GetDriftMonitoring(ctx context.Context, req *forecastv1.GetDriftMonitoringRequest) (*forecastv1.GetDriftMonitoringResponse, error)
	GetScoreDistribution(ctx context.Context, req *forecastv1.GetScoreDistributionRequest) (*forecastv1.GetScoreDistributionResponse, error)
	ReloadModel(ctx context.Context, req *forecastv1.ReloadModelRequest) (*forecastv1.ReloadModelResponse, error)
	DeployModel(ctx context.Context, req *forecastv1.DeployModelRequest) (*forecastv1.DeployModelResponse, error)
	PredictSignal(ctx context.Context, req *forecastv1.PredictSignalRequest) (*forecastv1.PredictSignalResponse, error)
}

type Handler struct {
	logger          *slog.Logger
	inferenceClient InferenceClient
	analyticsClient AnalyticsClient
	trainingClient  TrainingClient
	forecastClient  ForecastClient
	rulesProvider   rules.Provider
	pythonURL       string
	mlflowURL       string
	maxBodyBytes    int64
	logQueue        chan inferenceLogEvent
	logWg           sync.WaitGroup
	droppedLogs     atomic.Int64
}

type inferenceLogEvent struct {
	ctx         context.Context
	req         *crudv1.LogInferenceEventRequest
	spanContext trace.SpanContext
	requestID   string
}

func NewHandler(logger *slog.Logger, client InferenceClient, analyticsClient AnalyticsClient, trainingClient TrainingClient, forecastClient ForecastClient, provider rules.Provider, maxBodyBytes int64, pythonURL, mlflowURL string) *Handler {
	if maxBodyBytes <= 0 {
		maxBodyBytes = 1 << 20
	}
	h := &Handler{
		logger:          logger,
		inferenceClient: client,
		analyticsClient: analyticsClient,
		trainingClient:  trainingClient,
		forecastClient:  forecastClient,
		rulesProvider:   provider,
		maxBodyBytes:    maxBodyBytes,
		pythonURL:       pythonURL,
		mlflowURL:       mlflowURL,
		logQueue:        make(chan inferenceLogEvent, 100),
	}
	h.startWorkers()
	return h
}

func (h *Handler) startWorkers() {
	for i := 0; i < 5; i++ {
		h.logWg.Add(1)
		go h.logWorker()
	}
}

func (h *Handler) logWorker() {
	defer h.logWg.Done()
	for event := range h.logQueue {
		ctx := event.ctx
		// Create a link to the parent trace
		opts := []trace.SpanStartOption{
			trace.WithLinks(trace.Link{SpanContext: event.spanContext}),
			trace.WithAttributes(attribute.String("request_id", event.requestID)),
		}

		// Start a new root span for the async operation, linked to parent
		var span trace.Span
		if event.spanContext.IsValid() {
			ctx, span = otel.Tracer("async-logger").Start(ctx, "async_log_inference", opts...)
		} else {
			ctx, span = otel.Tracer("async-logger").Start(ctx, "async_log_inference")
		}
		defer span.End()

		_, err := h.analyticsClient.LogInferenceEvent(ctx, event.req)
		if err != nil {
			h.logger.Warn("failed to log inference event to CRUD", "error", err, "request_id", event.requestID)
		}
	}
}

func (h *Handler) Shutdown(ctx context.Context) error {
	close(h.logQueue)
	done := make(chan struct{})
	go func() {
		h.logWg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("/metrics", h.handleMetrics)
	mux.HandleFunc("/evaluate/signal", h.handleEvaluateSignal)
	mux.HandleFunc("/evaluate/rules", h.handleEvaluateRules)
	mux.HandleFunc("/evaluate/rules/diff", h.handleEvaluateRulesDiff)
	mux.HandleFunc("POST /rules/sandbox/evaluate", h.handleSandboxEvaluate)
	mux.HandleFunc("POST /rules/sandbox/diff", h.handleSandboxDiff)
	mux.HandleFunc("/ready", h.handleReady)
	mux.HandleFunc("/analytics/overview", h.handleAnalyticsOverview)
	mux.HandleFunc("/analytics/daily-stats", h.handleAnalyticsDailyStats)
	mux.HandleFunc("/analytics/transactions", h.handleAnalyticsTransactions)
	mux.HandleFunc("/analytics/recent-alerts", h.handleAnalyticsRecentAlerts)

	mux.HandleFunc("GET /analytics/attribution", h.handleAnalyticsAttribution)
	mux.HandleFunc("/analytics/feature-sample", h.handleAnalyticsFeatureSample)

	mux.HandleFunc("GET /analytics/rules/{rule_id}", h.handleAnalyticsRuleStats)
	mux.HandleFunc("/analytics/transactions/search", h.handleSearchTransactions)

	// Decision Explorer: list and search inference events
	mux.HandleFunc("GET /decisions", h.handleListDecisions)
	// Decision Detail: fetch full event with inputs
	mux.HandleFunc("GET /decisions/{request_id}", h.handleGetDecision)
	// Decision Trace: fetch rule execution steps
	mux.HandleFunc("GET /decisions/{request_id}/trace", h.handleGetDecisionTrace)
	// Rule Impact: fetch metrics for a specific rule over time
	mux.HandleFunc("GET /analytics/rules/{rule_id}/impact", h.handleGetRuleImpact)

	mux.HandleFunc("GET /kpis", h.handleGetKpis)
	mux.HandleFunc("GET /volume", h.handleGetVolumeSeries)
	mux.HandleFunc("GET /analytics/confusion-matrix", h.handleGetConfusionMatrix)

	mux.HandleFunc("GET /jobs", h.handleListJobs)
	mux.HandleFunc("GET /jobs/summary", h.handleGetJobSummary)
	mux.HandleFunc("GET /jobs/{id}", h.handleGetJob)
	mux.HandleFunc("GET /jobs/{id}/events", h.handleGetJobEvents)

	mux.HandleFunc("GET /dataset/summary", h.handleGetDatasetProfile) // Alias for latest/default
	mux.HandleFunc("GET /dataset/latest", h.handleGetLatestDatasetProfile)
	mux.HandleFunc("GET /dataset/profiles", h.handleListDatasetProfiles)
	mux.HandleFunc("GET /dataset/profiles/{id}", h.handleGetDatasetProfile)
	mux.HandleFunc("GET /dataset/profiles/compare", h.handleCompareDatasetProfiles)

	mux.HandleFunc("GET /training-runs", h.handleListTrainingRuns)
	mux.HandleFunc("GET /models/versions", h.handleListModelVersions)
	mux.HandleFunc("GET /training-runs/{id}", h.handleGetTrainingRun)
	mux.HandleFunc("GET /metrics/series", h.handleGetMetricSeries)

	mux.HandleFunc("/data/clear", h.handleDatasetClear)
	mux.HandleFunc("POST /data/generate", h.handleDatasetGenerate)
	mux.HandleFunc("/monitoring/drift", h.handleMonitoringDrift)
	mux.HandleFunc("/metrics/shadow/comparison", h.handleMetricsShadowComparison)
	mux.HandleFunc("/backtest/results", h.handleBacktestResults)
	mux.HandleFunc("POST /backtest/compare", h.handleBacktestCompare)
	mux.HandleFunc("GET /rules", h.handleListRules)
	mux.HandleFunc("POST /rules", h.handleCreateRule)
	mux.HandleFunc("GET /rules/{rule_id}", h.handleGetRule)
	mux.HandleFunc("PUT /rules/{rule_id}", h.handleUpdateRule)
	mux.HandleFunc("DELETE /rules/{rule_id}", h.handleDeleteRule)
	// Routes for later commits
	mux.HandleFunc("GET /rules/{rule_id}/history", h.handleListRuleVersions)
	mux.HandleFunc("GET /rules/{rule_id}/versions/{version_id}", h.handleGetRuleVersion)
	mux.HandleFunc("POST /rules/{rule_id}/publish", h.handlePublishRule)
	mux.HandleFunc("GET /rules/{rule_id}/readiness", h.handleRuleReadiness)
	mux.HandleFunc("GET /rules/{rule_id}/diff", h.handleRuleDiff)

	// Native gRPC handled routes
	mux.HandleFunc("POST /train", h.handleTrain)
	mux.HandleFunc("POST /models/deploy", h.handleDeployModel)
	// MLflow Proxy (prefix matching)
	mux.Handle("/api/2.0/mlflow/", proxy.NewHandler(h.mlflowURL, h.logger))
	mux.Handle("/ajax-api/2.0/mlflow/", proxy.NewHandler(h.mlflowURL, h.logger)) // MLflow UI often uses this
	mux.Handle("/static-files/", proxy.NewHandler(h.mlflowURL, h.logger))        // MLflow UI static files

}

func (h *Handler) handleReady(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	components := map[string]string{}
	ready := true

	if h.rulesProvider == nil {
		components["rules"] = "unavailable"
		ready = false
	} else if _, err := h.rulesProvider.GetRules(ctx); err != nil {
		components["rules"] = "error"
		ready = false
	} else {
		components["rules"] = "ok"
	}

	if h.inferenceClient == nil {
		components["inference"] = "unavailable"
		ready = false
	} else if err := h.inferenceClient.Ready(ctx); err != nil {
		components["inference"] = "error"
		ready = false
	} else {
		components["inference"] = "ok"
	}

	status := "ready"
	code := http.StatusOK
	if !ready {
		status = "not_ready"
		code = http.StatusServiceUnavailable
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":     status,
		"components": components,
	})
}

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
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req inferencev1.SignalRequest
	if err := (protojson.UnmarshalOptions{}).Unmarshal(body, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	normalizeSignalRequest(&req)
	if err := validateSignalRequest(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	if h.forecastClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "forecast backend unavailable")
		return
	}

	inferenceResp, err := h.forecastClient.PredictSignal(r.Context(), &forecastv1.PredictSignalRequest{
		UserId:              req.UserId,
		Amount:              req.Amount,
		Currency:            req.Currency,
		ClientTransactionId: req.ClientTransactionId,
	})
	if err != nil {
		writeRPCError(w, err)
		return
	}

	requestID := requestid.FromContext(r.Context())

	// Feature Hydration: Move ownership to Go
	features := map[string]any{}

	// 1. Try to fetch from Analytics
	if h.analyticsClient != nil {
		hydrated, err := h.analyticsClient.GetFeatures(r.Context(), req.UserId)
		if err != nil {
			h.logger.Warn("failed to hydrate features from analytics", "error", err, "user_id", req.UserId)
		} else if hydrated != nil {
			features = hydrated
		} else {
			// 2. Fallback to simulation for unknown users (matching Python behavior)
			features = simulateFeatures(req.UserId, req.Amount)
		}
	}

	// 3. Prediction features are not returned in PredictSignal for now
	// to simplify the migration and reduce payload size if not needed for UI.
	// If needed later, we can add a Features field to PredictSignalResponse.

	// Add transaction specific features
	features["transaction_amount"] = req.Amount

	ruleset, err := h.rulesProvider.GetRules(r.Context())
	if err != nil {
		h.logger.Warn("failed to load ruleset", "error", err)
		ruleset = rules.RuleSet{}
	}

	rawScore := int32(math.Round(inferenceResp.GetProbability() * 100))

	ruleResult, err := rules.EvaluateRules(features, int(rawScore), &ruleset, rules.EvalOptions{Debug: false})
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "rule evaluation failed")
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

	// Persist inference event to CRUD (fire-and-forget, don't fail the request)
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

		evt := inferenceLogEvent{
			ctx:         context.Background(),
			requestID:   requestID,
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

	writeProtoJSON(w, response)
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

func writeProtoJSON(w http.ResponseWriter, msg *inferencev1.SignalResponse) {
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

func buildRiskComponents(features map[string]any) []*inferencev1.RiskComponent {
	components := []*inferencev1.RiskComponent{}

	if toFloat(features["velocity_24h"]) > 5 {
		components = append(components, &inferencev1.RiskComponent{Key: "velocity", Label: "high_transaction_velocity"})
	}
	if toFloat(features["amount_to_avg_ratio_30d"]) > 3.0 {
		components = append(components, &inferencev1.RiskComponent{Key: "amount_ratio", Label: "unusual_transaction_amount"})
	}
	if toFloat(features["balance_volatility_z_score"]) < -2.0 {
		components = append(components, &inferencev1.RiskComponent{Key: "balance", Label: "low_balance_volatility"})
	}
	if toFloat(features["bank_connections_24h"]) > 4 {
		components = append(components, &inferencev1.RiskComponent{Key: "connections", Label: "connection_burst_detected"})
	}
	if toFloat(features["merchant_risk_score"]) > 70 {
		components = append(components, &inferencev1.RiskComponent{Key: "merchant", Label: "high_risk_merchant"})
	}
	if hasHistory, ok := features["has_history"].(bool); ok && !hasHistory {
		components = append(components, &inferencev1.RiskComponent{Key: "history", Label: "insufficient_history"})
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

func (h *Handler) handleTrain(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.trainingClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "training backend unavailable")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req trainingv1.TrainRequest
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(body, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	resp, err := h.trainingClient.Train(r.Context(), &req)
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *Handler) handleDeployModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.forecastClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "forecast backend unavailable")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req forecastv1.DeployModelRequest
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(body, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	resp, err := h.forecastClient.DeployModel(r.Context(), &req)
	if err != nil {
		writeRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

func simulateFeatures(userID string, amount float64) map[string]any {
	// Deterministic hash matching Python's approach
	h := fnv.New32a()
	_, _ = h.Write([]byte(userID))
	userHash := int(h.Sum32() % 1000)

	velocity := (userHash % 10) + 1
	amountRatio := 0.5 + float64(userHash%50)/10.0
	balanceZ := -3.0 + float64(userHash%60)/10.0
	connections := userHash % 8
	merchantRisk := userHash % 100

	return map[string]any{
		"velocity_24h":               float64(velocity),
		"amount_to_avg_ratio_30d":    amountRatio,
		"balance_volatility_z_score": balanceZ,
		"bank_connections_24h":       float64(connections),
		"merchant_risk_score":        float64(merchantRisk),
		"has_history":                false,
		"transaction_amount":         amount,
	}
}

type generateDataRequest struct {
	NumUsers     int32   `json:"num_users"`
	FraudRate    float64 `json:"fraud_rate"`
	DropExisting bool    `json:"drop_existing"`
	Seed         *int64  `json:"seed"`
}

type generateDataResponse struct {
	Success              bool   `json:"success"`
	TotalRecords         int64  `json:"total_records"`
	FraudRecords         int64  `json:"fraud_records"`
	FeaturesMaterialized int64  `json:"features_materialized"`
	Error                string `json:"error,omitempty"`
}

func (h *Handler) handleDatasetGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req generateDataRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	// Validate (basic)
	if req.NumUsers <= 0 {
		writeJSONError(w, http.StatusBadRequest, "num_users must be > 0")
		return
	}
	if req.FraudRate < 0 || req.FraudRate > 1.0 {
		writeJSONError(w, http.StatusBadRequest, "fraud_rate must be between 0 and 1")
		return
	}

	grpcReq := &crudv1.GenerateDataRequest{
		NumUsers:     req.NumUsers,
		FraudRate:    req.FraudRate,
		DropExisting: req.DropExisting,
		Seed:         req.Seed,
	}

	// Long timeout context handling should be in client or here?
	// The client implementation uses a 5x timeout multiplier.
	// But the handler context comes from request, which might be canceled by value in http.Server ReadTimeout/WriteTimeout?
	// We'll rely on client to handle logic, but we should make sure request context doesn't time out prematurely if set.
	// For now pass r.Context()

	resp, err := h.analyticsClient.GenerateData(r.Context(), grpcReq)
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(generateDataResponse{
		Success:              resp.Success,
		TotalRecords:         resp.TotalRecords,
		FraudRecords:         resp.FraudRecords,
		FeaturesMaterialized: resp.FeaturesMaterialized,
		Error:                resp.Error,
	})
}
