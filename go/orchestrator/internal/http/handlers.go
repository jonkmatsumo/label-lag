package httpserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"log/slog"
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
	GetHealth(ctx context.Context, req *trainingv1.GetHealthRequest) (*trainingv1.GetHealthResponse, error)
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
	workerCtx       context.Context
	workerCancel    context.CancelFunc

	// Last-known-good ruleset cache for fallback on provider failure.
	lastGoodRuleset *rules.RuleSet
	rulesetMu       sync.RWMutex
}

type inferenceLogEvent struct {
	ctx         context.Context
	req         *crudv1.LogInferenceEventRequest
	spanContext trace.SpanContext
	requestID   string
	tenantID    string
}

type HandlerOptions struct {
	Logger          *slog.Logger
	InferenceClient InferenceClient
	AnalyticsClient AnalyticsClient
	TrainingClient  TrainingClient
	ForecastClient  ForecastClient
	RulesProvider   rules.Provider
	MaxBodyBytes    int64
	PythonURL       string
	MlflowURL       string
}

func NewHandler(opts HandlerOptions) *Handler {
	if opts.MaxBodyBytes <= 0 {
		opts.MaxBodyBytes = 1 << 20
	}
	workerCtx, workerCancel := context.WithCancel(context.Background())
	h := &Handler{
		logger:          opts.Logger,
		inferenceClient: opts.InferenceClient,
		analyticsClient: opts.AnalyticsClient,
		trainingClient:  opts.TrainingClient,
		forecastClient:  opts.ForecastClient,
		rulesProvider:   opts.RulesProvider,
		maxBodyBytes:    opts.MaxBodyBytes,
		pythonURL:       opts.PythonURL,
		mlflowURL:       opts.MlflowURL,
		logQueue:        make(chan inferenceLogEvent, 100),
		workerCtx:       workerCtx,
		workerCancel:    workerCancel,
	}
	handlerLogQueueCapacity.Set(100)
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
		handlerLogQueueDepth.Set(float64(len(h.logQueue)))
		ctx := event.ctx
		// Create a link to the parent trace
		opts := []trace.SpanStartOption{
			trace.WithLinks(trace.Link{SpanContext: event.spanContext}),
			trace.WithAttributes(
				attribute.String("request_id", event.requestID),
				attribute.String("tenant_id", event.tenantID),
			),
		}

		// Start a new root span for the async operation, linked to parent
		var span trace.Span
		if event.spanContext.IsValid() {
			ctx, span = otel.Tracer("async-logger").Start(ctx, "async_log_inference", opts...)
		} else {
			ctx, span = otel.Tracer("async-logger").Start(ctx, "async_log_inference")
		}

		rpcCtx, rpcCancel := context.WithTimeout(h.workerCtx, 5*time.Second)
		// Preserve span context from the trace-aware ctx
		rpcCtx = trace.ContextWithSpan(rpcCtx, span)

		_, err := h.analyticsClient.DirectLogInferenceEvent(rpcCtx, event.req)
		rpcCancel()
		if err != nil {
			h.logger.Warn("failed to log inference event to CRUD", "error", err, "request_id", event.requestID)
			grpcclient.LogEventsDropped.WithLabelValues("handler", "send_error").Inc()
		}

		span.End()
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
		// Cancel in-flight RPCs so workers can exit
		h.workerCancel()
		// Record remaining items as shutdown drops
		remaining := len(h.logQueue)
		if remaining > 0 {
			grpcclient.LogEventsDropped.WithLabelValues("handler", "shutdown").Add(float64(remaining))
		}
		// Give workers a brief grace period to finish
		select {
		case <-done:
		case <-time.After(500 * time.Millisecond):
		}
		return ctx.Err()
	}
}

func (h *Handler) cacheRuleset(rs *rules.RuleSet) {
	h.rulesetMu.Lock()
	defer h.rulesetMu.Unlock()
	h.lastGoodRuleset = rs
}

func (h *Handler) getLastGoodRuleset() *rules.RuleSet {
	h.rulesetMu.RLock()
	defer h.rulesetMu.RUnlock()
	return h.lastGoodRuleset
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
	mux.HandleFunc("POST /jobs/{id}/cancel", h.handleCancelJob)
	mux.HandleFunc("POST /jobs/{id}/retry", h.handleRetryJob)

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

	if h.trainingClient == nil {
		components["training"] = "unavailable"
		ready = false
	} else if resp, err := h.trainingClient.GetHealth(ctx, &trainingv1.GetHealthRequest{}); err != nil {
		components["training"] = "error"
		ready = false
	} else {
		components["training"] = "ok"
		if resp.SpooledReportsCount > 0 {
			components["training_spool"] = fmt.Sprintf("%d reports pending", resp.SpooledReportsCount)
		}
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

func writeError(w http.ResponseWriter, r *http.Request, status int, code string, message string, details any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	requestID := requestid.FromContext(r.Context())
	if requestID == "" {
		requestID = "unknown"
	}

	resp := map[string]any{
		"error": map[string]any{
			"code":       code,
			"message":    message,
			"details":    details,
			"request_id": requestID,
		},
	}

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		// Fallback if JSON fails, though unlikely for map[string]any
		slog.Error("failed to encode error response", "error", err)
	}
}

func writeJSONError(w http.ResponseWriter, r *http.Request, status int, message string) {
	// Map status to a generic code if possible, or use status text
	code := http.StatusText(status)
	code = strings.ToUpper(strings.ReplaceAll(code, " ", "_"))
	writeError(w, r, status, code, message, nil)
}

func writeRPCError(w http.ResponseWriter, r *http.Request, err error) {
	var rpcErr *grpcclient.RPCError
	if errors.As(err, &rpcErr) {
		switch rpcErr.Code {
		case codes.InvalidArgument:
			writeError(w, r, http.StatusBadRequest, "INVALID_ARGUMENT", rpcErr.Message, nil)
		case codes.DeadlineExceeded, codes.Unavailable:
			writeError(w, r, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE", "inference backend timeout", nil)
		case codes.AlreadyExists:
			writeError(w, r, http.StatusConflict, "ALREADY_EXISTS", rpcErr.Message, nil)
		case codes.NotFound:
			writeError(w, r, http.StatusNotFound, "NOT_FOUND", rpcErr.Message, nil)
		case codes.PermissionDenied:
			writeError(w, r, http.StatusForbidden, "PERMISSION_DENIED", rpcErr.Message, nil)
		case codes.Unauthenticated:
			writeError(w, r, http.StatusUnauthorized, "UNAUTHENTICATED", rpcErr.Message, nil)
		default:
			writeError(w, r, http.StatusBadGateway, "UPSTREAM_ERROR", rpcErr.Message, nil)
		}
		return
	}
	writeError(w, r, http.StatusBadGateway, "INTERNAL_ERROR", "inference backend error", nil)
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

func (h *Handler) validatePaginationParams(w http.ResponseWriter, r *http.Request) error {
	if r.URL.Query().Get("cursor") != "" && r.URL.Query().Get("offset") != "" {
		h.logger.Warn("pagination contract violation: both cursor and offset provided",
			"path", r.URL.Path,
			"user_agent", r.UserAgent(),
		)
		// Increment contract violation metric
		_ = h.incrementContractViolation(r.Context(), "pagination_params_conflict")

		writeJSONError(w, r, http.StatusBadRequest, "cannot provide both cursor and offset")
		return errors.New("pagination contract violation")
	}
	return nil
}

func (h *Handler) incrementContractViolation(ctx context.Context, violationType string) error {
	// Simple counter increment - in real implementation this would use OTEL meter
	// For now just logging is sufficient for observablity requirement
	return nil
}

func (h *Handler) handleTrain(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.trainingClient == nil {
		writeJSONError(w, r, http.StatusServiceUnavailable, "training backend unavailable")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req trainingv1.TrainRequest
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(body, &req); err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "invalid json payload")
		return
	}
	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}
	req.TenantId = tenantID

	resp, err := h.trainingClient.Train(r.Context(), &req)
	if err != nil {
		writeRPCError(w, r, err)
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
		writeJSONError(w, r, http.StatusServiceUnavailable, "forecast backend unavailable")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req forecastv1.DeployModelRequest
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(body, &req); err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "invalid json payload")
		return
	}
	tenantID, err := mustTenantID(r)
	if err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "missing X-Tenant-Id")
		return
	}
	req.TenantId = tenantID

	resp, err := h.forecastClient.DeployModel(r.Context(), &req)
	if err != nil {
		writeRPCError(w, r, err)
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
		writeJSONError(w, r, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req generateDataRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSONError(w, r, http.StatusBadRequest, "invalid json payload")
		return
	}

	// Validate (basic)
	if req.NumUsers <= 0 {
		writeJSONError(w, r, http.StatusBadRequest, "num_users must be > 0")
		return
	}
	if req.FraudRate < 0 || req.FraudRate > 1.0 {
		writeJSONError(w, r, http.StatusBadRequest, "fraud_rate must be between 0 and 1")
		return
	}

	grpcReq := &crudv1.GenerateDataRequest{
		NumUsers:       req.NumUsers,
		FraudRate:      req.FraudRate,
		DropExisting:   req.DropExisting,
		Seed:           req.Seed,
		IdempotencyKey: requestid.FromContext(r.Context()), // Assuming we should add IdempotencyKey based on proto definition or just remove TenantId?
	}

	// Long timeout context handling should be in client or here?
	// The client implementation uses a 5x timeout multiplier.
	// But the handler context comes from request, which might be canceled by value in http.Server ReadTimeout/WriteTimeout?
	// We'll rely on client to handle logic, but we should make sure request context doesn't time out prematurely if set.
	// For now pass r.Context()

	resp, err := h.analyticsClient.GenerateData(r.Context(), grpcReq)
	if err != nil {
		writeAnalyticsRPCError(w, r, err)
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
