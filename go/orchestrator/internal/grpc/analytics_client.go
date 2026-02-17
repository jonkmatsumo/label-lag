package grpc

import (
	"context"
	"fmt"
	"os"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
)

var (
	inferenceEventsDropped = promauto.NewCounter(prometheus.CounterOpts{
		Name: "orchestrator_inference_events_dropped_total",
		Help: "Total number of inference events dropped due to full queue.",
	})

	analyticsQueueDepth = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "orchestrator_analytics_log_queue_depth",
		Help: "Current number of pending events in the analytics log queue.",
	})

	analyticsQueueCapacity = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "orchestrator_analytics_log_queue_capacity",
		Help: "Maximum capacity of the analytics log queue.",
	})
)

const (
	defaultQueueSize = 1000
)

type AnalyticsClient struct {
	target   string
	timeout  time.Duration
	conn     *grpc.ClientConn
	stub     crudv1.AnalyticsServiceClient
	breaker  *CircuitBreaker
	logQueue chan *crudv1.LogInferenceEventRequest
	stop     chan struct{}
}

func NewAnalyticsClient(target string, timeout time.Duration) (*AnalyticsClient, error) {
	if target == "" {
		target = os.Getenv("ANALYTICS_CRUD_TARGET")
	}
	if target == "" {
		target = "analytics-crud:50051"
	}
	if timeout == 0 {
		timeout = defaultTimeout
	}

	kacp := keepalive.ClientParameters{
		Time:                10 * time.Second,
		Timeout:             time.Second,
		PermitWithoutStream: true,
	}

	conn, err := grpc.Dial(target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(kacp),
		grpc.WithDefaultServiceConfig(`{"loadBalancingConfig": [{"round_robin":{}}]}`),
	)
	if err != nil {
		return nil, fmt.Errorf("dial analytics-crud target: %w", err)
	}

	client := &AnalyticsClient{
		target:   target,
		timeout:  timeout,
		conn:     conn,
		stub:     crudv1.NewAnalyticsServiceClient(conn),
		breaker:  NewCircuitBreakerWithPrefix("analytics"),
		logQueue: make(chan *crudv1.LogInferenceEventRequest, defaultQueueSize),
		stop:     make(chan struct{}),
	}
	analyticsQueueCapacity.Set(float64(defaultQueueSize))
	client.startWorker()

	return client, nil
}

func (c *AnalyticsClient) Close() error {
	if c.stop != nil {
		close(c.stop)
	}
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *AnalyticsClient) startWorker() {
	go func() {
		for {
			select {
			case req := <-c.logQueue:
				analyticsQueueDepth.Set(float64(len(c.logQueue)))
				// Use context.Background() since the original request context may have been cancelled
				ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
				_, _ = c.stub.LogInferenceEvent(ctx, req)
				cancel()
			case <-c.stop:
				return
			}
		}
	}()
}

func (c *AnalyticsClient) withMetadata(ctx context.Context) context.Context {
	if rid := requestid.FromContext(ctx); rid != "" {
		return metadata.AppendToOutgoingContext(ctx, "x-request-id", rid)
	}
	return ctx
}

func (c *AnalyticsClient) SearchTransactions(ctx context.Context, req *crudv1.SearchTransactionsRequest) (*crudv1.SearchTransactionsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.SearchTransactions(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetDailyStats(ctx context.Context, req *crudv1.GetDailyStatsRequest) (*crudv1.GetDailyStatsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetDailyStats(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetOverviewMetrics(ctx context.Context, req *crudv1.GetOverviewMetricsRequest) (*crudv1.GetOverviewMetricsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetOverviewMetrics(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetTransactionDetails(ctx context.Context, req *crudv1.GetTransactionDetailsRequest) (*crudv1.GetTransactionDetailsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetTransactionDetails(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRecentAlerts(ctx context.Context, req *crudv1.GetRecentAlertsRequest) (*crudv1.GetRecentAlertsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRecentAlerts(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetFeatureSample(ctx context.Context, req *crudv1.GetFeatureSampleRequest) (*crudv1.GetFeatureSampleResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetFeatureSample(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListBacktestResults(ctx context.Context, req *crudv1.ListBacktestResultsRequest) (*crudv1.ListBacktestResultsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListBacktestResults(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ClearAllData(ctx context.Context, req *crudv1.ClearAllDataRequest) (*crudv1.ClearAllDataResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ClearAllData(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetFeatures(ctx context.Context, userID string, tenantID string) (map[string]any, error) {
	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetLatestUserFeatures(c.withMetadata(callCtx), &crudv1.GetLatestUserFeaturesRequest{
		UserId:   userID,
		TenantId: tenantID,
	})
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}

	if !resp.GetFound() {
		return nil, nil // Not found
	}

	f := resp.GetFeatures()
	features := map[string]any{
		"velocity_24h":               float64(f.GetVelocity_24H()),
		"amount_to_avg_ratio_30d":    f.GetAmountToAvgRatio_30D(),
		"balance_volatility_z_score": f.GetBalanceVolatilityZScore(),
		"bank_connections_24h":       float64(f.GetBankConnections_24H()),
		"merchant_risk_score":        float64(f.GetMerchantRiskScore()),
		"has_history":                f.GetHasHistory(),
	}

	return features, nil
}

func (c *AnalyticsClient) ListRuleVersions(ctx context.Context, req *crudv1.ListRuleVersionsRequest) (*crudv1.ListRuleVersionsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListRuleVersions(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRuleVersion(ctx context.Context, req *crudv1.GetRuleVersionRequest) (*crudv1.GetRuleVersionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRuleVersion(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) PublishRuleVersion(ctx context.Context, req *crudv1.PublishRuleVersionRequest) (*crudv1.PublishRuleVersionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.PublishRuleVersion(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRuleReadiness(ctx context.Context, req *crudv1.GetRuleReadinessRequest) (*crudv1.GetRuleReadinessResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRuleReadiness(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) DiffRuleVersions(ctx context.Context, req *crudv1.DiffRuleVersionsRequest) (*crudv1.DiffRuleVersionsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.DiffRuleVersions(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) SaveRule(ctx context.Context, req *crudv1.SaveRuleRequest) (*crudv1.SaveRuleResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.SaveRule(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRule(ctx context.Context, req *crudv1.GetRuleRequest) (*crudv1.GetRuleResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRule(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListRules(ctx context.Context, req *crudv1.ListRulesRequest) (*crudv1.ListRulesResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListRules(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) DeleteRule(ctx context.Context, req *crudv1.DeleteRuleRequest) (*crudv1.DeleteRuleResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.DeleteRule(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRuleStats(ctx context.Context, req *crudv1.GetRuleStatsRequest) (*crudv1.GetRuleStatsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRuleStats(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetAttribution(ctx context.Context, req *crudv1.GetAttributionRequest) (*crudv1.GetAttributionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetAttribution(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) DirectLogInferenceEvent(ctx context.Context, req *crudv1.LogInferenceEventRequest) (*crudv1.LogInferenceEventResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.LogInferenceEvent(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (c *AnalyticsClient) LogInferenceEvent(ctx context.Context, req *crudv1.LogInferenceEventRequest) (*crudv1.LogInferenceEventResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	// Push to queue non-blocking
	select {
	case c.logQueue <- req:
		analyticsQueueDepth.Set(float64(len(c.logQueue)))
	default:
		// Queue full, drop event and increment counter
		inferenceEventsDropped.Inc()
	}

	// Always return success to the caller to avoid blocking inference
	return &crudv1.LogInferenceEventResponse{Success: true}, nil
}

func (c *AnalyticsClient) CompareBacktests(ctx context.Context, req *crudv1.CompareBacktestsRequest) (*crudv1.CompareBacktestsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.CompareBacktests(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetShadowComparison(ctx context.Context, req *crudv1.GetShadowComparisonRequest) (*crudv1.GetShadowComparisonResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetShadowComparison(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListDecisions(ctx context.Context, req *crudv1.ListDecisionsRequest) (*crudv1.ListDecisionsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListDecisions(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetDecision(ctx context.Context, req *crudv1.GetDecisionRequest) (*crudv1.GetDecisionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetDecision(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetDecisionTrace(ctx context.Context, req *crudv1.GetDecisionTraceRequest) (*crudv1.GetDecisionTraceResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetDecisionTrace(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRuleImpact(ctx context.Context, req *crudv1.GetRuleImpactRequest) (*crudv1.GetRuleImpactResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRuleImpact(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetKpis(ctx context.Context, req *crudv1.GetKpisRequest) (*crudv1.GetKpisResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetKpis(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetVolumeSeries(ctx context.Context, req *crudv1.GetVolumeSeriesRequest) (*crudv1.GetVolumeSeriesResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetVolumeSeries(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetConfusionMatrix(ctx context.Context, req *crudv1.GetConfusionMatrixRequest) (*crudv1.GetConfusionMatrixResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetConfusionMatrix(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListJobs(ctx context.Context, req *crudv1.ListJobsRequest) (*crudv1.ListJobsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListJobs(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetJob(ctx context.Context, req *crudv1.GetJobRequest) (*crudv1.GetJobResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetJob(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetJobEvents(ctx context.Context, req *crudv1.GetJobEventsRequest) (*crudv1.GetJobEventsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetJobEvents(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetDatasetSummary(ctx context.Context, req *crudv1.GetDatasetSummaryRequest) (*crudv1.GetDatasetSummaryResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetDatasetSummary(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListDatasetProfiles(ctx context.Context, req *crudv1.ListDatasetProfilesRequest) (*crudv1.ListDatasetProfilesResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListDatasetProfiles(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) CompareDatasetProfiles(ctx context.Context, req *crudv1.CompareDatasetProfilesRequest) (*crudv1.CompareDatasetProfilesResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.CompareDatasetProfiles(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListTrainingRuns(ctx context.Context, req *crudv1.ListTrainingRunsRequest) (*crudv1.ListTrainingRunsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListTrainingRuns(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetTrainingRun(ctx context.Context, req *crudv1.GetTrainingRunRequest) (*crudv1.GetTrainingRunResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetTrainingRun(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetMetricSeries(ctx context.Context, req *crudv1.GetMetricSeriesRequest) (*crudv1.GetMetricSeriesResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetMetricSeries(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetJobSummary(ctx context.Context, req *crudv1.GetJobSummaryRequest) (*crudv1.GetJobSummaryResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetJobSummary(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListModelVersions(ctx context.Context, req *crudv1.ListModelVersionsRequest) (*crudv1.ListModelVersionsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListModelVersions(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetLatestDatasetProfile(ctx context.Context, req *crudv1.GetLatestDatasetProfileRequest) (*crudv1.GetLatestDatasetProfileResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetLatestDatasetProfile(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}
func (c *AnalyticsClient) CancelJob(ctx context.Context, req *crudv1.CancelJobRequest) (*crudv1.CancelJobResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.CancelJob(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) RetryJob(ctx context.Context, req *crudv1.RetryJobRequest) (*crudv1.RetryJobResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("analytics.breaker_state", c.breaker.State().String()),
			attribute.String("analytics.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.RetryJob(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("analytics.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}
