package grpc

import (
	"context"
	"fmt"
	"os"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/requestid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

type AnalyticsClient struct {
	target  string
	timeout time.Duration
	conn    *grpc.ClientConn
	stub    crudv1.AnalyticsServiceClient
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

	conn, err := grpc.Dial(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial analytics-crud target: %w", err)
	}

	return &AnalyticsClient{
		target:  target,
		timeout: timeout,
		conn:    conn,
		stub:    crudv1.NewAnalyticsServiceClient(conn),
	}, nil
}

func (c *AnalyticsClient) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
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

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.SearchTransactions(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetDailyStats(ctx context.Context, req *crudv1.GetDailyStatsRequest) (*crudv1.GetDailyStatsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetDailyStats(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetOverviewMetrics(ctx context.Context, req *crudv1.GetOverviewMetricsRequest) (*crudv1.GetOverviewMetricsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetOverviewMetrics(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetTransactionDetails(ctx context.Context, req *crudv1.GetTransactionDetailsRequest) (*crudv1.GetTransactionDetailsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetTransactionDetails(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRecentAlerts(ctx context.Context, req *crudv1.GetRecentAlertsRequest) (*crudv1.GetRecentAlertsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRecentAlerts(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetDatasetFingerprint(ctx context.Context, req *crudv1.GetDatasetFingerprintRequest) (*crudv1.GetDatasetFingerprintResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetDatasetFingerprint(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetFeatureSample(ctx context.Context, req *crudv1.GetFeatureSampleRequest) (*crudv1.GetFeatureSampleResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetFeatureSample(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetSchemaSummary(ctx context.Context, req *crudv1.GetSchemaSummaryRequest) (*crudv1.GetSchemaSummaryResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetSchemaSummary(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListBacktestResults(ctx context.Context, req *crudv1.ListBacktestResultsRequest) (*crudv1.ListBacktestResultsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListBacktestResults(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ClearAllData(ctx context.Context, req *crudv1.ClearAllDataRequest) (*crudv1.ClearAllDataResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ClearAllData(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetFeatures(ctx context.Context, userID string) (map[string]any, error) {
	resp, err := c.SearchTransactions(ctx, &crudv1.SearchTransactionsRequest{
		UserId: userID,
		Limit:  1,
	})
	if err != nil {
		return nil, err
	}

	if len(resp.GetTransactions()) == 0 {
		return nil, nil // Not found
	}

	tx := resp.GetTransactions()[0]
	features := map[string]any{
		"velocity_24h":               float64(tx.GetVelocity_24H()),
		"amount_to_avg_ratio_30d":    tx.GetAmountToAvgRatio_30D(),
		"balance_volatility_z_score": tx.GetBalanceVolatilityZScore(),
		"bank_connections_24h":       0.0, // Not yet in Analytics
		"merchant_risk_score":        float64(tx.GetMerchantRiskScore()),
		"has_history":                true,
	}

	return features, nil
}

func (c *AnalyticsClient) ListRuleVersions(ctx context.Context, req *crudv1.ListRuleVersionsRequest) (*crudv1.ListRuleVersionsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListRuleVersions(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRuleVersion(ctx context.Context, req *crudv1.GetRuleVersionRequest) (*crudv1.GetRuleVersionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRuleVersion(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) PublishRuleVersion(ctx context.Context, req *crudv1.PublishRuleVersionRequest) (*crudv1.PublishRuleVersionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.PublishRuleVersion(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRuleReadiness(ctx context.Context, req *crudv1.GetRuleReadinessRequest) (*crudv1.GetRuleReadinessResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRuleReadiness(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) DiffRuleVersions(ctx context.Context, req *crudv1.DiffRuleVersionsRequest) (*crudv1.DiffRuleVersionsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.DiffRuleVersions(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) SaveRule(ctx context.Context, req *crudv1.SaveRuleRequest) (*crudv1.SaveRuleResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.SaveRule(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) GetRule(ctx context.Context, req *crudv1.GetRuleRequest) (*crudv1.GetRuleResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetRule(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) ListRules(ctx context.Context, req *crudv1.ListRulesRequest) (*crudv1.ListRulesResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ListRules(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *AnalyticsClient) DeleteRule(ctx context.Context, req *crudv1.DeleteRuleRequest) (*crudv1.DeleteRuleResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.DeleteRule(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}
