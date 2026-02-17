package grpc

import (
	"context"
	"fmt"
	"os"
	"time"

	forecastv1 "github.com/jonkmatsumo/label-lag/go/forecast/proto/forecastv1"
	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
)

type ForecastClient struct {
	target  string
	timeout time.Duration
	conn    *grpc.ClientConn
	stub    forecastv1.ForecastServiceClient
	breaker *CircuitBreaker
}

func NewForecastClient(target string, timeout time.Duration) (*ForecastClient, error) {
	if target == "" {
		target = os.Getenv("TRAINING_SERVER_GRPC_ADDR") // Same server as training
	}
	if target == "" {
		target = "training-server:50053"
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
		return nil, fmt.Errorf("dial forecast service target: %w", err)
	}

	return &ForecastClient{
		target:  target,
		timeout: timeout,
		conn:    conn,
		stub:    forecastv1.NewForecastServiceClient(conn),
		breaker: NewCircuitBreakerWithPrefix("FORECAST"),
	}, nil
}

func (c *ForecastClient) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *ForecastClient) withMetadata(ctx context.Context) context.Context {
	if rid := requestid.FromContext(ctx); rid != "" {
		return metadata.AppendToOutgoingContext(ctx, "x-request-id", rid)
	}
	return ctx
}

func (c *ForecastClient) GetHealth(ctx context.Context, req *forecastv1.GetHealthRequest) (*forecastv1.GetHealthResponse, error) {
	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("forecast.breaker_state", c.breaker.State().String()),
			attribute.String("forecast.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetHealth(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("forecast.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *ForecastClient) GetDriftMonitoring(ctx context.Context, req *forecastv1.GetDriftMonitoringRequest) (*forecastv1.GetDriftMonitoringResponse, error) {
	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("forecast.breaker_state", c.breaker.State().String()),
			attribute.String("forecast.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetDriftMonitoring(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("forecast.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *ForecastClient) GetScoreDistribution(ctx context.Context, req *forecastv1.GetScoreDistributionRequest) (*forecastv1.GetScoreDistributionResponse, error) {
	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("forecast.breaker_state", c.breaker.State().String()),
			attribute.String("forecast.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.GetScoreDistribution(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("forecast.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *ForecastClient) ReloadModel(ctx context.Context, req *forecastv1.ReloadModelRequest) (*forecastv1.ReloadModelResponse, error) {
	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("forecast.breaker_state", c.breaker.State().String()),
			attribute.String("forecast.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ReloadModel(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("forecast.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *ForecastClient) DeployModel(ctx context.Context, req *forecastv1.DeployModelRequest) (*forecastv1.DeployModelResponse, error) {
	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("forecast.breaker_state", c.breaker.State().String()),
			attribute.String("forecast.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.DeployModel(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("forecast.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *ForecastClient) PredictSignal(ctx context.Context, req *forecastv1.PredictSignalRequest) (*forecastv1.PredictSignalResponse, error) {
	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("forecast.breaker_state", c.breaker.State().String()),
			attribute.String("forecast.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.PredictSignal(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("forecast.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}
