package grpc

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/jonkmatsumo/label-lag/go/orchestrator/internal/requestid"
	trainingv1 "github.com/jonkmatsumo/label-lag/go/training/proto/trainingv1"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
)

type TrainingClient struct {
	target  string
	timeout time.Duration
	conn    *grpc.ClientConn
	stub    trainingv1.TrainingServiceClient
	breaker *CircuitBreaker
}

func NewTrainingClient(target string, timeout time.Duration) (*TrainingClient, error) {
	if target == "" {
		target = os.Getenv("TRAINING_SERVER_GRPC_ADDR")
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
		return nil, fmt.Errorf("dial training-server target: %w", err)
	}

	return &TrainingClient{
		target:  target,
		timeout: timeout,
		conn:    conn,
		stub:    trainingv1.NewTrainingServiceClient(conn),
		breaker: NewCircuitBreakerWithPrefix("training"),
	}, nil
}

func (c *TrainingClient) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *TrainingClient) withMetadata(ctx context.Context) context.Context {
	if rid := requestid.FromContext(ctx); rid != "" {
		return metadata.AppendToOutgoingContext(ctx, "x-request-id", rid)
	}
	return ctx
}

func (c *TrainingClient) Train(ctx context.Context, req *trainingv1.TrainRequest) (*trainingv1.TrainResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("training.breaker_state", c.breaker.State().String()),
			attribute.String("training.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.Train(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("training.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *TrainingClient) ClearData(ctx context.Context, req *trainingv1.ClearDataRequest) (*trainingv1.ClearDataResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("training.breaker_state", c.breaker.State().String()),
			attribute.String("training.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.ClearData(c.withMetadata(callCtx), req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("training.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

func (c *TrainingClient) GetHealth(ctx context.Context, req *trainingv1.GetHealthRequest) (*trainingv1.GetHealthResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("training.breaker_state", c.breaker.State().String()),
			attribute.String("training.breaker_open_reason", "cooldown"),
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
		attribute.String("training.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}
