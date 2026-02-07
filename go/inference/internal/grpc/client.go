package grpc

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"sync"
	"time"

	inferencev1 "github.com/jonkmatsumo/label-lag/go/inference/internal/grpc/inferencev1"
	"github.com/jonkmatsumo/label-lag/go/inference/internal/requestid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
)

const (
	defaultTimeout = 5 * time.Second
	// R3: Breaker defaults
	defaultBreakerThreshold = 5
	defaultBreakerTimeout   = 10 * time.Second
)

type BreakerState int

const (
	StateClosed BreakerState = iota
	StateOpen
	StateHalfOpen
)

func (s BreakerState) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

type CircuitBreaker struct {
	mu               sync.RWMutex
	state            BreakerState
	failureCount     int
	failureThreshold int
	resetTimeout     time.Duration
	lastFailureTime  time.Time
}

func NewCircuitBreaker() *CircuitBreaker {
	threshold := defaultBreakerThreshold
	if val := os.Getenv("INFERENCE_BREAKER_FAILURE_THRESHOLD"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			threshold = i
		}
	}
	timeout := defaultBreakerTimeout
	if val := os.Getenv("INFERENCE_BREAKER_RESET_TIMEOUT_MS"); val != "" {
		if ms, err := strconv.Atoi(val); err == nil {
			timeout = time.Duration(ms) * time.Millisecond
		}
	}

	return &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: threshold,
		resetTimeout:     timeout,
	}
}

func (cb *CircuitBreaker) State() BreakerState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

func (cb *CircuitBreaker) Allow() error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.state == StateOpen {
		if time.Since(cb.lastFailureTime) > cb.resetTimeout {
			cb.state = StateHalfOpen
			slog.Info("inference circuit breaker: half-open", "reason", "cooldown elapsed")
			return nil
		}
		return status.Error(codes.Unavailable, "circuit breaker is open")
	}
	return nil
}

func (cb *CircuitBreaker) RecordResult(err error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err == nil {
		if cb.state == StateHalfOpen {
			slog.Info("inference circuit breaker: closed", "reason", "success in half-open")
		}
		cb.state = StateClosed
		cb.failureCount = 0
		return
	}

	cb.failureCount++
	cb.lastFailureTime = time.Now()

	if cb.state == StateHalfOpen || cb.failureCount >= cb.failureThreshold {
		if cb.state != StateOpen {
			slog.Warn("inference circuit breaker: open", "failures", cb.failureCount, "error", err)
		}
		cb.state = StateOpen
	}
}

type InferenceClient struct {
	target  string
	timeout time.Duration
	conn    *grpc.ClientConn
	stub    inferencev1.InferenceServiceClient
	breaker *CircuitBreaker
}

func NewInferenceClient(target string, timeout time.Duration) (*InferenceClient, error) {
	if target == "" {
		target = os.Getenv("INFERENCE_GATEWAY_PYTHON_GRPC_ADDR")
	}
	if target == "" {
		target = "inference-server:50052"
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
		return nil, fmt.Errorf("dial python inference target: %w", err)
	}

	return &InferenceClient{
		target:  target,
		timeout: timeout,
		conn:    conn,
		stub:    inferencev1.NewInferenceServiceClient(conn),
		breaker: NewCircuitBreaker(),
	}, nil
}

func (c *InferenceClient) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *InferenceClient) Ready(ctx context.Context) error {
	if c == nil || c.conn == nil {
		return fmt.Errorf("inference client not initialized")
	}

	state := c.conn.GetState()
	if state != connectivity.Ready {
		c.conn.Connect()
		if !c.conn.WaitForStateChange(ctx, state) {
			return fmt.Errorf("inference backend not ready")
		}
		if c.conn.GetState() != connectivity.Ready {
			return fmt.Errorf("inference backend not ready")
		}
	}
	return nil
}

func (c *InferenceClient) Score(ctx context.Context, req *inferencev1.ScoreRequest) (*inferencev1.ScoreResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	span := trace.SpanFromContext(ctx)
	if err := c.breaker.Allow(); err != nil {
		span.SetAttributes(
			attribute.String("inference.breaker_state", c.breaker.State().String()),
			attribute.String("inference.breaker_open_reason", "cooldown"),
		)
		return nil, mapRPCError(err)
	}

	if req.RequestId == "" {
		req.RequestId = requestid.FromContext(ctx)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.Score(callCtx, req)
	c.breaker.RecordResult(err)

	span.SetAttributes(
		attribute.String("inference.breaker_state", c.breaker.State().String()),
	)

	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}

type RPCError struct {
	Code    codes.Code
	Message string
}

func (e *RPCError) Error() string {
	return fmt.Sprintf("rpc error: %s", e.Message)
}

func mapRPCError(err error) error {
	st, ok := status.FromError(err)
	if !ok {
		return err
	}
	return &RPCError{Code: st.Code(), Message: st.Message()}
}
