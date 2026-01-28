package grpc

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/grpc/inferencev1"
	httpserver "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/http"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

const defaultTimeout = 5 * time.Second

type InferenceClient struct {
	target  string
	timeout time.Duration
	conn    *grpc.ClientConn
	stub    inferencev1.InferenceServiceClient
}

func NewInferenceClient(target string, timeout time.Duration) (*InferenceClient, error) {
	if target == "" {
		target = os.Getenv("PYTHON_INFERENCE_TARGET")
	}
	if target == "" {
		target = "localhost:9001"
	}
	if timeout == 0 {
		timeout = defaultTimeout
	}

	conn, err := grpc.Dial(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial python inference target: %w", err)
	}

	return &InferenceClient{
		target:  target,
		timeout: timeout,
		conn:    conn,
		stub:    inferencev1.NewInferenceServiceClient(conn),
	}, nil
}

func (c *InferenceClient) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *InferenceClient) Score(ctx context.Context, req *inferencev1.ScoreRequest) (*inferencev1.ScoreResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}
	if req.RequestId == "" {
		req.RequestId = httpserver.RequestIDFromContext(ctx)
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	resp, err := c.stub.Score(callCtx, req)
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
