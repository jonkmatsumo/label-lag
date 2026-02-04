package grpc

import (
	"context"
	"fmt"
	"os"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
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

	resp, err := c.stub.SearchTransactions(callCtx, req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}
