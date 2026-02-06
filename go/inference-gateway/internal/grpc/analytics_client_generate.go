package grpc

import (
	"context"
	"fmt"

	crudv1 "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
)

func (c *AnalyticsClient) GenerateData(ctx context.Context, req *crudv1.GenerateDataRequest) (*crudv1.GenerateDataResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil request")
	}

	callCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.timeout*5) // Generous timeout for generation
		defer cancel()
	}

	resp, err := c.stub.GenerateData(c.withMetadata(callCtx), req)
	if err != nil {
		return nil, mapRPCError(err)
	}
	return resp, nil
}
