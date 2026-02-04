package grpc

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	gatewayv1 "github.com/jonkmatsumo/label-lag/src/services/inference-gateway/internal/http/gatewayv1/gateway/v1"
	"google.golang.org/protobuf/encoding/protojson"
)

type LegacyClient struct {
	baseURL    string
	httpClient *http.Client
}

func NewLegacyClient(baseURL string, timeout time.Duration) *LegacyClient {
	if baseURL == "" {
		baseURL = "http://api:8000"
	}
	if timeout == 0 {
		timeout = 5 * time.Second
	}
	return &LegacyClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *LegacyClient) Evaluate(ctx context.Context, req *gatewayv1.SignalRequest) (*gatewayv1.SignalResponse, error) {
	url := fmt.Sprintf("%s/evaluate/signal", c.baseURL)

	payload, err := protojson.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(payload))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var signalResp gatewayv1.SignalResponse
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(body, &signalResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &signalResp, nil
}