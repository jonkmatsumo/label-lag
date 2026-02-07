package grpc

import (
	"context"

	gatewayv1 "github.com/jonkmatsumo/label-lag/go/inference/internal/http/gatewayv1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GatewayServer implements the GatewayService gRPC interface.
type GatewayServer struct {
	gatewayv1.UnimplementedGatewayServiceServer
}

// NewGatewayServer creates a new instance of GatewayServer.
func NewGatewayServer() *GatewayServer {
	return &GatewayServer{}
}

// Register registers the GatewayServer with the gRPC server.
func (s *GatewayServer) Register(server *grpc.Server) {
	gatewayv1.RegisterGatewayServiceServer(server, s)
}

// EvaluateRules implements the EvaluateRules RPC.
func (s *GatewayServer) EvaluateRules(ctx context.Context, req *gatewayv1.EvaluateRulesRequest) (*gatewayv1.EvaluateRulesResponse, error) {
	// TODO: Phase 2 - Implement logic
	return nil, status.Error(codes.Unimplemented, "EvaluateRules is not yet implemented")
}

// EvaluateRulesDiff implements the EvaluateRulesDiff RPC.
func (s *GatewayServer) EvaluateRulesDiff(ctx context.Context, req *gatewayv1.EvaluateRulesDiffRequest) (*gatewayv1.EvaluateRulesDiffResponse, error) {
	// TODO: Phase 2 - Implement logic
	return nil, status.Error(codes.Unimplemented, "EvaluateRulesDiff is not yet implemented")
}
