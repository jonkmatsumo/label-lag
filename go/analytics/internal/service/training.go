package service

import (
	"context"
	"log/slog"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	commonv1 "github.com/jonkmatsumo/label-lag/go/common/proto/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (s *Service) ListTrainingRuns(ctx context.Context, req *pb.ListTrainingRunsRequest) (*pb.ListTrainingRunsResponse, error) {
	limit, truncated := normalizeLimit(req.Limit, 50, 250)
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}
	req.Limit = limit
	req.Offset = offset

	if req.StartDate != nil && req.EndDate != nil && req.StartDate.AsTime().After(req.EndDate.AsTime()) {
		return nil, status.Error(codes.InvalidArgument, "start_date must be <= end_date")
	}

	runs, total, nextCursor, err := s.store.ListTrainingRuns(ctx, req)
	if err != nil {
		return nil, err
	}

	resp := &pb.ListTrainingRunsResponse{
		Runs: runs,
		Pagination: &commonv1.CursorPageResponse{
			NextCursor: nextCursor,
		},
		Meta: &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		},
	}
	if req.Pagination == nil || req.Pagination.Cursor == "" {
		resp.Pagination.Total = &total
	}
	return resp, nil
}

func (s *Service) ListModelVersions(ctx context.Context, req *pb.ListModelVersionsRequest) (*pb.ListModelVersionsResponse, error) {
	if req.ModelName == "" {
		return nil, status.Error(codes.InvalidArgument, "model_name required")
	}

	limit, truncated := normalizeLimit(req.Limit, 50, 250)
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}
	req.Limit = limit
	req.Offset = offset

	versions, total, nextCursor, err := s.store.ListModelVersions(ctx, req)
	if err != nil {
		return nil, err
	}

	resp := &pb.ListModelVersionsResponse{
		Versions: versions,
		Pagination: &commonv1.CursorPageResponse{
			NextCursor: nextCursor,
		},
		Meta: &pb.AnalyticsMeta{
			EffectiveLimit: limit,
			Truncated:      truncated,
		},
	}
	if req.Pagination == nil || req.Pagination.Cursor == "" {
		resp.Pagination.Total = &total
	}
	return resp, nil
}

func (s *Service) GetTrainingRun(ctx context.Context, req *pb.GetTrainingRunRequest) (*pb.GetTrainingRunResponse, error) {
	if req.RunId == "" {
		return nil, status.Error(codes.InvalidArgument, "run_id required")
	}

	run, err := s.store.GetTrainingRun(ctx, req.RunId, req.TenantId)
	if err != nil {
		return nil, err
	}

	return &pb.GetTrainingRunResponse{
		Run: run,
	}, nil
}

func (s *Service) GetMetricSeries(ctx context.Context, req *pb.GetMetricSeriesRequest) (*pb.GetMetricSeriesResponse, error) {
	if req.ModelName == "" || req.MetricName == "" {
		return nil, status.Error(codes.InvalidArgument, "model_name and metric_name required")
	}
	if req.StartDate == nil || req.EndDate == nil {
		return nil, status.Error(codes.InvalidArgument, "start_date and end_date required")
	}

	start := req.StartDate.AsTime()
	end := req.EndDate.AsTime()
	if start.After(end) {
		return nil, status.Error(codes.InvalidArgument, "start_date must be <= end_date")
	}

	points, err := s.store.GetMetricSeries(ctx, req)
	if err != nil {
		return nil, err
	}

	return &pb.GetMetricSeriesResponse{
		Points: points,
		Meta: &pb.AnalyticsMeta{
			Truncated: false,
		},
	}, nil
}

func (s *Service) ReportTrainingRun(ctx context.Context, req *pb.ReportTrainingRunRequest) (*pb.ReportTrainingRunResponse, error) {
	if req.Run == nil || req.Run.RunId == "" {
		return nil, status.Error(codes.InvalidArgument, "run and run_id required")
	}
	tenantID, err := resolveWriteTenantID(req.TenantId, req.Run.TenantId)
	if err != nil {
		return nil, err
	}
	req.Run.TenantId = tenantID

	// Observability: Log receipt of report (helps track retries)
	slog.Info("received training run report", "run_id", req.Run.RunId, "tenant_id", req.Run.TenantId)

	if err := s.store.SaveTrainingRun(ctx, req.Run); err != nil {
		slog.Error("failed to save training run", "error", err, "run_id", req.Run.RunId)
		return nil, err
	}

	return &pb.ReportTrainingRunResponse{Success: true}, nil
}

func resolveWriteTenantID(requestTenantID, payloadTenantID string) (string, error) {
	if requestTenantID == "" && payloadTenantID == "" {
		return "", status.Error(codes.InvalidArgument, "tenant_id required")
	}
	if requestTenantID != "" && payloadTenantID != "" && requestTenantID != payloadTenantID {
		return "", status.Error(codes.InvalidArgument, "tenant_id mismatch between request and payload")
	}
	if payloadTenantID != "" {
		return payloadTenantID, nil
	}
	return requestTenantID, nil
}
