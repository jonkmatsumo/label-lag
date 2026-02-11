package service

import (
	"context"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (s *Service) ListTrainingRuns(ctx context.Context, req *pb.ListTrainingRunsRequest) (*pb.ListTrainingRunsResponse, error) {
	limit, err := normalizeLimit(req.Limit, 50, 250, "limit")
	if err != nil {
		return nil, err
	}
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}
	req.Limit = limit
	req.Offset = offset

	runs, total, err := s.store.ListTrainingRuns(ctx, req)
	if err != nil {
		return nil, err
	}

	return &pb.ListTrainingRunsResponse{
		Runs:  runs,
		Total: total,
	}, nil
}

func (s *Service) GetTrainingRun(ctx context.Context, req *pb.GetTrainingRunRequest) (*pb.GetTrainingRunResponse, error) {
	if req.RunId == "" {
		return nil, status.Error(codes.InvalidArgument, "run_id required")
	}

	run, err := s.store.GetTrainingRun(ctx, req.RunId)
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

	points, err := s.store.GetMetricSeries(ctx, req.ModelName, req.MetricName, start, end)
	if err != nil {
		return nil, err
	}

	return &pb.GetMetricSeriesResponse{
		Points: points,
	}, nil
}

func (s *Service) ReportTrainingRun(ctx context.Context, req *pb.ReportTrainingRunRequest) (*pb.ReportTrainingRunResponse, error) {
	if req.Run == nil || req.Run.RunId == "" {
		return nil, status.Error(codes.InvalidArgument, "run and run_id required")
	}

	if err := s.store.SaveTrainingRun(ctx, req.Run); err != nil {
		return nil, err
	}

	return &pb.ReportTrainingRunResponse{Success: true}, nil
}
