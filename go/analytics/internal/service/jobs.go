package service

import (
	"context"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (s *Service) ListJobs(ctx context.Context, req *pb.ListJobsRequest) (*pb.ListJobsResponse, error) {
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

	if req.StartDate != nil && req.EndDate != nil && req.StartDate.AsTime().After(req.EndDate.AsTime()) {
		return nil, status.Error(codes.InvalidArgument, "start_date must be <= end_date")
	}

	jobs, total, err := s.store.ListJobs(ctx, req)
	if err != nil {
		return nil, err
	}

	return &pb.ListJobsResponse{
		Jobs:  jobs,
		Total: total,
	}, nil
}

func (s *Service) GetJob(ctx context.Context, req *pb.GetJobRequest) (*pb.GetJobResponse, error) {
	if req.JobId == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id required")
	}

	job, err := s.store.GetJob(ctx, req.JobId)
	if err != nil {
		return nil, err
	}

	return &pb.GetJobResponse{
		Job: job,
	}, nil
}

func (s *Service) GetJobEvents(ctx context.Context, req *pb.GetJobEventsRequest) (*pb.GetJobEventsResponse, error) {
	if req.JobId == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id required")
	}

	limit, err := normalizeLimit(req.Limit, 100, 500, "limit")
	if err != nil {
		return nil, err
	}
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}

	// Verify job exists first
	_, err = s.store.GetJob(ctx, req.JobId)
	if err != nil {
		return nil, err
	}

	events, err := s.store.GetJobEvents(ctx, req.JobId, limit, offset)
	if err != nil {
		return nil, err
	}

	return &pb.GetJobEventsResponse{
		Events: events,
	}, nil
}
