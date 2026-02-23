package service

import (
	"context"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	commonv1 "github.com/jonkmatsumo/label-lag/go/common/proto/v1"
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

	jobs, total, nextCursor, err := s.store.ListJobs(ctx, req)
	if err != nil {
		return nil, err
	}

	resp := &pb.ListJobsResponse{
		Jobs: jobs,
		Pagination: &commonv1.CursorPageResponse{
			NextCursor: nextCursor,
		},
	}
	if req.Pagination == nil || req.Pagination.Cursor == "" {
		resp.Pagination.Total = &total
	}
	return resp, nil
}

func (s *Service) GetJob(ctx context.Context, req *pb.GetJobRequest) (*pb.GetJobResponse, error) {
	if req.JobId == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id required")
	}

	job, err := s.store.GetJob(ctx, req.JobId, req.TenantId)
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
	if req.BeforeId < 0 {
		return nil, status.Error(codes.InvalidArgument, "before_id must be >= 0")
	}
	req.Limit = limit
	req.Offset = offset

	// Verify job exists first
	_, err = s.store.GetJob(ctx, req.JobId, req.TenantId)
	if err != nil {
		return nil, err
	}

	events, err := s.store.GetJobEvents(ctx, req)
	if err != nil {
		return nil, err
	}

	return &pb.GetJobEventsResponse{
		Events: events,
	}, nil
}

func (s *Service) GetJobSummary(ctx context.Context, req *pb.GetJobSummaryRequest) (*pb.GetJobSummaryResponse, error) {
	startTime, endTime, err := mergeWindowFromEnvelope(req.StartTime, req.EndTime, req.GetQuery())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "query.start_time and query.end_time must be RFC3339 or YYYY-MM-DD")
	}
	req.StartTime = startTime
	req.EndTime = endTime

	if req.StartTime != nil && req.EndTime != nil && req.StartTime.AsTime().After(req.EndTime.AsTime()) {
		return nil, status.Error(codes.InvalidArgument, "start_time must be <= end_time")
	}

	summaries, err := s.store.GetJobSummary(ctx, req)
	if err != nil {
		return nil, err
	}

	return &pb.GetJobSummaryResponse{
		Summaries: summaries,
	}, nil
}

func (s *Service) CancelJob(ctx context.Context, req *pb.CancelJobRequest) (*pb.CancelJobResponse, error) {
	if req.JobId == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id required")
	}
	if err := s.store.CancelJob(ctx, req.JobId, req.TenantId); err != nil {
		return nil, err
	}
	return &pb.CancelJobResponse{Success: true}, nil
}

func (s *Service) RetryJob(ctx context.Context, req *pb.RetryJobRequest) (*pb.RetryJobResponse, error) {
	if req.JobId == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id required")
	}
	newJobID, err := s.store.RetryJob(ctx, req.JobId, req.TenantId)
	if err != nil {
		return nil, err
	}
	return &pb.RetryJobResponse{NewJobId: newJobID}, nil
}
