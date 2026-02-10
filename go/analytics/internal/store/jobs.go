package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func (s *SQLStore) ListJobs(ctx context.Context, req *pb.ListJobsRequest) ([]*pb.Job, int64, error) {
	queryBuilder := db.NewQueryBuilder(`
		SELECT
			job_id,
			job_type,
			status,
			created_at,
			started_at,
			ended_at,
			error_code,
			error_message,
			params,
			metrics
		FROM jobs
	`)

	if req.JobType != "" {
		queryBuilder.AddCondition("job_type = ?", req.JobType)
	}
	if req.Status != "" {
		queryBuilder.AddCondition("status = ?", req.Status)
	}
	if req.StartDate != nil {
		queryBuilder.AddCondition("created_at >= ?", req.StartDate.AsTime())
	}
	if req.EndDate != nil {
		queryBuilder.AddCondition("created_at <= ?", req.EndDate.AsTime())
	}

	// Get total count
	countQuery, countArgs := queryBuilder.BuildCount()
	var total int64
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, countQuery, countArgs...).Scan(&total)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}

	// Get results
	queryBuilder.AddOrderBy("created_at DESC")
	queryBuilder.SetLimit(req.Limit)
	queryBuilder.SetOffset(req.Offset)
	selectQuery, selectArgs := queryBuilder.BuildSelect()

	rows, err := s.db.QueryContext(queryCtx, selectQuery, selectArgs...)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}
	defer rows.Close()

	var jobs []*pb.Job
	for rows.Next() {
		var j pb.Job
		var createdAt time.Time
		var startedAt, endedAt sql.NullTime
		var errorCode, errorMessage sql.NullString
		var paramsJSON, metricsJSON []byte

		if err := rows.Scan(
			&j.JobId,
			&j.JobType,
			&j.Status,
			&createdAt,
			&startedAt,
			&endedAt,
			&errorCode,
			&errorMessage,
			&paramsJSON,
			&metricsJSON,
		); err != nil {
			return nil, 0, fmt.Errorf("failed to scan job: %v", err)
		}

		j.CreatedAt = timestamppb.New(createdAt)
		if startedAt.Valid {
			j.StartedAt = timestamppb.New(startedAt.Time)
		}
		if endedAt.Valid {
			j.EndedAt = timestamppb.New(endedAt.Time)
		}
		if errorCode.Valid {
			j.ErrorCode = errorCode.String
		}
		if errorMessage.Valid {
			j.ErrorMessage = errorMessage.String
		}
		if len(paramsJSON) > 0 {
			j.ParamsJson = string(paramsJSON)
		}
		if len(metricsJSON) > 0 {
			j.MetricsJson = string(metricsJSON)
		}

		jobs = append(jobs, &j)
	}
	return jobs, total, nil
}

func (s *SQLStore) GetJob(ctx context.Context, jobID string) (*pb.Job, error) {
	query := `
		SELECT
			job_id,
			job_type,
			status,
			created_at,
			started_at,
			ended_at,
			error_code,
			error_message,
			params,
			metrics
		FROM jobs
		WHERE job_id = $1
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var j pb.Job
	var createdAt time.Time
	var startedAt, endedAt sql.NullTime
	var errorCode, errorMessage sql.NullString
	var paramsJSON, metricsJSON []byte

	err := s.db.QueryRowContext(queryCtx, query, jobID).Scan(
		&j.JobId,
		&j.JobType,
		&j.Status,
		&createdAt,
		&startedAt,
		&endedAt,
		&errorCode,
		&errorMessage,
		&paramsJSON,
		&metricsJSON,
	)
	if err == sql.ErrNoRows {
		return nil, status.Errorf(codes.NotFound, "job not found: %s", jobID)
	}
	if err != nil {
		return nil, db.MapDBError(err)
	}

	j.CreatedAt = timestamppb.New(createdAt)
	if startedAt.Valid {
		j.StartedAt = timestamppb.New(startedAt.Time)
	}
	if endedAt.Valid {
		j.EndedAt = timestamppb.New(endedAt.Time)
	}
	if errorCode.Valid {
		j.ErrorCode = errorCode.String
	}
	if errorMessage.Valid {
		j.ErrorMessage = errorMessage.String
	}
	if len(paramsJSON) > 0 {
		j.ParamsJson = string(paramsJSON)
	}
	if len(metricsJSON) > 0 {
		j.MetricsJson = string(metricsJSON)
	}

	return &j, nil
}

func (s *SQLStore) GetJobEvents(ctx context.Context, jobID string) ([]*pb.JobEvent, error) {
	query := `
		SELECT
			event_id,
			job_id,
			event_type,
			timestamp,
			details
		FROM job_events
		WHERE job_id = $1
		ORDER BY timestamp ASC
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, jobID)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var events []*pb.JobEvent
	for rows.Next() {
		var e pb.JobEvent
		var ts time.Time
		var detailsJSON []byte

		if err := rows.Scan(
			&e.EventId,
			&e.JobId,
			&e.EventType,
			&ts,
			&detailsJSON,
		); err != nil {
			return nil, fmt.Errorf("failed to scan job event: %v", err)
		}

		e.Timestamp = timestamppb.New(ts)
		if len(detailsJSON) > 0 {
			e.DetailsJson = string(detailsJSON)
		}

		events = append(events, &e)
	}
	return events, nil
}
