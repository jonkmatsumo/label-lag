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

func (s *SQLStore) ListJobs(ctx context.Context, req *pb.ListJobsRequest) ([]*pb.Job, int64, string, error) {
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
	if req.TenantId != "" {
		queryBuilder.AddCondition("tenant_id = ?", req.TenantId)
	}

	limit := int32(20)
	if req.Limit > 0 {
		limit = req.Limit
	}
	if req.Pagination != nil && req.Pagination.Limit > 0 {
		limit = req.Pagination.Limit
	}

	// Cursor pagination
	var cursorObj *jobCursor
	if req.Pagination != nil && req.Pagination.Cursor != "" {
		var err error
		cursorObj, err = decodeJobCursor(req.Pagination.Cursor)
		if err != nil {
			return nil, 0, "", status.Errorf(codes.InvalidArgument, "invalid cursor: %v", err)
		}
		queryBuilder.AddCondition("(created_at, job_id) < (?, ?)", cursorObj.CreatedAt, cursorObj.JobId)
	}

	// Get total count
	// Get total count
	var total int64
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	if cursorObj == nil {
		countQuery, countArgs := queryBuilder.BuildCount()
		err := s.db.QueryRowContext(queryCtx, countQuery, countArgs...).Scan(&total)
		if err != nil {
			return nil, 0, "", db.MapDBError(err)
		}
	}

	// Get results
	queryBuilder.AddOrderBy("created_at DESC, job_id DESC")
	queryBuilder.SetLimit(limit)
	if cursorObj == nil {
		queryBuilder.SetOffset(req.Offset)
	}
	selectQuery, selectArgs := queryBuilder.BuildSelect()

	rows, err := s.db.QueryContext(queryCtx, selectQuery, selectArgs...)
	if err != nil {
		return nil, 0, "", db.MapDBError(err)
	}
	defer rows.Close()

	var jobs []*pb.Job
	var lastCreatedAt time.Time
	var lastJobID string

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
			return nil, 0, "", fmt.Errorf("failed to scan job: %v", err)
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
		lastCreatedAt = createdAt
		lastJobID = j.JobId
	}

	var nextCursor string
	if int32(len(jobs)) == limit && limit > 0 {
		nextCursor = encodeJobCursor(lastCreatedAt, lastJobID)
	}

	return jobs, total, nextCursor, nil
}

func (s *SQLStore) GetJob(ctx context.Context, jobID string, tenantID string) (*pb.Job, error) {
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
	args := []interface{}{jobID}
	if tenantID != "" {
		query += " AND tenant_id = $2"
		args = append(args, tenantID)
	}
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var j pb.Job
	var createdAt time.Time
	var startedAt, endedAt sql.NullTime
	var errorCode, errorMessage sql.NullString
	var paramsJSON, metricsJSON []byte

	err := s.db.QueryRowContext(queryCtx, query, args...).Scan(
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

func (s *SQLStore) GetJobEvents(ctx context.Context, req *pb.GetJobEventsRequest) ([]*pb.JobEvent, error) {
	query := `
		SELECT
			event_id,
			je.job_id,
			event_type,
			timestamp,
			details
		FROM job_events je
		INNER JOIN jobs j ON je.job_id = j.job_id
		WHERE je.job_id = $1
	`
	args := []interface{}{req.JobId, req.Limit, req.Offset}
	nextArg := 4
	if req.TenantId != "" {
		query += fmt.Sprintf(" AND j.tenant_id = $%d", nextArg)
		args = append(args, req.TenantId)
		nextArg++
	}
	if req.BeforeTs != nil {
		if req.BeforeId > 0 {
			query += fmt.Sprintf(" AND (je.timestamp, je.event_id) < ($%d, $%d)", nextArg, nextArg+1)
			args = append(args, req.BeforeTs.AsTime(), req.BeforeId)
			nextArg += 2
		} else {
			query += fmt.Sprintf(" AND je.timestamp < $%d", nextArg)
			args = append(args, req.BeforeTs.AsTime())
			nextArg++
		}
	} else if req.BeforeId > 0 {
		query += fmt.Sprintf(" AND je.event_id < $%d", nextArg)
		args = append(args, req.BeforeId)
		nextArg++
	}
	query += " ORDER BY je.timestamp DESC, je.event_id DESC LIMIT $2 OFFSET $3"
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
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

func (s *SQLStore) GetJobSummary(ctx context.Context, req *pb.GetJobSummaryRequest) ([]*pb.JobSummaryBucket, error) {
	if err := ctx.Err(); err != nil {
		return nil, db.MapDBError(err)
	}

	queryCtx, cancel := context.WithTimeout(ctx, hotAnalyticsQueryTimeout)
	defer cancel()

	query := `
		SELECT
			date_trunc('hour', created_at) as bucket,
			COUNT(*) as total,
			COUNT(*) FILTER (WHERE status = 'completed') as completed,
			COUNT(*) FILTER (WHERE status = 'failed') as failed
		FROM jobs
		WHERE created_at >= $1 AND created_at <= $2 %s
		GROUP BY bucket
		ORDER BY bucket ASC
	`
	tenantDetail := ""
	start := time.Now().AddDate(0, 0, -7)
	if req.StartTime != nil {
		start = req.StartTime.AsTime()
	}
	end := time.Now()
	if req.EndTime != nil {
		end = req.EndTime.AsTime()
	}

	args := []interface{}{start, end}
	if req.TenantId != "" {
		tenantDetail = " AND tenant_id = $3"
		args = append(args, req.TenantId)
	}
	query = fmt.Sprintf(query, tenantDetail)

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var summaries []*pb.JobSummaryBucket
	for rows.Next() {
		var b pb.JobSummaryBucket
		var bucket time.Time
		if err := rows.Scan(&bucket, &b.TotalJobs, &b.CompletedJobs, &b.FailedJobs); err != nil {
			return nil, fmt.Errorf("failed to scan job summary: %v", err)
		}
		b.BucketTime = timestamppb.New(bucket)
		summaries = append(summaries, &b)
	}
	return summaries, nil
}
func (s *SQLStore) CancelJob(ctx context.Context, jobID string, tenantID string) error {
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	// Check current status - only QUEUED or RUNNING jobs can be cancelled
	var currentStatus string
	err := s.db.QueryRowContext(queryCtx, "SELECT status FROM jobs WHERE job_id = $1 AND tenant_id = $2", jobID, tenantID).Scan(&currentStatus)
	if err == sql.ErrNoRows {
		return status.Errorf(codes.NotFound, "job not found: %s", jobID)
	}
	if err != nil {
		return db.MapDBError(err)
	}

	if currentStatus != "queued" && currentStatus != "running" {
		return status.Errorf(codes.FailedPrecondition, "cannot cancel job in state: %s", currentStatus)
	}

	_, err = s.db.ExecContext(queryCtx, `
		UPDATE jobs
		SET status = 'cancel_requested', cancel_requested_at = NOW()
		WHERE job_id = $1 AND tenant_id = $2
	`, jobID, tenantID)
	if err != nil {
		return db.MapDBError(err)
	}

	return s.emitJobEvent(queryCtx, jobID, "cancel_requested", `{"reason": "user_requested"}`)
}

func (s *SQLStore) RetryJob(ctx context.Context, jobID string, tenantID string) (string, error) {
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	// Get original job details
	var jobType string
	var paramsJSON []byte
	err := s.db.QueryRowContext(queryCtx, "SELECT job_type, params FROM jobs WHERE job_id = $1 AND tenant_id = $2", jobID, tenantID).Scan(&jobType, &paramsJSON)
	if err == sql.ErrNoRows {
		return "", status.Errorf(codes.NotFound, "job not found: %s", jobID)
	}
	if err != nil {
		return "", db.MapDBError(err)
	}

	newJobID := fmt.Sprintf("job-%d", time.Now().UnixNano())
	_, err = s.db.ExecContext(queryCtx, `
		INSERT INTO jobs (job_id, job_type, status, params, tenant_id, retry_of_job_id)
		VALUES ($1, $2, 'queued', $3, $4, $5)
	`, newJobID, jobType, paramsJSON, tenantID, jobID)
	if err != nil {
		return "", db.MapDBError(err)
	}

	// Emit event on original job
	_ = s.emitJobEvent(queryCtx, jobID, "retried", fmt.Sprintf(`{"new_job_id": %q}`, newJobID))

	return newJobID, nil
}

func (s *SQLStore) emitJobEvent(ctx context.Context, jobID string, eventType string, details string) error {
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err := s.db.ExecContext(queryCtx, `
		INSERT INTO job_events (job_id, event_type, details)
		VALUES ($1, $2, $3)
	`, jobID, eventType, details)
	return err
}
