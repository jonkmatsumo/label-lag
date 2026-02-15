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

func (s *SQLStore) SaveTrainingRun(ctx context.Context, run *pb.TrainingRun) error {
	if run == nil || run.RunId == "" {
		return status.Error(codes.InvalidArgument, "run and run_id required")
	}
	if run.TenantId == "" {
		return status.Error(codes.InvalidArgument, "tenant_id required")
	}

	query := `
		INSERT INTO training_runs (
			run_id, model_name, status, started_at, ended_at, metrics, params, dataset_id, mlflow_run_id, tenant_id
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10
		)
		ON CONFLICT (run_id) DO UPDATE SET
			status = EXCLUDED.status,
			ended_at = EXCLUDED.ended_at,
			metrics = EXCLUDED.metrics,
			params = EXCLUDED.params,
			mlflow_run_id = EXCLUDED.mlflow_run_id
	`

	startedAt := run.StartedAt.AsTime()
	var endedAt sql.NullTime
	if run.EndedAt != nil {
		endedAt = sql.NullTime{Time: run.EndedAt.AsTime(), Valid: true}
	}

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err := s.db.ExecContext(queryCtx, query,
		run.RunId,
		run.ModelName,
		run.Status,
		startedAt,
		endedAt,
		[]byte(run.MetricsJson),
		[]byte(run.ParamsJson),
		run.DatasetId,
		run.MlflowRunId,
		run.TenantId,
	)
	if err != nil {
		return db.MapDBError(err)
	}

	return nil
}

func (s *SQLStore) ListTrainingRuns(ctx context.Context, req *pb.ListTrainingRunsRequest) ([]*pb.TrainingRun, int64, string, error) {
	queryBuilder := db.NewQueryBuilder(`
		SELECT
			run_id, model_name, status, started_at, ended_at, metrics, params, dataset_id, mlflow_run_id, tenant_id
		FROM training_runs
	`)

	if req.ModelName != "" {
		queryBuilder.AddCondition("model_name = ?", req.ModelName)
	}
	if req.Status != "" {
		queryBuilder.AddCondition("status = ?", req.Status)
	}
	if req.StartDate != nil {
		queryBuilder.AddCondition("started_at >= ?", req.StartDate.AsTime())
	}
	if req.EndDate != nil {
		queryBuilder.AddCondition("started_at <= ?", req.EndDate.AsTime())
	}
	if req.TenantId != "" {
		queryBuilder.AddCondition("tenant_id = ?", req.TenantId)
	}

	limit := int32(50)
	if req.Limit > 0 {
		limit = req.Limit
	}
	if req.Pagination != nil && req.Pagination.Limit > 0 {
		limit = req.Pagination.Limit
	}

	// Cursor pagination
	var cursorObj *trainingRunCursor
	if req.Pagination != nil && req.Pagination.Cursor != "" {
		var err error
		cursorObj, err = decodeTrainingRunCursor(req.Pagination.Cursor)
		if err != nil {
			return nil, 0, "", status.Errorf(codes.InvalidArgument, "invalid cursor: %v", err)
		}
		queryBuilder.AddCondition("(started_at, run_id) < (?, ?)", cursorObj.StartedAt, cursorObj.RunId)
	}

	// Count
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

	// List
	queryBuilder.AddOrderBy("started_at DESC, run_id DESC")
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

	var runs []*pb.TrainingRun
	var lastStartedAt time.Time
	var lastRunID string

	for rows.Next() {
		var r pb.TrainingRun
		var startedAt time.Time
		var endedAt sql.NullTime
		var metricsJSON, paramsJSON []byte
		var datasetID, mlflowID sql.NullString

		if err := rows.Scan(
			&r.RunId,
			&r.ModelName,
			&r.Status,
			&startedAt,
			&endedAt,
			&metricsJSON,
			&paramsJSON,
			&datasetID,
			&mlflowID,
			&r.TenantId,
		); err != nil {
			return nil, 0, "", fmt.Errorf("failed to scan run: %v", err)
		}

		r.StartedAt = timestamppb.New(startedAt)
		if endedAt.Valid {
			jts := endedAt.Time
			r.EndedAt = timestamppb.New(jts)
		}
		if metricsJSON != nil {
			r.MetricsJson = string(metricsJSON)
		}
		if paramsJSON != nil {
			r.ParamsJson = string(paramsJSON)
		}
		if datasetID.Valid {
			r.DatasetId = datasetID.String
		}
		if mlflowID.Valid {
			r.MlflowRunId = mlflowID.String
		}

		runs = append(runs, &r)
		lastStartedAt = startedAt
		lastRunID = r.RunId
	}

	var nextCursor string
	if int32(len(runs)) == limit && limit > 0 {
		nextCursor = encodeTrainingRunCursor(lastStartedAt, lastRunID)
	}

	return runs, total, nextCursor, nil
}

func (s *SQLStore) ListModelVersions(ctx context.Context, req *pb.ListModelVersionsRequest) ([]*pb.TrainingRun, int64, string, error) {
	// For now, versions are just completed training runs for a model
	return s.ListTrainingRuns(ctx, &pb.ListTrainingRunsRequest{
		ModelName:  req.ModelName,
		Status:     "completed",
		Limit:      req.Limit,
		Offset:     req.Offset,
		TenantId:   req.TenantId,
		Pagination: req.Pagination,
	})
}

func (s *SQLStore) GetTrainingRun(ctx context.Context, runID string, tenantID string) (*pb.TrainingRun, error) {
	query := `
		SELECT
			run_id, model_name, status, started_at, ended_at, metrics, params, dataset_id, mlflow_run_id, tenant_id
		FROM training_runs
		WHERE run_id = $1
	`
	args := []interface{}{runID}
	if tenantID != "" {
		query += " AND tenant_id = $2"
		args = append(args, tenantID)
	}
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var r pb.TrainingRun
	var startedAt time.Time
	var endedAt sql.NullTime
	var metricsJSON, paramsJSON []byte
	var datasetID, mlflowID sql.NullString

	err := s.db.QueryRowContext(queryCtx, query, args...).Scan(
		&r.RunId,
		&r.ModelName,
		&r.Status,
		&startedAt,
		&endedAt,
		&metricsJSON,
		&paramsJSON,
		&datasetID,
		&mlflowID,
		&r.TenantId,
	)
	if err == sql.ErrNoRows {
		return nil, status.Errorf(codes.NotFound, "run not found: %s", runID)
	}
	if err != nil {
		return nil, db.MapDBError(err)
	}

	r.StartedAt = timestamppb.New(startedAt)
	if endedAt.Valid {
		r.EndedAt = timestamppb.New(endedAt.Time)
	}
	if metricsJSON != nil {
		r.MetricsJson = string(metricsJSON)
	}
	if paramsJSON != nil {
		r.ParamsJson = string(paramsJSON)
	}
	if datasetID.Valid {
		r.DatasetId = datasetID.String
	}
	if mlflowID.Valid {
		r.MlflowRunId = mlflowID.String
	}

	return &r, nil
}

func (s *SQLStore) GetMetricSeries(ctx context.Context, req *pb.GetMetricSeriesRequest) ([]*pb.MetricPoint, error) {
	if req.ModelName == "" || req.MetricName == "" {
		return nil, status.Error(codes.InvalidArgument, "model_name and metric_name required")
	}

	start := time.Now().AddDate(0, 0, -30)
	if req.StartDate != nil {
		start = req.StartDate.AsTime()
	}
	end := time.Now()
	if req.EndDate != nil {
		end = req.EndDate.AsTime()
	}

	if end.Sub(start) > 90*24*time.Hour {
		return nil, status.Error(codes.InvalidArgument, "query window cannot exceed 90 days")
	}

	query := fmt.Sprintf(`
		SELECT
			started_at,
			(metrics->>'%s')::FLOAT,
			run_id
		FROM training_runs
		WHERE model_name = $1
		  AND started_at >= $2
		  AND started_at <= $3
		  AND metrics ? $4
	`, req.MetricName)

	args := []interface{}{req.ModelName, start, end, req.MetricName}
	if req.TenantId != "" {
		query += " AND tenant_id = $5"
		args = append(args, req.TenantId)
	}
	query += " ORDER BY started_at ASC"

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var points []*pb.MetricPoint
	for rows.Next() {
		var p pb.MetricPoint
		var ts time.Time
		if err := rows.Scan(&ts, &p.Value, &p.RunId); err != nil {
			continue
		}
		p.Timestamp = timestamppb.New(ts)
		points = append(points, &p)
	}

	return points, nil
}
