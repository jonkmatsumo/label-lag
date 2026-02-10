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
	query := `
		INSERT INTO training_runs (
			run_id, model_name, status, started_at, ended_at, metrics, params, dataset_id, mlflow_run_id
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9
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
	)
	if err != nil {
		return db.MapDBError(err)
	}

	return nil
}

func (s *SQLStore) ListTrainingRuns(ctx context.Context, modelName string, limit, offset int32) ([]*pb.TrainingRun, int64, error) {
	queryBuilder := db.NewQueryBuilder(`
		SELECT
			run_id, model_name, status, started_at, ended_at, metrics, params, dataset_id, mlflow_run_id
		FROM training_runs
	`)

	if modelName != "" {
		queryBuilder.AddCondition("model_name = ?", modelName)
	}

	// Count
	countQuery, countArgs := queryBuilder.BuildCount()
	var total int64
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, countQuery, countArgs...).Scan(&total)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}

	// List
	queryBuilder.AddOrderBy("started_at DESC")
	queryBuilder.SetLimit(limit)
	queryBuilder.SetOffset(offset)
	selectQuery, selectArgs := queryBuilder.BuildSelect()

	rows, err := s.db.QueryContext(queryCtx, selectQuery, selectArgs...)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}
	defer rows.Close()

	var runs []*pb.TrainingRun
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
		); err != nil {
			return nil, 0, fmt.Errorf("failed to scan run: %v", err)
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

		runs = append(runs, &r)
	}

	return runs, total, nil
}

func (s *SQLStore) GetTrainingRun(ctx context.Context, runID string) (*pb.TrainingRun, error) {
	query := `
		SELECT
			run_id, model_name, status, started_at, ended_at, metrics, params, dataset_id, mlflow_run_id
		FROM training_runs
		WHERE run_id = $1
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var r pb.TrainingRun
	var startedAt time.Time
	var endedAt sql.NullTime
	var metricsJSON, paramsJSON []byte
	var datasetID, mlflowID sql.NullString

	err := s.db.QueryRowContext(queryCtx, query, runID).Scan(
		&r.RunId,
		&r.ModelName,
		&r.Status,
		&startedAt,
		&endedAt,
		&metricsJSON,
		&paramsJSON,
		&datasetID,
		&mlflowID,
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

func (s *SQLStore) GetMetricSeries(ctx context.Context, modelName, metricName string, start, end time.Time) ([]*pb.MetricPoint, error) {
	// Extract metric from JSONB
	// Assumes metrics is a flat object
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
		ORDER BY started_at ASC
	`, metricName)

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, modelName, start, end, metricName)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var points []*pb.MetricPoint
	for rows.Next() {
		var p pb.MetricPoint
		var ts time.Time
		if err := rows.Scan(&ts, &p.Value, &p.RunId); err != nil {
			// Skip malformed? or fail
			continue
		}
		p.Timestamp = timestamppb.New(ts)
		points = append(points, &p)
	}

	return points, nil
}
