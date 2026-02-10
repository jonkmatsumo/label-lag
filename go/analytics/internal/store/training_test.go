package store

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestSaveTrainingRun(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	run := &pb.TrainingRun{
		RunId:       "run-1",
		ModelName:   "model-1",
		Status:      "COMPLETED",
		StartedAt:   timestamppb.New(time.Now().UTC()),
		MetricsJson: `{"accuracy": 0.9}`,
		ParamsJson:  `{"lr": 0.1}`,
	}

	mock.ExpectExec("INSERT INTO training_runs").
		WithArgs("run-1", "model-1", "COMPLETED", sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	err = s.SaveTrainingRun(context.Background(), run)
	require.NoError(t, err)
}

func TestListTrainingRuns(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	mock.ExpectQuery("SELECT COUNT").
		WithArgs("model-1").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(1))

	mock.ExpectQuery("SELECT run_id").
		WithArgs("model-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"run_id", "model_name", "status", "started_at", "ended_at", "metrics", "params", "dataset_id", "mlflow_run_id",
		}).AddRow("run-1", "model-1", "COMPLETED", time.Now(), nil, nil, nil, nil, nil))

	runs, total, err := s.ListTrainingRuns(context.Background(), "model-1", 10, 0)
	require.NoError(t, err)
	assert.Equal(t, int64(1), total)
	assert.Len(t, runs, 1)
}

func TestGetMetricSeries(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	start := time.Now().Add(-time.Hour).UTC()
	end := time.Now().UTC()

	mock.ExpectQuery("SELECT started_at").
		WithArgs("model-1", start, end, "accuracy").
		WillReturnRows(sqlmock.NewRows([]string{"started_at", "val", "run_id"}).
			AddRow(start, 0.85, "run-1").
			AddRow(end, 0.90, "run-2"))

	points, err := s.GetMetricSeries(context.Background(), "model-1", "accuracy", start, end)
	require.NoError(t, err)
	assert.Len(t, points, 2)
	assert.Equal(t, 0.85, points[0].Value)
	assert.Equal(t, 0.90, points[1].Value)
}
