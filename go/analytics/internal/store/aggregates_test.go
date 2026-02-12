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

func TestGetKpis(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	startTime := time.Now().AddDate(0, 0, -7).UTC()
	endTime := time.Now().UTC()

	req := &pb.GetKpisRequest{
		StartTime: timestamppb.New(startTime),
		EndTime:   timestamppb.New(endTime),
		GroupBy:   "day",
	}

	// Mock summary query
	mock.ExpectQuery(`SELECT .* FROM aggregates_daily`).
		WithArgs(startTime, endTime).
		WillReturnRows(sqlmock.NewRows([]string{"decisions", "alerts", "score", "fired"}).
			AddRow(100, 10, 5000, 200))

	// Mock buckets query
	mock.ExpectQuery(`SELECT date, total_decisions, total_alerts, .* FROM aggregates_daily`).
		WithArgs(startTime, endTime).
		WillReturnRows(sqlmock.NewRows([]string{"date", "decisions", "alerts", "avg_score"}).
			AddRow(startTime, 50, 5, 50.0).
			AddRow(endTime, 50, 5, 50.0))

	resp, err := s.GetKpis(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, int64(100), resp.TotalDecisions)
	assert.Equal(t, int64(10), resp.TotalAlerts)
	assert.Equal(t, 50.0, resp.AvgScore)
	assert.Len(t, resp.Buckets, 2)
}

func TestGetVolumeSeries(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	req := &pb.GetVolumeSeriesRequest{
		Granularity: "hour",
	}

	mock.ExpectQuery(`SELECT hour, total_decisions, total_alerts FROM aggregates_hourly`).
		WillReturnRows(sqlmock.NewRows([]string{"hour", "count", "alerts"}).
			AddRow(time.Now().Truncate(time.Hour), 10, 1))

	resp, err := s.GetVolumeSeries(context.Background(), req)
	require.NoError(t, err)
	assert.Len(t, resp.Points, 1)
	assert.Equal(t, int64(10), resp.Points[0].Count)
}

func TestLogInferenceEvent_IncrementsAggregates(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	event := &pb.InferenceEvent{
		RequestId:  "req-1",
		Timestamp:  timestamppb.New(time.Now().UTC()),
		FinalScore: 85,
		Decision:   DecisionReject,
		RuleImpacts: []*pb.RuleImpact{
			{RuleId: "rule-1", ScoreDelta: 10},
		},
	}

	mock.ExpectBegin()
	mock.ExpectExec("INSERT INTO inference_events").WillReturnResult(sqlmock.NewResult(1, 1))

	// Daily aggregate increment
	mock.ExpectExec("INSERT INTO aggregates_daily").
		WithArgs("", sqlmock.AnyArg(), 1, 85, 1).
		WillReturnResult(sqlmock.NewResult(1, 1))

	// Hourly aggregate increment
	mock.ExpectExec("INSERT INTO aggregates_hourly").
		WithArgs("", sqlmock.AnyArg(), 1, 85, 1).
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectPrepare("INSERT INTO rule_impacts")
	mock.ExpectExec("INSERT INTO rule_impacts").WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectCommit()

	err = s.LogInferenceEvent(context.Background(), event)
	require.NoError(t, err)
	require.NoError(t, mock.ExpectationsWereMet())
}
