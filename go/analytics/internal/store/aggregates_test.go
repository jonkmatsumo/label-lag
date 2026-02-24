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
	mock.ExpectQuery(`SELECT\s+date,\s*total_decisions,\s*total_alerts,\s*CASE WHEN total_decisions > 0 THEN sum_score::float / total_decisions ELSE 0 END\s+FROM aggregates_daily\s+WHERE .*ORDER BY date DESC\s+LIMIT 2001`).
		WithArgs(startTime, endTime).
		WillReturnRows(sqlmock.NewRows([]string{"date", "decisions", "alerts", "avg_score"}).
			AddRow(endTime, 50, 5, 50.0).
			AddRow(startTime, 50, 5, 50.0))

	resp, err := s.GetKpis(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, int64(100), resp.TotalDecisions)
	assert.Equal(t, int64(10), resp.TotalAlerts)
	assert.Equal(t, 50.0, resp.AvgScore)
	assert.Len(t, resp.Buckets, 2)
	require.NotNil(t, resp.Current)
	assert.Equal(t, resp.TotalDecisions, resp.Current.TotalDecisions)
	assert.False(t, resp.Meta.Partial)
	assert.True(t, resp.Buckets[0].Timestamp.AsTime().Before(resp.Buckets[1].Timestamp.AsTime()))
}

func TestGetKpis_CompareToPrevious(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	start := time.Date(2024, 1, 10, 0, 0, 0, 0, time.UTC)
	end := time.Date(2024, 1, 12, 0, 0, 0, 0, time.UTC)
	prevStart := time.Date(2024, 1, 8, 0, 0, 0, 0, time.UTC)
	prevEnd := start

	req := &pb.GetKpisRequest{
		StartTime:         timestamppb.New(start),
		EndTime:           timestamppb.New(end),
		CompareToPrevious: true,
	}

	mock.ExpectQuery(`SELECT .* FROM aggregates_daily`).
		WithArgs(start, end).
		WillReturnRows(sqlmock.NewRows([]string{"decisions", "alerts", "score", "fired"}).
			AddRow(100, 10, 5000, 200))

	mock.ExpectQuery(`SELECT .* FROM aggregates_daily`).
		WithArgs(prevStart, prevEnd).
		WillReturnRows(sqlmock.NewRows([]string{"decisions", "alerts", "score", "fired"}).
			AddRow(80, 8, 3600, 160))

	resp, err := s.GetKpis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp.Current)
	require.NotNil(t, resp.Previous)
	assert.Equal(t, int64(100), resp.TotalDecisions)
	assert.Equal(t, int64(100), resp.Current.TotalDecisions)
	assert.Equal(t, int64(80), resp.Previous.TotalDecisions)
	assert.Equal(t, int64(160), resp.Previous.RulesFiredTotal)
	assert.False(t, resp.Meta.Partial)
}

func TestGetVolumeSeries(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	req := &pb.GetVolumeSeriesRequest{
		Granularity: "hour",
	}

	mock.ExpectQuery(`SELECT hour, total_decisions, total_alerts\s+FROM aggregates_hourly\s+WHERE 1=1\s+ORDER BY hour DESC\s+LIMIT 1001`).
		WillReturnRows(sqlmock.NewRows([]string{"hour", "count", "alerts"}).
			AddRow(time.Now().Truncate(time.Hour), 10, 1))

	resp, err := s.GetVolumeSeries(context.Background(), req)
	require.NoError(t, err)
	assert.Len(t, resp.Points, 1)
	assert.Equal(t, int64(10), resp.Points[0].Count)
	require.NotNil(t, resp.Current)
	assert.Equal(t, resp.Points[0].Count, resp.Current.Points[0].Count)
	assert.False(t, resp.Meta.Partial)
}

func TestGetVolumeSeries_CompareToPrevious(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	start := time.Date(2024, 1, 10, 10, 0, 0, 0, time.UTC)
	end := time.Date(2024, 1, 10, 12, 0, 0, 0, time.UTC)
	prevStart := time.Date(2024, 1, 10, 8, 0, 0, 0, time.UTC)
	prevEnd := start

	req := &pb.GetVolumeSeriesRequest{
		Granularity:       "hour",
		StartTime:         timestamppb.New(start),
		EndTime:           timestamppb.New(end),
		CompareToPrevious: true,
	}

	mock.ExpectQuery(`SELECT hour, total_decisions, total_alerts\s+FROM aggregates_hourly\s+WHERE .*ORDER BY hour DESC\s+LIMIT 1001`).
		WithArgs(start, end).
		WillReturnRows(sqlmock.NewRows([]string{"hour", "count", "alerts"}).
			AddRow(end, 30, 3).
			AddRow(start, 10, 1))

	mock.ExpectQuery(`SELECT hour, total_decisions, total_alerts\s+FROM aggregates_hourly\s+WHERE .*ORDER BY hour DESC\s+LIMIT 1001`).
		WithArgs(prevStart, prevEnd).
		WillReturnRows(sqlmock.NewRows([]string{"hour", "count", "alerts"}).
			AddRow(prevEnd, 20, 2).
			AddRow(prevStart, 8, 1))

	resp, err := s.GetVolumeSeries(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp.Current)
	require.NotNil(t, resp.Previous)
	require.Len(t, resp.Points, 2)
	require.Len(t, resp.Current.Points, 2)
	require.Len(t, resp.Previous.Points, 2)
	assert.Equal(t, int64(10), resp.Points[0].Count)
	assert.Equal(t, int64(20), resp.Previous.Points[1].Count)
}

func TestGetVolumeSeries_DownsamplesHourlyToDailyWhenWindowExceedsCap(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	end := start.Add((MaxPointsHourly + 24) * time.Hour)
	req := &pb.GetVolumeSeriesRequest{
		Granularity: "hour",
		StartTime:   timestamppb.New(start),
		EndTime:     timestamppb.New(end),
	}

	mock.ExpectQuery(`SELECT date, total_decisions, total_alerts\s+FROM aggregates_daily\s+WHERE .*ORDER BY date DESC\s+LIMIT 2001`).
		WithArgs(start, end).
		WillReturnRows(sqlmock.NewRows([]string{"date", "count", "alerts"}).
			AddRow(end.Truncate(24*time.Hour), 120, 12))

	resp, err := s.GetVolumeSeries(context.Background(), req)
	require.NoError(t, err)
	require.Len(t, resp.Points, 1)
	assert.True(t, resp.Meta.Partial)
}

func TestGetVolumeSeries_TruncatesDailyToCap(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	req := &pb.GetVolumeSeriesRequest{
		Granularity: "day",
	}

	rows := sqlmock.NewRows([]string{"date", "count", "alerts"})
	now := time.Now().UTC().Truncate(24 * time.Hour)
	for i := 0; i < MaxPointsDaily+1; i++ {
		rows.AddRow(now.AddDate(0, 0, -i), int64(100+i), int64(10+i))
	}

	mock.ExpectQuery(`SELECT date, total_decisions, total_alerts\s+FROM aggregates_daily\s+WHERE 1=1\s+ORDER BY date DESC\s+LIMIT 2001`).
		WillReturnRows(rows)

	resp, err := s.GetVolumeSeries(context.Background(), req)
	require.NoError(t, err)
	require.Len(t, resp.Points, MaxPointsDaily)
	assert.True(t, resp.Meta.Partial)
	assert.True(t, resp.Points[0].Timestamp.AsTime().Before(resp.Points[len(resp.Points)-1].Timestamp.AsTime()))
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
		TenantId:   "tenant-1",
		RuleImpacts: []*pb.RuleImpact{
			{RuleId: "rule-1", ScoreDelta: 10},
		},
	}

	mock.ExpectBegin()
	mock.ExpectExec("INSERT INTO inference_events").WillReturnResult(sqlmock.NewResult(1, 1))

	// Daily aggregate increment
	mock.ExpectExec("INSERT INTO aggregates_daily").
		WithArgs("tenant-1", sqlmock.AnyArg(), 1, 85, 1).
		WillReturnResult(sqlmock.NewResult(1, 1))

	// Hourly aggregate increment
	mock.ExpectExec("INSERT INTO aggregates_hourly").
		WithArgs("tenant-1", sqlmock.AnyArg(), 1, 85, 1).
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectPrepare("INSERT INTO rule_impacts")
	mock.ExpectExec("INSERT INTO rule_impacts").WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectCommit()

	err = s.LogInferenceEvent(context.Background(), event)
	require.NoError(t, err)
	require.NoError(t, mock.ExpectationsWereMet())
}
