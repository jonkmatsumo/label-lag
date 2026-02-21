package store

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestListJobs_IncludesTenantFilter(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	req := &pb.ListJobsRequest{
		TenantId: "tenant-1",
		Limit:    10,
		Offset:   0,
	}

	mock.ExpectQuery(`SELECT COUNT\(\*\) FROM\s*\(.*FROM jobs WHERE tenant_id = \$1\)\s*as count_query`).
		WithArgs("tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(0))

	mock.ExpectQuery(`SELECT\s+job_id,.*FROM jobs WHERE tenant_id = \$1 ORDER BY created_at DESC, job_id DESC LIMIT 10`).
		WithArgs("tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"job_id", "job_type", "status", "created_at", "started_at", "ended_at", "error_code", "error_message", "params", "metrics",
		}))

	jobs, total, nextCursor, err := s.ListJobs(context.Background(), req)
	require.NoError(t, err)
	_ = nextCursor
	assert.Equal(t, int64(0), total)
	assert.Len(t, jobs, 0)
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestListJobs(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	req := &pb.ListJobsRequest{
		JobType:  "backfill",
		Status:   "completed",
		TenantId: "tenant-1",
		Limit:    10,
		Offset:   0,
	}

	mock.ExpectQuery("SELECT COUNT").
		WithArgs("backfill", "completed", "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(1))

	ts := time.Now()
	mock.ExpectQuery("SELECT job_id, job_type, status, created_at").
		WithArgs("backfill", "completed", "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"job_id", "job_type", "status", "created_at", "started_at", "ended_at", "error_code", "error_message", "params", "metrics",
		}).AddRow("job-1", "backfill", "completed", ts, ts, ts, nil, nil, []byte("{}"), []byte("{}")))

	jobs, total, nextCursor, err := s.ListJobs(context.Background(), req)
	require.NoError(t, err)
	_ = nextCursor
	assert.Equal(t, int64(1), total)
	assert.Len(t, jobs, 1)
	assert.Equal(t, "job-1", jobs[0].JobId)
}

func TestGetJob(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	ts := time.Now()
	mock.ExpectQuery("SELECT job_id").
		WithArgs("job-1", "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"job_id", "job_type", "status", "created_at", "started_at", "ended_at", "error_code", "error_message", "params", "metrics",
		}).AddRow("job-1", "backfill", "running", ts, ts, nil, nil, nil, nil, nil))

	job, err := s.GetJob(context.Background(), "job-1", "tenant-1")
	require.NoError(t, err)
	assert.Equal(t, "job-1", job.JobId)
	assert.Equal(t, "running", job.Status)
}

func TestGetJobEvents(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	ts := time.Now()
	mock.ExpectQuery("SELECT event_id").
		WithArgs("job-1", 10, 0, "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"event_id", "job_id", "event_type", "timestamp", "details",
		}).AddRow(1, "job-1", "started", ts, []byte(`{"foo":"bar"}`)))

	events, err := s.GetJobEvents(context.Background(), &pb.GetJobEventsRequest{
		JobId:    "job-1",
		Limit:    10,
		Offset:   0,
		TenantId: "tenant-1",
	})
	require.NoError(t, err)
	assert.Len(t, events, 1)
	assert.Equal(t, "started", events[0].EventType)
	assert.JSONEq(t, `{"foo":"bar"}`, events[0].DetailsJson)
}

func TestGetJobEvents_WithCursor(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	beforeTS := time.Now().UTC()
	mock.ExpectQuery(`SELECT event_id`).
		WithArgs("job-1", 10, 0, "tenant-1", beforeTS, int64(99)).
		WillReturnRows(sqlmock.NewRows([]string{
			"event_id", "job_id", "event_type", "timestamp", "details",
		}))

	events, err := s.GetJobEvents(context.Background(), &pb.GetJobEventsRequest{
		JobId:    "job-1",
		Limit:    10,
		Offset:   0,
		TenantId: "tenant-1",
		BeforeTs: timestamppb.New(beforeTS),
		BeforeId: 99,
	})
	require.NoError(t, err)
	assert.Len(t, events, 0)
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestListJobs_DateFilter(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	start := time.Now().Add(-time.Hour).UTC()
	end := time.Now().UTC()
	req := &pb.ListJobsRequest{
		StartDate: timestamppb.New(start),
		EndDate:   timestamppb.New(end),
		TenantId:  "tenant-1",
	}

	mock.ExpectQuery("SELECT COUNT").
		WithArgs(start, end, "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(0))

	mock.ExpectQuery("SELECT job_id").
		WithArgs(start, end, "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"job_id", "job_type", "status", "created_at", "started_at", "ended_at", "error_code", "error_message", "params", "metrics",
		}))

	_, _, _, err = s.ListJobs(context.Background(), req)
	require.NoError(t, err)
}
func TestCancelJob(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	mock.ExpectQuery("SELECT status FROM jobs").
		WithArgs("job-1", "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"status"}).AddRow("queued"))

	mock.ExpectExec("UPDATE jobs SET status = 'cancel_requested'").
		WithArgs("job-1", "tenant-1").
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectExec("INSERT INTO job_events").
		WithArgs("job-1", "cancel_requested", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	err = s.CancelJob(context.Background(), "job-1", "tenant-1")
	require.NoError(t, err)
}

func TestRetryJob(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	mock.ExpectQuery("SELECT job_type, params FROM jobs").
		WithArgs("job-1", "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"job_type", "params"}).AddRow("backfill", []byte("{}")))

	mock.ExpectExec("INSERT INTO jobs").
		WithArgs(sqlmock.AnyArg(), "backfill", []byte("{}"), "tenant-1", "job-1").
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectExec("INSERT INTO job_events").
		WithArgs("job-1", "retried", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	newID, err := s.RetryJob(context.Background(), "job-1", "tenant-1")
	require.NoError(t, err)
	assert.NotEmpty(t, newID)
}

func TestGetJobSummary_CanceledContext(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	resp, err := s.GetJobSummary(ctx, &pb.GetJobSummaryRequest{})
	require.Nil(t, resp)
	require.Error(t, err)
	assert.Equal(t, codes.Canceled, status.Code(err))
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestGetJobSummary_DeadlineExceededContext(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	resp, err := s.GetJobSummary(ctx, &pb.GetJobSummaryRequest{})
	require.Nil(t, resp)
	require.Error(t, err)
	assert.Equal(t, codes.DeadlineExceeded, status.Code(err))
	require.NoError(t, mock.ExpectationsWereMet())
}
