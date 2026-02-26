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

func TestStoreGeneratedData_DeadlineExceededRollsBack(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	now := time.Now().UTC()
	records := []*pb.GeneratedRecord{
		newGeneratedRecordForTxTest("rec-1", now),
		newGeneratedRecordForTxTest("rec-2", now.Add(time.Second)),
	}

	mock.ExpectBegin()
	mock.ExpectExec("INSERT INTO generated_records").WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectExec("INSERT INTO generated_records").WillReturnError(context.DeadlineExceeded)
	mock.ExpectRollback()

	inserted, err := s.StoreGeneratedData(context.Background(), records, nil)
	require.Error(t, err)
	assert.Equal(t, int64(0), inserted)
	assert.Equal(t, codes.DeadlineExceeded, status.Code(err))
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestLogInferenceEvent_DeadlineExceededRollsBack(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	event := &pb.InferenceEvent{
		RequestId:  "req-timeout",
		Timestamp:  timestamppb.New(time.Now().UTC()),
		TenantId:   "tenant-1",
		UserId:     "user-1",
		ModelScore: 70,
		FinalScore: 80,
		Decision:   DecisionReview,
		RuleImpacts: []*pb.RuleImpact{
			{RuleId: "rule-1", ScoreDelta: 10},
		},
	}

	mock.ExpectBegin()
	mock.ExpectExec("INSERT INTO inference_events").WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectExec("INSERT INTO aggregates_daily").WillReturnError(context.DeadlineExceeded)
	mock.ExpectRollback()

	err = s.LogInferenceEvent(context.Background(), event)
	require.Error(t, err)
	assert.Equal(t, codes.DeadlineExceeded, status.Code(err))
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestStoreGeneratedData_CanceledContextRollsBack(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	now := time.Now().UTC()
	records := []*pb.GeneratedRecord{
		newGeneratedRecordForTxTest("rec-cancel-1", now),
		newGeneratedRecordForTxTest("rec-cancel-2", now.Add(time.Second)),
	}

	mock.ExpectBegin()
	mock.ExpectExec("INSERT INTO generated_records").WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectExec("INSERT INTO generated_records").WillReturnError(context.Canceled)
	mock.ExpectRollback()

	inserted, err := s.StoreGeneratedData(context.Background(), records, nil)
	require.Error(t, err)
	assert.Equal(t, int64(0), inserted)
	assert.Equal(t, codes.Canceled, status.Code(err))
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestLogInferenceEvent_CanceledContextRollsBack(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	event := &pb.InferenceEvent{
		RequestId:  "req-cancel",
		Timestamp:  timestamppb.New(time.Now().UTC()),
		TenantId:   "tenant-1",
		UserId:     "user-1",
		ModelScore: 40,
		FinalScore: 55,
		Decision:   DecisionReview,
		RuleImpacts: []*pb.RuleImpact{
			{RuleId: "rule-1", ScoreDelta: 8},
		},
	}

	mock.ExpectBegin()
	mock.ExpectExec("INSERT INTO inference_events").WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectExec("INSERT INTO aggregates_daily").WillReturnError(context.Canceled)
	mock.ExpectRollback()

	err = s.LogInferenceEvent(context.Background(), event)
	require.Error(t, err)
	assert.Equal(t, codes.Canceled, status.Code(err))
	require.NoError(t, mock.ExpectationsWereMet())
}

func newGeneratedRecordForTxTest(id string, ts time.Time) *pb.GeneratedRecord {
	timestamp := timestamppb.New(ts)
	return &pb.GeneratedRecord{
		RecordId:             id,
		UserId:               "user-1",
		FullName:             "User One",
		Email:                "user@example.com",
		Phone:                "555-0100",
		TransactionTimestamp: timestamp,
		EmailChangedAt:       timestamp,
		PhoneChangedAt:       timestamp,
		NumericalFeatures:    map[string]float64{"n": 1},
		CategoricalFeatures:  map[string]string{"c": "v"},
	}
}
