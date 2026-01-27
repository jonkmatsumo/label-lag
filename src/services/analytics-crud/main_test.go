package main

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	pb "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	"github.com/stretchr/testify/assert"
)

func TestGetDailyStats(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("an error '%s' was not expected when opening a stub database connection", err)
	}
	defer db.Close()

	s := &server{db: db}

	rows := sqlmock.NewRows([]string{"date", "total_transactions", "fraud_count", "fraud_rate", "total_amount", "avg_z_score"}).
		AddRow(time.Now(), 100, 5, 5.0, 1000.0, 0.5)

	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	req := &pb.GetDailyStatsRequest{Days: 30}
	resp, err := s.GetDailyStats(context.Background(), req)

	assert.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Len(t, resp.Stats, 1)
	assert.Equal(t, int64(100), resp.Stats[0].TotalTransactions)
	assert.Equal(t, int64(5), resp.Stats[0].FraudCount)
}

func TestGetOverviewMetrics(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("an error '%s' was not expected when opening a stub database connection", err)
	}
	defer db.Close()

	s := &server{db: db}

	rows := sqlmock.NewRows([]string{
		"total_records", "fraud_records", "unique_users",
		"min_transaction_timestamp", "max_transaction_timestamp",
		"min_created_at", "max_created_at",
		"total_amount", "fraud_amount",
	}).AddRow(1000, 50, 100, time.Now(), time.Now(), time.Now(), time.Now(), 50000.0, 2500.0)

	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	resp, err := s.GetOverviewMetrics(context.Background(), &pb.GetOverviewMetricsRequest{})

	assert.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, int64(1000), resp.TotalRecords)
	assert.Equal(t, int64(50), resp.FraudRecords)
	assert.Equal(t, float64(5.0), resp.FraudRate)
}
