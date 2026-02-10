package store

import (
	"context"
	"time"

	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
)

func (s *SQLStore) GetBacktestFeatures(ctx context.Context, start, end time.Time) ([]*pb.BacktestFeatureVector, error) {
	return nil, nil
}

func (s *SQLStore) SaveBacktestResult(ctx context.Context, res *pb.BacktestResult) error {
	return nil
}

func (s *SQLStore) ListBacktestResults(ctx context.Context, ruleID string, start, end *time.Time, limit, offset int32) ([]*pb.BacktestResult, error) {
	return nil, nil
}

func (s *SQLStore) GetBacktestResult(ctx context.Context, jobID string) (*pb.BacktestResult, error) {
	return nil, nil
}

func (s *SQLStore) GetTrainingData(ctx context.Context, cutoff time.Time) (train []*pb.TransactionDetail, test []*pb.TransactionDetail, err error) {
	return nil, nil, nil
}
