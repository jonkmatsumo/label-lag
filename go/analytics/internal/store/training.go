package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func (s *SQLStore) GetTrainingData(ctx context.Context, cutoff time.Time) ([]*pb.TransactionDetail, []*pb.TransactionDetail, error) {
	trainQuery := `
		SELECT
			fs.record_id,
			fs.user_id,
			em.created_at,
			em.is_train_eligible,
			em.is_pre_fraud,
			gr.amount,
			gr.is_off_hours_txn,
			gr.merchant_risk_score,
			fs.velocity_24h,
			fs.amount_to_avg_ratio_30d,
			fs.balance_volatility_z_score,
			CASE
				WHEN gr.is_fraudulent = TRUE
					 AND em.fraud_confirmed_at IS NOT NULL
					 AND em.fraud_confirmed_at <= $1
				THEN TRUE
				ELSE FALSE
			END AS is_fraudulent,
			COALESCE(gr.fraud_type, ''),
			gr.numerical_features,
			gr.categorical_features
		FROM feature_snapshots fs
		INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
		INNER JOIN generated_records gr ON fs.record_id = gr.record_id
		WHERE gr.transaction_timestamp < $1
		  AND em.is_train_eligible = TRUE
		ORDER BY gr.transaction_timestamp
	`

	testQuery := `
		SELECT
			fs.record_id,
			fs.user_id,
			em.created_at,
			em.is_train_eligible,
			em.is_pre_fraud,
			gr.amount,
			gr.is_off_hours_txn,
			gr.merchant_risk_score,
			fs.velocity_24h,
			fs.amount_to_avg_ratio_30d,
			fs.balance_volatility_z_score,
			gr.is_fraudulent,
			COALESCE(gr.fraud_type, ''),
			gr.numerical_features,
			gr.categorical_features
		FROM feature_snapshots fs
		INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
		INNER JOIN generated_records gr ON fs.record_id = gr.record_id
		WHERE gr.transaction_timestamp >= $1
		ORDER BY gr.transaction_timestamp
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	trainRecords, err := s.queryTrainingRecords(queryCtx, trainQuery, cutoff)
	if err != nil {
		return nil, nil, err
	}

	testRecords, err := s.queryTrainingRecords(queryCtx, testQuery, cutoff)
	if err != nil {
		return nil, nil, err
	}

	return trainRecords, testRecords, nil
}

func (s *SQLStore) queryTrainingRecords(ctx context.Context, query string, cutoff time.Time) ([]*pb.TransactionDetail, error) {
	rows, err := s.db.QueryContext(ctx, query, cutoff)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var records []*pb.TransactionDetail
	for rows.Next() {
		var tx pb.TransactionDetail
		var createdAt time.Time
		var numFeaturesJSON, catFeaturesJSON []byte
		err := rows.Scan(
			&tx.RecordId,
			&tx.UserId,
			&createdAt,
			&tx.IsTrainEligible,
			&tx.IsPreFraud,
			&tx.Amount,
			&tx.IsOffHoursTxn,
			&tx.MerchantRiskScore,
			&tx.Velocity_24H,
			&tx.AmountToAvgRatio_30D,
			&tx.BalanceVolatilityZScore,
			&tx.IsFraudulent,
			&tx.FraudType,
			&numFeaturesJSON,
			&catFeaturesJSON,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan training record: %v", err)
		}
		tx.CreatedAt = timestamppb.New(createdAt)

		if len(numFeaturesJSON) > 0 {
			json.Unmarshal(numFeaturesJSON, &tx.NumericalFeatures)
		}
		if len(catFeaturesJSON) > 0 {
			json.Unmarshal(catFeaturesJSON, &tx.CategoricalFeatures)
		}

		records = append(records, &tx)
	}
	return records, nil
}

func (s *SQLStore) GetBacktestFeatures(ctx context.Context, start, end time.Time) ([]*pb.BacktestFeatureVector, error) {
	query := `
		SELECT
			record_id,
			velocity_24h,
			amount_to_avg_ratio_30d,
			balance_volatility_z_score,
			COALESCE(experimental_signals::text, '{}') as experimental_signals_json
		FROM feature_snapshots
		WHERE computed_at >= $1 AND computed_at <= $2
		ORDER BY computed_at
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, start, end)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var features []*pb.BacktestFeatureVector
	for rows.Next() {
		var f pb.BacktestFeatureVector
		if err := rows.Scan(
			&f.RecordId,
			&f.Velocity_24H,
			&f.AmountToAvgRatio_30D,
			&f.BalanceVolatilityZScore,
			&f.ExperimentalSignalsJson,
		); err != nil {
			return nil, fmt.Errorf("failed to scan backtest feature: %v", err)
		}
		features = append(features, &f)
	}

	return features, nil
}

func (s *SQLStore) SaveBacktestResult(ctx context.Context, res *pb.BacktestResult) error {
	metricsJSON, _ := json.Marshal(res.Metrics)

	query := `
		INSERT INTO backtest_results (
			job_id, rule_id, ruleset_version, start_date, end_date,
			metrics, completed_at, error
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (job_id) DO UPDATE SET
			metrics = EXCLUDED.metrics,
			completed_at = EXCLUDED.completed_at,
			error = EXCLUDED.error
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err := s.db.ExecContext(queryCtx, query,
		res.JobId, res.RuleId, res.RulesetVersion,
		res.StartDate.AsTime(), res.EndDate.AsTime(),
		metricsJSON, res.CompletedAt.AsTime(), res.Error,
	)

	if err != nil {
		return db.MapDBError(err)
	}
	return nil
}

func (s *SQLStore) ListBacktestResults(ctx context.Context, ruleID string, start, end *time.Time) ([]*pb.BacktestResult, error) {
	query := `
		SELECT
			job_id, rule_id, ruleset_version, start_date, end_date,
			metrics, completed_at, error
		FROM backtest_results
		WHERE 1=1
	`
	args := []interface{}{}

	if ruleID != "" {
		args = append(args, ruleID)
		query += fmt.Sprintf(" AND rule_id = $%d", len(args))
	}
	if start != nil {
		args = append(args, *start)
		query += fmt.Sprintf(" AND completed_at >= $%d", len(args))
	}
	if end != nil {
		args = append(args, *end)
		query += fmt.Sprintf(" AND completed_at <= $%d", len(args))
	}

	query += " ORDER BY completed_at DESC LIMIT 100"

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var results []*pb.BacktestResult
	for rows.Next() {
		var res pb.BacktestResult
		var start, end, completed time.Time
		var metricsJSON []byte
		var rID sql.NullString

		if err := rows.Scan(
			&res.JobId, &rID, &res.RulesetVersion, &start, &end,
			&metricsJSON, &completed, &res.Error,
		); err != nil {
			return nil, fmt.Errorf("failed to scan backtest result: %v", err)
		}

		res.RuleId = rID.String
		res.StartDate = timestamppb.New(start)
		res.EndDate = timestamppb.New(end)
		res.CompletedAt = timestamppb.New(completed)

		var metrics pb.BacktestMetrics
		if err := json.Unmarshal(metricsJSON, &metrics); err == nil {
			res.Metrics = &metrics
		}
		results = append(results, &res)
	}
	return results, nil
}

func (s *SQLStore) GetBacktestResult(ctx context.Context, jobID string) (*pb.BacktestResult, error) {
	query := `
		SELECT
			job_id, rule_id, ruleset_version, start_date, end_date,
			metrics, completed_at, error
		FROM backtest_results
		WHERE job_id = $1
	`

	var res pb.BacktestResult
	var start, end, completed time.Time
	var metricsJSON []byte
	var rID sql.NullString

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, query, jobID).Scan(
		&res.JobId, &rID, &res.RulesetVersion, &start, &end,
		&metricsJSON, &completed, &res.Error,
	)

	if err != nil {
		return nil, db.MapDBError(err)
	}

	res.RuleId = rID.String
	res.StartDate = timestamppb.New(start)
	res.EndDate = timestamppb.New(end)
	res.CompletedAt = timestamppb.New(completed)

	var metrics pb.BacktestMetrics
	if err := json.Unmarshal(metricsJSON, &metrics); err == nil {
		res.Metrics = &metrics
	}

	return &res, nil
}
