package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// ... previous methods ...

func (s *SQLStore) StoreGeneratedData(ctx context.Context, records []*pb.GeneratedRecord, metadata []*pb.EvaluationMetadata) (int64, error) {
	if len(records) == 0 {
		return 0, nil
	}

	txCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	tx, err := s.db.BeginTx(txCtx, nil)
	if err != nil {
		return 0, db.MapDBError(fmt.Errorf("failed to begin transaction: %w", err))
	}
	defer tx.Rollback()

	// Insert generated records
	recordQuery := `
		INSERT INTO generated_records (
			record_id, user_id, full_name, email, phone, transaction_timestamp,
			is_off_hours_txn, available_balance, balance_to_transaction_ratio,
			avg_available_balance_30d, balance_volatility_z_score,
			bank_connections_count_24h, bank_connections_count_7d,
			bank_connections_avg_30d, amount, amount_to_avg_ratio,
			merchant_risk_score, is_returned, email_changed_at, phone_changed_at,
			is_fraudulent, fraud_type,
			numerical_features, categorical_features
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
		)
	`
	for _, r := range records {
		select {
		case <-txCtx.Done():
			return 0, db.MapDBError(txCtx.Err())
		default:
		}
		numFeaturesJSON, _ := json.Marshal(r.NumericalFeatures)
		catFeaturesJSON, _ := json.Marshal(r.CategoricalFeatures)

		_, err := tx.ExecContext(txCtx, recordQuery,
			r.RecordId, r.UserId, r.FullName, r.Email, r.Phone,
			r.TransactionTimestamp.AsTime(), r.IsOffHoursTxn, r.AvailableBalance,
			r.BalanceToTransactionRatio, r.AvgAvailableBalance_30D,
			r.BalanceVolatilityZScore, r.BankConnectionsCount_24H,
			r.BankConnectionsCount_7D, r.BankConnectionsAvg_30D,
			r.Amount, r.AmountToAvgRatio, r.MerchantRiskScore, r.IsReturned,
			r.EmailChangedAt.AsTime(), r.PhoneChangedAt.AsTime(),
			r.IsFraudulent, r.FraudType,
			numFeaturesJSON, catFeaturesJSON,
		)
		if err != nil {
			return 0, db.MapDBError(fmt.Errorf("failed to insert record %s: %w", r.RecordId, err))
		}
	}

	// Insert evaluation metadata
	metadataQuery := `
		INSERT INTO evaluation_metadata (
			user_id, record_id, sequence_number, fraud_confirmed_at,
			is_pre_fraud, days_to_fraud, is_train_eligible
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7
		)
	`

	for _, m := range metadata {
		select {
		case <-txCtx.Done():
			return 0, db.MapDBError(txCtx.Err())
		default:
		}
		var fraudConfirmedAt sql.NullTime
		if m.FraudConfirmedAt != nil {
			fraudConfirmedAt = sql.NullTime{Time: m.FraudConfirmedAt.AsTime(), Valid: true}
		}

		_, err := tx.ExecContext(txCtx, metadataQuery,
			m.UserId, m.RecordId, m.SequenceNumber,
			fraudConfirmedAt, m.IsPreFraud, m.DaysToFraud,
			m.IsTrainEligible,
		)
		if err != nil {
			return 0, db.MapDBError(fmt.Errorf("failed to insert metadata for record %s: %w", m.RecordId, err))
		}
	}

	if err := tx.Commit(); err != nil {
		return 0, db.MapDBError(fmt.Errorf("transaction commit failed: %w", err))
	}

	return int64(len(records)), nil
}

func (s *SQLStore) GetFeatureSample(ctx context.Context, sampleSize int32, stratify bool, tenantID string) ([]*pb.FeatureSample, error) {
	pgVersion, _ := getPostgresVersion(ctx, s.db)
	stats := tableStats{}
	var err error
	if tenantID != "" {
		queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
		defer cancel()
		err = s.db.QueryRowContext(queryCtx, `
			SELECT COALESCE(MIN(gr.id), 0), COALESCE(MAX(gr.id), 0), COUNT(*)
			FROM generated_records gr
			INNER JOIN inference_events ie ON gr.record_id = ie.request_id
			WHERE ie.tenant_id = $1
		`, tenantID).Scan(&stats.minID, &stats.maxID, &stats.totalCount)
	} else {
		stats, err = getTableStats(ctx, s.db, "generated_records")
	}
	if err != nil {
		return nil, db.MapDBError(fmt.Errorf("failed to get table stats: %w", err))
	}

	if stats.totalCount == 0 {
		return []*pb.FeatureSample{}, nil
	}

	var samples []*pb.FeatureSample

	if stratify {
		// Get fraud rate for stratification
		var fraudRate float64
		queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
		defer cancel()
		fraudRateQuery := "SELECT CAST(SUM(CASE WHEN gr.is_fraudulent THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) FROM generated_records gr"
		var fraudArgs []interface{}
		if tenantID != "" {
			fraudRateQuery += " INNER JOIN inference_events ie ON gr.record_id = ie.request_id WHERE ie.tenant_id = $1"
			fraudArgs = append(fraudArgs, tenantID)
		}
		err = s.db.QueryRowContext(queryCtx, fraudRateQuery, fraudArgs...).Scan(&fraudRate)
		if err != nil {
			fraudRate = 0.0 // Fallback
		}

		fraudTarget, nonFraudTarget := calculateStratifiedCounts(stats.totalCount, fraudRate, sampleSize, 10)

		// Sample fraud
		if fraudTarget > 0 {
			fSamples, err := s.sampleClass(ctx, true, fraudTarget, pgVersion, stats, tenantID)
			if err != nil {
				return nil, db.MapDBError(fmt.Errorf("failed to sample fraud class: %w", err))
			}
			samples = append(samples, fSamples...)
		}

		// Sample non-fraud
		if nonFraudTarget > 0 {
			nfSamples, err := s.sampleClass(ctx, false, nonFraudTarget, pgVersion, stats, tenantID)
			if err != nil {
				return nil, db.MapDBError(fmt.Errorf("failed to sample non-fraud class: %w", err))
			}
			samples = append(samples, nfSamples...)
		}
	}

	if len(samples) == 0 {
		// Fallback to generic sampling if stratification failed or not requested
		samples, err = s.sampleGeneric(ctx, sampleSize, pgVersion, stats, tenantID)
		if err != nil {
			return nil, db.MapDBError(err)
		}
	}

	return samples, nil
}

func (s *SQLStore) sampleClass(ctx context.Context, isFraudulent bool, limit int32, pgVersion int, stats tableStats, tenantID string) ([]*pb.FeatureSample, error) {
	tenantJoin := ""
	tenantPredicate := ""
	queryArgs := []interface{}{}
	if tenantID != "" {
		tenantJoin = " INNER JOIN inference_events ie ON gr.record_id = ie.request_id"
		tenantPredicate = " AND ie.tenant_id = $1"
		queryArgs = append(queryArgs, tenantID)
	}

	var query string
	if pgVersion >= 16 && stats.totalCount > 100000 {
		fraction := float64(limit) / float64(stats.totalCount)
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr TABLESAMPLE SYSTEM (%f)
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			%s
			WHERE gr.is_fraudulent = %t%s
			LIMIT %d`, fraction*100, tenantJoin, isFraudulent, tenantPredicate, limit)
	} else if stats.maxID > stats.minID && stats.totalCount > 10000 {
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			%s
			WHERE gr.id BETWEEN %d AND %d AND gr.is_fraudulent = %t%s
			LIMIT %d`, tenantJoin, stats.minID, stats.maxID, isFraudulent, tenantPredicate, limit)
	} else {
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			%s
			WHERE gr.is_fraudulent = %t%s
			ORDER BY RANDOM()
			LIMIT %d`, tenantJoin, isFraudulent, tenantPredicate, limit)
	}
	return s.executeQuery(ctx, query, queryArgs...)
}

func (s *SQLStore) sampleGeneric(ctx context.Context, limit int32, pgVersion int, stats tableStats, tenantID string) ([]*pb.FeatureSample, error) {
	tenantJoin := ""
	tenantWhere := ""
	tenantPredicate := ""
	queryArgs := []interface{}{}
	if tenantID != "" {
		tenantJoin = " INNER JOIN inference_events ie ON gr.record_id = ie.request_id"
		tenantWhere = " WHERE ie.tenant_id = $1"
		tenantPredicate = " AND ie.tenant_id = $1"
		queryArgs = append(queryArgs, tenantID)
	}

	var query string
	if pgVersion >= 16 && stats.totalCount > 100000 {
		fraction := float64(limit) / float64(stats.totalCount)
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr TABLESAMPLE SYSTEM (%f)
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			%s
			%s
			LIMIT %d`, fraction*100, tenantJoin, tenantWhere, limit)
	} else if stats.maxID > stats.minID && stats.totalCount > 10000 {
		// Uniform ID sampling hint
		step := (stats.maxID - stats.minID) / int64(limit)
		if step < 1 {
			step = 1
		}
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			%s
			WHERE gr.id IN (SELECT generate_series(%d, %d, %d))%s
			LIMIT %d`, tenantJoin, stats.minID, stats.maxID, step, tenantPredicate, limit)
	} else {
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			%s
			%s
			ORDER BY RANDOM()
			LIMIT %d`, tenantJoin, tenantWhere, limit)
	}
	return s.executeQuery(ctx, query, queryArgs...)
}

func (s *SQLStore) executeQuery(ctx context.Context, query string, args ...interface{}) ([]*pb.FeatureSample, error) {
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var samples []*pb.FeatureSample
	for rows.Next() {
		var sample pb.FeatureSample
		err := rows.Scan(
			&sample.RecordId,
			&sample.IsFraudulent,
			&sample.Velocity_24H,
			&sample.AmountToAvgRatio_30D,
			&sample.BalanceVolatilityZScore,
		)
		if err != nil {
			return nil, db.MapDBError(err)
		}
		samples = append(samples, &sample)
	}
	return samples, nil
}

func calculateStratifiedCounts(total int64, fraudRate float64, sampleSize int32, minPerClass int32) (int32, int32) {
	if total == 0 {
		return 0, 0
	}

	fraudCount := int64(float64(total) * fraudRate)
	nonFraudCount := total - fraudCount

	if total < int64(minPerClass)*2 {
		fraudSample := int32(fraudCount)
		if fraudSample > sampleSize/2 {
			fraudSample = sampleSize / 2
		}
		nonFraudSample := int32(nonFraudCount)
		if nonFraudSample > sampleSize-fraudSample {
			nonFraudSample = sampleSize - fraudSample
		}
		return fraudSample, nonFraudSample
	}

	fraudSample := int32(float64(sampleSize) * (float64(fraudCount) / float64(total)))
	nonFraudSample := sampleSize - fraudSample

	if fraudSample < minPerClass && fraudCount >= int64(minPerClass) {
		fraudSample = minPerClass
		nonFraudSample = sampleSize - fraudSample
		if nonFraudSample < 0 {
			nonFraudSample = 0
		}
	}
	return fraudSample, nonFraudSample
}

func (s *SQLStore) GetDriftWindow(ctx context.Context, cutoff time.Time, tenantID string) ([]*pb.TransactionDetail, error) {
	query := `
		SELECT
			fs.record_id,
			fs.user_id,
			fs.computed_at,
			fs.velocity_24h,
			fs.amount_to_avg_ratio_30d,
			fs.balance_volatility_z_score
		FROM feature_snapshots fs
	`
	args := []interface{}{cutoff}
	where := " WHERE fs.computed_at >= $1"
	if tenantID != "" {
		query += " INNER JOIN inference_events ie ON fs.record_id = ie.request_id"
		where += " AND ie.tenant_id = $2"
		args = append(args, tenantID)
	}
	query += where + " ORDER BY fs.computed_at DESC"

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(fmt.Errorf("failed to query drift window: %w", err))
	}
	defer rows.Close()

	var txs []*pb.TransactionDetail
	for rows.Next() {
		var tx pb.TransactionDetail
		var createdAt time.Time
		if err := rows.Scan(
			&tx.RecordId,
			&tx.UserId,
			&createdAt,
			&tx.Velocity_24H,
			&tx.AmountToAvgRatio_30D,
			&tx.BalanceVolatilityZScore,
		); err != nil {
			return nil, db.MapDBError(fmt.Errorf("failed to scan drift window record: %w", err))
		}
		tx.CreatedAt = timestamppb.New(createdAt)
		txs = append(txs, &tx)
	}

	return txs, nil
}

func (s *SQLStore) GetInferenceScores(ctx context.Context, cutoff time.Time, tenantID string) ([]int32, error) {
	query := `
		SELECT final_score
		FROM inference_events
	`
	args := []interface{}{cutoff}
	where := " WHERE ts >= $1"
	if tenantID != "" {
		where += " AND tenant_id = $2"
		args = append(args, tenantID)
	}
	query += where + " ORDER BY ts DESC"

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(fmt.Errorf("failed to query inference scores: %w", err))
	}
	defer rows.Close()

	var scores []int32
	for rows.Next() {
		var score int32
		if err := rows.Scan(&score); err != nil {
			return nil, db.MapDBError(fmt.Errorf("failed to scan score: %w", err))
		}
		scores = append(scores, score)
	}

	return scores, nil
}

func (s *SQLStore) GetGenerationJob(ctx context.Context, key string) (*pb.GenerateDataResponse, string, error) {
	query := `
		SELECT status, total_records, fraud_records, features_materialized, error
		FROM generation_jobs
		WHERE idempotency_key = $1
	`
	var status string
	var total, fraud, materialized sql.NullInt64
	var errMsg sql.NullString

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, query, key).Scan(&status, &total, &fraud, &materialized, &errMsg)
	if err == sql.ErrNoRows {
		return nil, "", nil
	}
	if err != nil {
		return nil, "", db.MapDBError(err)
	}

	resp := &pb.GenerateDataResponse{
		Success:              status == "completed",
		TotalRecords:         total.Int64,
		FraudRecords:         fraud.Int64,
		FeaturesMaterialized: materialized.Int64,
		Error:                errMsg.String,
	}

	return resp, status, nil
}

func (s *SQLStore) CreateGenerationJob(ctx context.Context, key string) error {
	query := `
		INSERT INTO generation_jobs (idempotency_key, status, created_at)
		VALUES ($1, 'in_progress', NOW())
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err := s.db.ExecContext(queryCtx, query, key)
	return db.MapDBError(err)
}

func (s *SQLStore) CompleteGenerationJob(ctx context.Context, key string, resp *pb.GenerateDataResponse) error {
	query := `
		UPDATE generation_jobs
		SET status = 'completed',
		    total_records = $1,
		    fraud_records = $2,
		    features_materialized = $3,
		    completed_at = NOW()
		WHERE idempotency_key = $4
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err := s.db.ExecContext(queryCtx, query, resp.TotalRecords, resp.FraudRecords, resp.FeaturesMaterialized, key)
	return db.MapDBError(err)
}

func (s *SQLStore) FailGenerationJob(ctx context.Context, key string, errMsg string) error {
	query := `
		UPDATE generation_jobs
		SET status = 'failed',
		    error = $1,
		    completed_at = NOW()
		WHERE idempotency_key = $2
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err := s.db.ExecContext(queryCtx, query, errMsg, key)
	return db.MapDBError(err)
}

func (s *SQLStore) ClearAllData(ctx context.Context) ([]string, error) {
	tables := []string{"generated_records", "feature_snapshots", "rules", "rule_versions", "backtest_results", "inference_events"}
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	tx, err := s.db.BeginTx(queryCtx, nil)
	if err != nil {
		return nil, db.MapDBError(fmt.Errorf("failed to begin transaction: %w", err))
	}
	defer tx.Rollback()

	for _, table := range tables {
		_, err := tx.ExecContext(queryCtx, fmt.Sprintf("TRUNCATE TABLE %s RESTART IDENTITY CASCADE", table))
		if err != nil {
			return nil, db.MapDBError(fmt.Errorf("failed to truncate table %s: %w", table, err))
		}
	}

	if err := tx.Commit(); err != nil {
		return nil, db.MapDBError(fmt.Errorf("failed to commit transaction: %w", err))
	}

	return tables, nil
}

func (s *SQLStore) LogInferenceEvent(ctx context.Context, event *pb.InferenceEvent) error {
	if event == nil || event.RequestId == "" {
		return status.Error(codes.InvalidArgument, "event and request_id required")
	}
	if event.TenantId == "" {
		return status.Error(codes.InvalidArgument, "tenant_id required")
	}

	txCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	tx, err := s.db.BeginTx(txCtx, nil)
	if err != nil {
		return db.MapDBError(fmt.Errorf("failed to begin transaction: %w", err))
	}
	defer tx.Rollback()

	// Convert rule impacts to JSON for the de-normalized column
	impactsJSON, err := json.Marshal(event.RuleImpacts)
	if err != nil {
		return fmt.Errorf("failed to marshal rule impacts: %w", err)
	}

	// Insert into inference_events
	queryEvent := `
		INSERT INTO inference_events (
			request_id, ts, model_version, rules_version,
			model_score, final_score, rule_impacts, user_id, decision, tenant_id
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`
	_, err = tx.ExecContext(txCtx, queryEvent,
		event.RequestId,
		event.Timestamp.AsTime(),
		event.ModelVersion,
		event.RulesVersion,
		event.ModelScore,
		event.FinalScore,
		impactsJSON,
		event.UserId,
		event.Decision,
		event.TenantId,
	)
	if err != nil {
		return db.MapDBError(fmt.Errorf("failed to insert inference event: %w", err))
	}

	// Increment aggregates
	isAlert := event.Decision == DecisionReview || event.Decision == DecisionReject
	alertIncr := 0
	if isAlert {
		alertIncr = 1
	}
	rulesFired := len(event.RuleImpacts)
	ts := time.Now()
	if event.Timestamp != nil {
		ts = event.Timestamp.AsTime()
	}
	hour := ts.Truncate(time.Hour)
	date := ts.Truncate(24 * time.Hour)

	queryDaily := `
		INSERT INTO aggregates_daily (tenant_id, date, total_decisions, total_alerts, sum_score, rules_fired_total)
		VALUES ($1, $2, 1, $3, $4, $5)
		ON CONFLICT (tenant_id, date) DO UPDATE SET
			total_decisions = aggregates_daily.total_decisions + 1,
			total_alerts = aggregates_daily.total_alerts + EXCLUDED.total_alerts,
			sum_score = aggregates_daily.sum_score + EXCLUDED.sum_score,
			rules_fired_total = aggregates_daily.rules_fired_total + EXCLUDED.rules_fired_total
	`
	_, err = tx.ExecContext(txCtx, queryDaily, event.TenantId, date, alertIncr, event.FinalScore, rulesFired)
	if err != nil {
		return db.MapDBError(fmt.Errorf("failed to update daily aggregates: %w", err))
	}

	queryHourly := `
		INSERT INTO aggregates_hourly (tenant_id, hour, total_decisions, total_alerts, sum_score, rules_fired_total)
		VALUES ($1, $2, 1, $3, $4, $5)
		ON CONFLICT (tenant_id, hour) DO UPDATE SET
			total_decisions = aggregates_hourly.total_decisions + 1,
			total_alerts = aggregates_hourly.total_alerts + EXCLUDED.total_alerts,
			sum_score = aggregates_hourly.sum_score + EXCLUDED.sum_score,
			rules_fired_total = aggregates_hourly.rules_fired_total + EXCLUDED.rules_fired_total
	`
	_, err = tx.ExecContext(txCtx, queryHourly, event.TenantId, hour, alertIncr, event.FinalScore, rulesFired)
	if err != nil {
		return db.MapDBError(fmt.Errorf("failed to update hourly aggregates: %w", err))
	}

	// Insert rule impacts
	queryImpact := `
		INSERT INTO rule_impacts (
			request_id, rule_id, is_shadow, score_delta
		) VALUES ($1, $2, $3, $4)
	`
	stmt, err := tx.PrepareContext(txCtx, queryImpact)
	if err != nil {
		return db.MapDBError(fmt.Errorf("failed to prepare impact statement: %w", err))
	}
	defer stmt.Close()

	for _, impact := range event.RuleImpacts {
		_, err := stmt.ExecContext(txCtx,
			event.RequestId,
			impact.RuleId,
			impact.IsShadow,
			impact.ScoreDelta,
		)
		if err != nil {
			return db.MapDBError(fmt.Errorf("failed to insert rule impact: %w", err))
		}
	}

	if err := tx.Commit(); err != nil {
		return db.MapDBError(fmt.Errorf("failed to commit transaction: %w", err))
	}

	return nil
}

func (s *SQLStore) MaterializeFeatures(ctx context.Context) (int64, error) {
	// SQL taken from materialize_features.py
	materializeSQL := `
		INSERT INTO feature_snapshots (
			record_id, user_id, velocity_24h, amount_to_avg_ratio_30d,
			balance_volatility_z_score, experimental_signals, computed_at
		)
		SELECT
			fc.record_id,
			fc.user_id,
			fc.velocity_24h::INTEGER,
			fc.amount_to_avg_ratio_30d::FLOAT,
			fc.balance_volatility_z_score::FLOAT,
			fc.experimental_signals,
			NOW()
		FROM (
			WITH feature_calculations AS (
				SELECT
					gr.record_id,
					gr.user_id,
					gr.transaction_timestamp,
					gr.amount,
					gr.available_balance,
					COUNT(*) OVER (
						PARTITION BY gr.user_id
						ORDER BY gr.transaction_timestamp
						RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
					) AS velocity_24h,
					CASE
						WHEN AVG(gr.amount) OVER (
							PARTITION BY gr.user_id
							ORDER BY gr.transaction_timestamp
							RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
						) > 0
						THEN gr.amount / AVG(gr.amount) OVER (
							PARTITION BY gr.user_id
							ORDER BY gr.transaction_timestamp
							RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
						)
						ELSE 1.0
					END AS amount_to_avg_ratio_30d,
					CASE
						WHEN COALESCE(STDDEV(gr.available_balance) OVER (
							PARTITION BY gr.user_id
							ORDER BY gr.transaction_timestamp
							RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
						), 0) > 0
						THEN (
							gr.available_balance - AVG(gr.available_balance) OVER (
								PARTITION BY gr.user_id
								ORDER BY gr.transaction_timestamp
								RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
							)
						) / STDDEV(gr.available_balance) OVER (
							PARTITION BY gr.user_id
							ORDER BY gr.transaction_timestamp
							RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
						)
						ELSE 0.0
					END AS balance_volatility_z_score,
					COUNT(*) OVER (
						PARTITION BY gr.user_id
						ORDER BY gr.transaction_timestamp
						RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
					) AS velocity_7d,
					MAX(gr.amount) OVER (
						PARTITION BY gr.user_id
						ORDER BY gr.transaction_timestamp
						RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
					) AS max_amount_30d,
					SUM(CASE WHEN gr.is_off_hours_txn THEN 1 ELSE 0 END) OVER (
						PARTITION BY gr.user_id
						ORDER BY gr.transaction_timestamp
						RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
					) AS off_hours_count_7d,
					gr.bank_connections_count_24h,
					gr.merchant_risk_score
				FROM generated_records gr
				WHERE gr.record_id NOT IN (SELECT record_id FROM feature_snapshots)
			)
			SELECT
				record_id,
				user_id,
				velocity_24h,
				amount_to_avg_ratio_30d,
				balance_volatility_z_score,
				jsonb_build_object(
					'velocity_7d', velocity_7d,
					'max_amount_30d', max_amount_30d::FLOAT,
					'off_hours_count_7d', off_hours_count_7d,
					'bank_connections_24h', bank_connections_count_24h,
					'merchant_risk_score', merchant_risk_score
				) AS experimental_signals
			FROM feature_calculations
		) fc
		ON CONFLICT (record_id) DO UPDATE SET
			velocity_24h = EXCLUDED.velocity_24h,
			amount_to_avg_ratio_30d = EXCLUDED.amount_to_avg_ratio_30d,
			balance_volatility_z_score = EXCLUDED.balance_volatility_z_score,
			experimental_signals = EXCLUDED.experimental_signals,
			computed_at = EXCLUDED.computed_at;
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	res, err := s.db.ExecContext(queryCtx, materializeSQL)
	if err != nil {
		return 0, db.MapDBError(fmt.Errorf("failed to materialize features: %w", err))
	}

	rowsAffected, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to get rows affected: %w", err)
	}

	return rowsAffected, nil
}
