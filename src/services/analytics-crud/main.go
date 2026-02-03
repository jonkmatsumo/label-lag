package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	pb "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	_ "github.com/lib/pq"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type server struct {
	pb.UnimplementedAnalyticsServiceServer
	db *sql.DB
}

const (
	maxDaysLimit         = 365
	maxTransactionLimit  = 5000
	maxAlertLimit        = 1000
	maxSampleSizeLimit   = 5000
	defaultDailyStatsDay = 30
	defaultTxnDays       = 7
	defaultTxnLimit      = 1000
	defaultAlertLimit    = 50
	defaultSampleSize    = 100
	defaultSearchLimit   = 100
	maxSearchLimit       = 1000
	defaultDatabaseURL   = "postgresql://synthetic:synthetic_dev_password@localhost:5542/synthetic_data?sslmode=disable"
)

func (s *server) GetDailyStats(ctx context.Context, req *pb.GetDailyStatsRequest) (*pb.GetDailyStatsResponse, error) {
	days, err := normalizeDays(req.Days, defaultDailyStatsDay, maxDaysLimit)
	if err != nil {
		return nil, err
	}
	cutoffDate := time.Now().AddDate(0, 0, -int(days))

	query := `
		SELECT
			DATE(em.created_at) as date,
			COUNT(*) as total_transactions,
			SUM(CASE WHEN gr.is_fraudulent THEN 1 ELSE 0 END) as fraud_count,
			ROUND(
				100.0 * SUM(CASE WHEN gr.is_fraudulent THEN 1 ELSE 0 END) / COUNT(*),
				2
			) as fraud_rate,
			COALESCE(SUM(gr.amount), 0) as total_amount,
			ROUND(AVG(fs.balance_volatility_z_score)::numeric, 2) as avg_z_score
		FROM evaluation_metadata em
		LEFT JOIN generated_records gr ON em.record_id = gr.record_id
		LEFT JOIN feature_snapshots fs ON em.record_id = fs.record_id
		WHERE em.created_at >= $1
		GROUP BY DATE(em.created_at)
		ORDER BY date DESC
	`

	rows, err := s.db.QueryContext(ctx, query, cutoffDate)
	if err != nil {
		return nil, fmt.Errorf("failed to query daily stats: %v", err)
	}
	defer rows.Close()

	var stats []*pb.DailyStat
	for rows.Next() {
		var date time.Time
		var s_stat pb.DailyStat
		err := rows.Scan(
			&date,
			&s_stat.TotalTransactions,
			&s_stat.FraudCount,
			&s_stat.FraudRate,
			&s_stat.TotalAmount,
			&s_stat.AvgZScore,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan daily stat: %v", err)
		}
		s_stat.Date = date.Format("2006-01-02")
		stats = append(stats, &s_stat)
	}

	return &pb.GetDailyStatsResponse{Stats: stats}, nil
}

func (s *server) GetTransactionDetails(ctx context.Context, req *pb.GetTransactionDetailsRequest) (*pb.GetTransactionDetailsResponse, error) {
	days, err := normalizeDays(req.Days, defaultTxnDays, maxDaysLimit)
	if err != nil {
		return nil, err
	}
	limit, err := normalizeLimit(req.Limit, defaultTxnLimit, maxTransactionLimit, "limit")
	if err != nil {
		return nil, err
	}
	cutoffDate := time.Now().AddDate(0, 0, -int(days))

	query := `
		SELECT
			em.record_id,
			em.user_id,
			em.created_at,
			em.is_train_eligible,
			em.is_pre_fraud,
			gr.amount,
			gr.is_fraudulent,
			COALESCE(gr.fraud_type, ''),
			gr.is_off_hours_txn,
			gr.merchant_risk_score,
			fs.velocity_24h,
			fs.amount_to_avg_ratio_30d,
			fs.balance_volatility_z_score
		FROM evaluation_metadata em
		LEFT JOIN generated_records gr ON em.record_id = gr.record_id
		LEFT JOIN feature_snapshots fs ON em.record_id = fs.record_id
		WHERE em.created_at >= $1
		ORDER BY em.created_at DESC
		LIMIT $2
	`

	rows, err := s.db.QueryContext(ctx, query, cutoffDate, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query transaction details: %v", err)
	}
	defer rows.Close()

	var txs []*pb.TransactionDetail
	for rows.Next() {
		var tx pb.TransactionDetail
		var createdAt time.Time
		err := rows.Scan(
			&tx.RecordId,
			&tx.UserId,
			&createdAt,
			&tx.IsTrainEligible,
			&tx.IsPreFraud,
			&tx.Amount,
			&tx.IsFraudulent,
			&tx.FraudType,
			&tx.IsOffHoursTxn,
			&tx.MerchantRiskScore,
			&tx.Velocity_24H,
			&tx.AmountToAvgRatio_30D,
			&tx.BalanceVolatilityZScore,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan transaction detail: %v", err)
		}
		tx.CreatedAt = timestamppb.New(createdAt)
		txs = append(txs, &tx)
	}

	return &pb.GetTransactionDetailsResponse{Transactions: txs}, nil
}

func (s *server) SearchTransactions(ctx context.Context, req *pb.SearchTransactionsRequest) (*pb.SearchTransactionsResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}

	limit, err := normalizeLimit(req.Limit, defaultSearchLimit, maxSearchLimit, "limit")
	if err != nil {
		return nil, err
	}
	offset, err := normalizeOffset(req.Offset)
	if err != nil {
		return nil, err
	}

	conditions := make([]string, 0)
	args := make([]any, 0)

	if req.UserId != "" {
		conditions = append(conditions, fmt.Sprintf("user_id = $%d", len(args)+1))
		args = append(args, req.UserId)
	}
	if req.TransactionId != "" {
		conditions = append(conditions, fmt.Sprintf("record_id = $%d", len(args)+1))
		args = append(args, req.TransactionId)
	}
	if req.MinAmount != nil {
		conditions = append(conditions, fmt.Sprintf("amount >= $%d", len(args)+1))
		args = append(args, req.GetMinAmount())
	}
	if req.MaxAmount != nil {
		conditions = append(conditions, fmt.Sprintf("amount <= $%d", len(args)+1))
		args = append(args, req.GetMaxAmount())
	}
	if req.StartDate != "" {
		if parsed, ok := parseISODate(req.StartDate); ok {
			conditions = append(conditions, fmt.Sprintf("transaction_timestamp >= $%d", len(args)+1))
			args = append(args, parsed)
		}
	}
	if req.EndDate != "" {
		if parsed, ok := parseISODate(req.EndDate); ok {
			conditions = append(conditions, fmt.Sprintf("transaction_timestamp <= $%d", len(args)+1))
			args = append(args, parsed)
		}
	}
	if req.IsFraudulent != nil {
		conditions = append(conditions, fmt.Sprintf("is_fraudulent = $%d", len(args)+1))
		args = append(args, req.GetIsFraudulent())
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = " WHERE " + strings.Join(conditions, " AND ")
	}

	countQuery := "SELECT COUNT(*) FROM generated_records" + whereClause
	var total int64
	if err := s.db.QueryRowContext(ctx, countQuery, args...).Scan(&total); err != nil {
		return nil, fmt.Errorf("failed to query transaction count: %v", err)
	}

	query := `
		SELECT
			record_id,
			user_id,
			transaction_timestamp,
			amount,
			is_fraudulent,
			COALESCE(fraud_type, ''),
			is_off_hours_txn,
			merchant_risk_score,
			amount_to_avg_ratio,
			balance_volatility_z_score
		FROM generated_records` + whereClause + fmt.Sprintf(" ORDER BY transaction_timestamp DESC OFFSET $%d LIMIT $%d", len(args)+1, len(args)+2)

	args = append(args, offset, limit)
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query transactions: %v", err)
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
			&tx.Amount,
			&tx.IsFraudulent,
			&tx.FraudType,
			&tx.IsOffHoursTxn,
			&tx.MerchantRiskScore,
			&tx.AmountToAvgRatio_30D,
			&tx.BalanceVolatilityZScore,
		); err != nil {
			return nil, fmt.Errorf("failed to scan transaction: %v", err)
		}

		tx.CreatedAt = timestamppb.New(createdAt)
		tx.IsTrainEligible = true
		tx.IsPreFraud = true
		tx.Velocity_24H = 0
		txs = append(txs, &tx)
	}

	return &pb.SearchTransactionsResponse{Transactions: txs, Total: total}, nil
}

func (s *server) GetRecentAlerts(ctx context.Context, req *pb.GetRecentAlertsRequest) (*pb.GetRecentAlertsResponse, error) {
	limit, err := normalizeLimit(req.Limit, defaultAlertLimit, maxAlertLimit, "limit")
	if err != nil {
		return nil, err
	}

	// Constants taken from data_service.py
	alertThreshold := 80

	query := `
		SELECT * FROM (
			SELECT
				em.record_id,
				em.user_id,
				em.created_at,
				gr.amount,
				gr.is_fraudulent,
				COALESCE(gr.fraud_type, ''),
				gr.merchant_risk_score,
				fs.velocity_24h,
				fs.amount_to_avg_ratio_30d,
				fs.balance_volatility_z_score,
				(
					CASE WHEN fs.velocity_24h > 5 THEN 20 ELSE 0 END +
					CASE WHEN fs.amount_to_avg_ratio_30d > 3.0 THEN 25 ELSE 0 END +
					CASE WHEN fs.balance_volatility_z_score < -2.0 THEN 20 ELSE 0 END +
					CASE WHEN gr.merchant_risk_score > 70 THEN 20 ELSE 0 END +
					CASE WHEN gr.is_off_hours_txn THEN 15 ELSE 0 END
				) as computed_risk_score
			FROM evaluation_metadata em
			INNER JOIN generated_records gr ON em.record_id = gr.record_id
			INNER JOIN feature_snapshots fs ON em.record_id = fs.record_id
			WHERE
				fs.velocity_24h > 5
				OR fs.amount_to_avg_ratio_30d > 3.0
				OR fs.balance_volatility_z_score < -2.0
				OR gr.merchant_risk_score > 70
		) as scored_alerts
		WHERE computed_risk_score >= $1
		ORDER BY created_at DESC
		LIMIT $2
	`

	rows, err := s.db.QueryContext(ctx, query, alertThreshold, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent alerts: %v", err)
	}
	defer rows.Close()

	var alerts []*pb.Alert
	for rows.Next() {
		var alert pb.Alert
		var createdAt time.Time
		err := rows.Scan(
			&alert.RecordId,
			&alert.UserId,
			&createdAt,
			&alert.Amount,
			&alert.IsFraudulent,
			&alert.FraudType,
			&alert.MerchantRiskScore,
			&alert.Velocity_24H,
			&alert.AmountToAvgRatio_30D,
			&alert.BalanceVolatilityZScore,
			&alert.ComputedRiskScore,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan alert: %v", err)
		}
		alert.CreatedAt = timestamppb.New(createdAt)
		alerts = append(alerts, &alert)
	}

	return &pb.GetRecentAlertsResponse{Alerts: alerts}, nil
}

func (s *server) GetOverviewMetrics(ctx context.Context, req *pb.GetOverviewMetricsRequest) (*pb.GetOverviewMetricsResponse, error) {
	query := `
		SELECT
			COUNT(*) as total_records,
			SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) as fraud_records,
			COUNT(DISTINCT user_id) as unique_users,
			MIN(transaction_timestamp) as min_transaction_timestamp,
			MAX(transaction_timestamp) as max_transaction_timestamp,
			MIN(created_at) as min_created_at,
			MAX(created_at) as max_created_at,
			COALESCE(SUM(amount), 0) as total_amount,
			COALESCE(SUM(CASE WHEN is_fraudulent THEN amount ELSE 0 END), 0) as fraud_amount
		FROM generated_records
	`

	var resp pb.GetOverviewMetricsResponse
	var minTx, maxTx, minCr, maxCr sql.NullTime

	err := s.db.QueryRowContext(ctx, query).Scan(
		&resp.TotalRecords,
		&resp.FraudRecords,
		&resp.UniqueUsers,
		&minTx,
		&maxTx,
		&minCr,
		&maxCr,
		&resp.TotalAmount,
		&resp.FraudAmount,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to query overview metrics: %v", err)
	}

	if resp.TotalRecords > 0 {
		resp.FraudRate = (float64(resp.FraudRecords) / float64(resp.TotalRecords)) * 100.0
	}

	if minTx.Valid {
		resp.MinTransactionTimestamp = timestamppb.New(minTx.Time)
	}
	if maxTx.Valid {
		resp.MaxTransactionTimestamp = timestamppb.New(maxTx.Time)
	}
	if minCr.Valid {
		resp.MinCreatedAt = timestamppb.New(minCr.Time)
	}
	if maxCr.Valid {
		resp.MaxCreatedAt = timestamppb.New(maxCr.Time)
	}

	return &resp, nil
}

func (s *server) GetDatasetFingerprint(ctx context.Context, req *pb.GetDatasetFingerprintRequest) (*pb.GetDatasetFingerprintResponse, error) {
	queryGR := `
		SELECT
			COUNT(*) as count,
			MAX(created_at) as max_created_at,
			MAX(transaction_timestamp) as max_transaction_timestamp,
			MAX(id) as max_id
		FROM generated_records
	`
	queryFS := `
		SELECT
			COUNT(*) as count,
			MAX(computed_at) as max_computed_at,
			MAX(snapshot_id) as max_snapshot_id
		FROM feature_snapshots
	`

	resp := &pb.GetDatasetFingerprintResponse{
		GeneratedRecords: &pb.TableFingerprint{},
		FeatureSnapshots: &pb.TableFingerprint{},
	}

	var maxCr, maxTx sql.NullTime
	var maxId sql.NullInt64

	err := s.db.QueryRowContext(ctx, queryGR).Scan(
		&resp.GeneratedRecords.Count,
		&maxCr,
		&maxTx,
		&maxId,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to query generated_records fingerprint: %v", err)
	}
	if maxCr.Valid {
		resp.GeneratedRecords.MaxCreatedAt = timestamppb.New(maxCr.Time)
	}
	if maxTx.Valid {
		resp.GeneratedRecords.MaxTimestamp = timestamppb.New(maxTx.Time)
	}
	if maxId.Valid {
		resp.GeneratedRecords.MaxId = maxId.Int64
	}

	var maxComp sql.NullTime
	var maxSnapshotId sql.NullInt64

	err = s.db.QueryRowContext(ctx, queryFS).Scan(
		&resp.FeatureSnapshots.Count,
		&maxComp,
		&maxSnapshotId,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to query feature_snapshots fingerprint: %v", err)
	}
	if maxComp.Valid {
		resp.FeatureSnapshots.MaxCreatedAt = timestamppb.New(maxComp.Time)
	}
	if maxSnapshotId.Valid {
		resp.FeatureSnapshots.MaxId = maxSnapshotId.Int64
	}

	return resp, nil
}

// Helper functions for advanced sampling

func getPostgresVersion(ctx context.Context, db *sql.DB) (int, error) {
	var versionStr string
	err := db.QueryRowContext(ctx, "SELECT version()").Scan(&versionStr)
	if err != nil {
		return 0, err
	}

	// Extract major version from string like "PostgreSQL 16.1 ..."
	parts := strings.Split(versionStr, " ")
	for i, part := range parts {
		if part == "PostgreSQL" && i+1 < len(parts) {
			versionParts := strings.Split(parts[i+1], ".")
			if len(versionParts) > 0 {
				major, err := strconv.Atoi(versionParts[0])
				if err == nil {
					return major, nil
				}
			}
		}
	}
	return 0, fmt.Errorf("could not parse postgres version: %s", versionStr)
}

type tableStats struct {
	minID      int64
	maxID      int64
	totalCount int64
}

func getTableStats(ctx context.Context, db *sql.DB, table string) (tableStats, error) {
	var stats tableStats
	query := fmt.Sprintf("SELECT MIN(id), MAX(id), COUNT(*) FROM %s", table)
	err := db.QueryRowContext(ctx, query).Scan(&stats.minID, &stats.maxID, &stats.totalCount)
	if err != nil {
		return stats, err
	}
	return stats, nil
}

func calculateStratifiedCounts(total int64, fraudRate float64, sampleSize int32, minPerClass int32) (int32, int32) {
	if total == 0 {
		return 0, 0
	}

	fraudCount := int64(float64(total) * fraudRate)
	nonFraudCount := total - fraudCount

	// If dataset is too small for minimums, return what we can
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

	// Calculate proportional sample sizes
	fraudSample := int32(float64(sampleSize) * (float64(fraudCount) / float64(total)))
	nonFraudSample := sampleSize - fraudSample

	// Enforce minimums
	if fraudSample < minPerClass && fraudCount >= int64(minPerClass) {
		fraudSample = minPerClass
		nonFraudSample = sampleSize - fraudSample
		if nonFraudSample < 0 {
			nonFraudSample = 0
		}
	}

	if nonFraudSample < minPerClass && nonFraudCount >= int64(minPerClass) {
		nonFraudSample = minPerClass
		fraudSample = sampleSize - nonFraudSample
		if fraudSample < 0 {
			fraudSample = 0
		}
	}

	// Ensure we don't exceed available counts
	if int64(fraudSample) > fraudCount {
		fraudSample = int32(fraudCount)
	}
	if int64(nonFraudSample) > nonFraudCount {
		nonFraudSample = int32(nonFraudCount)
	}

	return fraudSample, nonFraudSample
}

func (s *server) GetSchemaSummary(ctx context.Context, req *pb.GetSchemaSummaryRequest) (*pb.GetSchemaSummaryResponse, error) {
	tableNames := req.TableNames
	if len(tableNames) == 0 {
		tableNames = []string{"generated_records", "feature_snapshots"}
	}

	arrStr := "{" + strings.Join(tableNames, ",") + "}"

	query := `
		SELECT
			table_name,
			column_name,
			data_type,
			is_nullable,
			ordinal_position
		FROM information_schema.columns
		WHERE table_schema = 'public'
		  AND table_name = ANY($1::text[])
		ORDER BY table_name, ordinal_position
	`

	rows, err := s.db.QueryContext(ctx, query, arrStr)
	if err != nil {
		return nil, fmt.Errorf("failed to query schema summary: %v", err)
	}
	defer rows.Close()

	var columns []*pb.ColumnInfo
	for rows.Next() {
		var col pb.ColumnInfo
		err := rows.Scan(
			&col.TableName,
			&col.ColumnName,
			&col.DataType,
			&col.IsNullable,
			&col.OrdinalPosition,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan column info: %v", err)
		}
		// Normalize column name to lowercase
		col.ColumnName = strings.ToLower(col.ColumnName)
		columns = append(columns, &col)
	}

	return &pb.GetSchemaSummaryResponse{Columns: columns}, nil
}

func (s *server) GetTrainingData(ctx context.Context, req *pb.GetTrainingDataRequest) (*pb.GetTrainingDataResponse, error) {
	if req == nil || req.CutoffDate == nil {
		return nil, status.Error(codes.InvalidArgument, "cutoff_date required")
	}
	cutoff := req.CutoffDate.AsTime()

	// Train set query: transaction_timestamp < cutoff AND is_train_eligible = True
	// Knowledge Horizon: Only label fraud if confirmed before cutoff
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
			COALESCE(gr.fraud_type, '')
		FROM feature_snapshots fs
		INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
		INNER JOIN generated_records gr ON fs.record_id = gr.record_id
		WHERE gr.transaction_timestamp < $1
		  AND em.is_train_eligible = TRUE
		ORDER BY gr.transaction_timestamp
	`

	// Test set query: transaction_timestamp >= cutoff
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
			COALESCE(gr.fraud_type, '')
		FROM feature_snapshots fs
		INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
		INNER JOIN generated_records gr ON fs.record_id = gr.record_id
		WHERE gr.transaction_timestamp >= $1
		ORDER BY gr.transaction_timestamp
	`

	trainRecords, err := s.queryTrainingRecords(ctx, trainQuery, cutoff)
	if err != nil {
		return nil, err
	}

	testRecords, err := s.queryTrainingRecords(ctx, testQuery, cutoff)
	if err != nil {
		return nil, err
	}

	return &pb.GetTrainingDataResponse{
		TrainRecords: trainRecords,
		TestRecords:  testRecords,
	}, nil
}

func (s *server) queryTrainingRecords(ctx context.Context, query string, cutoff time.Time) ([]*pb.TransactionDetail, error) {
	rows, err := s.db.QueryContext(ctx, query, cutoff)
	if err != nil {
		return nil, fmt.Errorf("failed to query training records: %v", err)
	}
	defer rows.Close()

	var records []*pb.TransactionDetail
	for rows.Next() {
		var tx pb.TransactionDetail
		var createdAt time.Time
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
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan training record: %v", err)
		}
		tx.CreatedAt = timestamppb.New(createdAt)
		records = append(records, &tx)
	}
	return records, nil
}

func (s *server) GetBacktestFeatures(ctx context.Context, req *pb.GetBacktestFeaturesRequest) (*pb.GetBacktestFeaturesResponse, error) {
	if req == nil || req.StartDate == nil || req.EndDate == nil {
		return nil, status.Error(codes.InvalidArgument, "start_date and end_date required")
	}
	start := req.StartDate.AsTime()
	end := req.EndDate.AsTime()

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

	rows, err := s.db.QueryContext(ctx, query, start, end)
	if err != nil {
		return nil, fmt.Errorf("failed to query backtest features: %v", err)
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

	return &pb.GetBacktestFeaturesResponse{Features: features}, nil
}

func (s *server) SaveBacktestResult(ctx context.Context, req *pb.SaveBacktestResultRequest) (*pb.SaveBacktestResultResponse, error) {
	if req == nil || req.Result == nil {
		return nil, status.Error(codes.InvalidArgument, "result required")
	}
	res := req.Result

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

	_, err := s.db.ExecContext(ctx, query,
		res.JobId, res.RuleId, res.RulesetVersion,
		res.StartDate.AsTime(), res.EndDate.AsTime(),
		metricsJSON, res.CompletedAt.AsTime(), res.Error,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to save backtest result: %v", err)
	}

	return &pb.SaveBacktestResultResponse{Success: true}, nil
}

func (s *server) ListBacktestResults(ctx context.Context, req *pb.ListBacktestResultsRequest) (*pb.ListBacktestResultsResponse, error) {
	query := `
		SELECT
			job_id, rule_id, ruleset_version, start_date, end_date,
			metrics, completed_at, error
		FROM backtest_results
		WHERE 1=1
	`
	args := []interface{}{}

	if req.RuleId != "" {
		args = append(args, req.RuleId)
		query += fmt.Sprintf(" AND rule_id = $%d", len(args))
	}
	if req.StartDate != nil {
		args = append(args, req.StartDate.AsTime())
		query += fmt.Sprintf(" AND completed_at >= $%d", len(args))
	}
	if req.EndDate != nil {
		args = append(args, req.EndDate.AsTime())
		query += fmt.Sprintf(" AND completed_at <= $%d", len(args))
	}

	query += " ORDER BY completed_at DESC LIMIT 100"

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query backtest results: %v", err)
	}
	defer rows.Close()

	var results []*pb.BacktestResult
	for rows.Next() {
		var res pb.BacktestResult
		var start, end, completed time.Time
		var metricsJSON []byte
		var ruleID sql.NullString

		if err := rows.Scan(
			&res.JobId, &ruleID, &res.RulesetVersion, &start, &end,
			&metricsJSON, &completed, &res.Error,
		); err != nil {
			return nil, fmt.Errorf("failed to scan backtest result: %v", err)
		}

		res.RuleId = ruleID.String
		res.StartDate = timestamppb.New(start)
		res.EndDate = timestamppb.New(end)
		res.CompletedAt = timestamppb.New(completed)

		var metrics pb.BacktestMetrics
		if err := json.Unmarshal(metricsJSON, &metrics); err == nil {
			res.Metrics = &metrics
		}

		results = append(results, &res)
	}

	return &pb.ListBacktestResultsResponse{Results: results}, nil
}

func (s *server) GetBacktestResult(ctx context.Context, req *pb.GetBacktestResultRequest) (*pb.GetBacktestResultResponse, error) {
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
	var ruleID sql.NullString

	err := s.db.QueryRowContext(ctx, query, req.JobId).Scan(
		&res.JobId, &ruleID, &res.RulesetVersion, &start, &end,
		&metricsJSON, &completed, &res.Error,
	)

	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "backtest result not found")
	} else if err != nil {
		return nil, fmt.Errorf("failed to query backtest result: %v", err)
	}

	res.RuleId = ruleID.String
	res.StartDate = timestamppb.New(start)
	res.EndDate = timestamppb.New(end)
	res.CompletedAt = timestamppb.New(completed)

	var metrics pb.BacktestMetrics
	if err := json.Unmarshal(metricsJSON, &metrics); err == nil {
		res.Metrics = &metrics
	}

	return &pb.GetBacktestResultResponse{Result: &res}, nil
}

func (s *server) GetFeatureSample(ctx context.Context, req *pb.GetFeatureSampleRequest) (*pb.GetFeatureSampleResponse, error) {

	sampleSize, err := normalizeLimit(req.SampleSize, defaultSampleSize, maxSampleSizeLimit, "sample_size")
	if err != nil {
		return nil, err
	}

	pgVersion, _ := getPostgresVersion(ctx, s.db)
	stats, err := getTableStats(ctx, s.db, "generated_records")
	if err != nil {
		return nil, fmt.Errorf("failed to get table stats: %v", err)
	}

	if stats.totalCount == 0 {
		return &pb.GetFeatureSampleResponse{}, nil
	}

	// Get fraud rate for stratification
	var fraudRate float64
	err = s.db.QueryRowContext(ctx, "SELECT CAST(SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) FROM generated_records").Scan(&fraudRate)
	if err != nil {
		fraudRate = 0.0 // Fallback
	}

	var samples []*pb.FeatureSample
	if req.Stratify {
		fraudTarget, nonFraudTarget := calculateStratifiedCounts(stats.totalCount, fraudRate, sampleSize, 10)

		// Sample fraud
		if fraudTarget > 0 {
			fSamples, err := s.sampleClass(ctx, true, fraudTarget, pgVersion, stats)
			if err != nil {
				return nil, fmt.Errorf("failed to sample fraud class: %v", err)
			}
			samples = append(samples, fSamples...)
		}

		// Sample non-fraud
		if nonFraudTarget > 0 {
			nfSamples, err := s.sampleClass(ctx, false, nonFraudTarget, pgVersion, stats)
			if err != nil {
				return nil, fmt.Errorf("failed to sample non-fraud class: %v", err)
			}
			samples = append(samples, nfSamples...)
		}
	} else {
		samples, err = s.sampleGeneric(ctx, sampleSize, pgVersion, stats)
		if err != nil {
			return nil, err
		}
	}

	return &pb.GetFeatureSampleResponse{Samples: samples}, nil
}

func (s *server) sampleClass(ctx context.Context, isFraudulent bool, limit int32, pgVersion int, stats tableStats) ([]*pb.FeatureSample, error) {
	var query string
	if pgVersion >= 16 && stats.totalCount > 100000 {
		fraction := float64(limit) / float64(stats.totalCount)
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr TABLESAMPLE SYSTEM (%f)
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			WHERE gr.is_fraudulent = %t
			LIMIT %d`, fraction*100, isFraudulent, limit)
	} else if stats.maxID > stats.minID && stats.totalCount > 10000 {
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			WHERE gr.id BETWEEN %d AND %d AND gr.is_fraudulent = %t
			LIMIT %d`, stats.minID, stats.maxID, isFraudulent, limit)
	} else {
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			WHERE gr.is_fraudulent = %t
			ORDER BY RANDOM()
			LIMIT %d`, isFraudulent, limit)
	}
	return s.executeQuery(ctx, query)
}

func (s *server) sampleGeneric(ctx context.Context, limit int32, pgVersion int, stats tableStats) ([]*pb.FeatureSample, error) {
	var query string
	if pgVersion >= 16 && stats.totalCount > 100000 {
		fraction := float64(limit) / float64(stats.totalCount)
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr TABLESAMPLE SYSTEM (%f)
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			LIMIT %d`, fraction*100, limit)
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
			WHERE gr.id IN (SELECT generate_series(%d, %d, %d))
			LIMIT %d`, stats.minID, stats.maxID, step, limit)
	} else {
		query = fmt.Sprintf(`
			SELECT gr.record_id, gr.is_fraudulent, fs.velocity_24h, fs.amount_to_avg_ratio_30d, fs.balance_volatility_z_score
			FROM generated_records gr
			INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
			ORDER BY RANDOM()
			LIMIT %d`, limit)
	}
	return s.executeQuery(ctx, query)
}

func (s *server) executeQuery(ctx context.Context, query string) ([]*pb.FeatureSample, error) {
	rows, err := s.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
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
			return nil, err
		}
		samples = append(samples, &sample)
	}
	return samples, nil
}

func normalizeDays(value, fallback, max int32) (int32, error) {
	if value == 0 {
		value = fallback
	}
	if value < 1 || value > max {
		return 0, status.Errorf(codes.InvalidArgument, "days must be between 1 and %d", max)
	}
	return value, nil
}

func normalizeLimit(value, fallback, max int32, field string) (int32, error) {
	if value == 0 {
		value = fallback
	}
	if value < 1 || value > max {
		return 0, status.Errorf(codes.InvalidArgument, "%s must be between 1 and %d", field, max)
	}
	return value, nil
}

func normalizeOffset(value int32) (int32, error) {
	if value < 0 {
		return 0, status.Error(codes.InvalidArgument, "offset must be >= 0")
	}
	return value, nil
}

func parseISODate(value string) (time.Time, bool) {
	layouts := []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05.999999",
		"2006-01-02T15:04:05",
		"2006-01-02",
	}
	for _, layout := range layouts {
		if parsed, err := time.Parse(layout, value); err == nil {
			return parsed, true
		}
	}
	return time.Time{}, false
}

// loggingInterceptor logs the details of each gRPC request and response.
func loggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()

	// Create context with logger loaded with method info
	logger := slog.With("method", info.FullMethod)

	resp, err := handler(ctx, req)

	duration := time.Since(start)

	if err != nil {
		st, _ := status.FromError(err)
		logger.Error("request failed",
			"duration", duration,
			"code", st.Code().String(),
			"error", err,
		)
	} else {
		logger.Info("request completed",
			"duration", duration,
			"code", codes.OK.String(),
		)
	}

	return resp, err
}

// initTracer initializes an OTLP exporter, and configures the corresponding trace provider.
func initTracer(ctx context.Context) (*sdktrace.TracerProvider, error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		// Use default temporary endpoint or return nil if we don't want to enforce tracing without config
		// For now, let's default to localhost:4317 if not set, or skip if empty?
		// Usually in k8s/docker it's set. If not set, maybe disable tracing?
		// Let's check if OTEL_EXPORTER_OTLP_ENDPOINT is set.
		return nil, nil
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName("analytics-crud"),
			semconv.ServiceVersion("0.1.0"),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Set up trace exporter
	traceExporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithInsecure(),
		otlptracegrpc.WithEndpoint(endpoint),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace exporter: %w", err)
	}

	// Register the trace exporter with a TracerProvider, using a batch
	// span processor to aggregate spans before export.
	bsp := sdktrace.NewBatchSpanProcessor(traceExporter)
	tracerProvider := sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
		sdktrace.WithResource(res),
		sdktrace.WithSpanProcessor(bsp),
	)

	// set global propagator to tracecontext (the default is no-op).
	otel.SetTextMapPropagator(propagation.TraceContext{})
	otel.SetTracerProvider(tracerProvider)

	return tracerProvider, nil
}

func main() {
	// Configure structured logging
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	// Build context
	ctx := context.Background()

	// Initialize OpenTelemetry
	tp, err := initTracer(ctx)
	if err != nil {
		slog.Error("failed to initialize tracer", "error", err)
	} else if tp != nil {
		defer func() {
			if err := tp.Shutdown(ctx); err != nil {
				slog.Error("failed to shutdown tracer provider", "error", err)
			}
		}()
		slog.Info("opentelemetry tracer initialized")
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}

	dbURL, err := resolveDatabaseURL(os.Getenv)
	if err != nil {
		slog.Error("failed to resolve database url", "error", err)
		os.Exit(1)
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		slog.Error("failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	if err := initDB(db); err != nil {
		slog.Error("failed to initialize database", "error", err)
		os.Exit(1)
	}

	// Configure connection pool
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(time.Hour)

	if err := db.Ping(); err != nil {
		slog.Warn("failed to ping database", "error", err)
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", port))
	if err != nil {
		slog.Error("failed to listen", "error", err)
		os.Exit(1)
	}

	// Add interceptors: logging and otel tracing
	opts := []grpc.ServerOption{
		grpc.UnaryInterceptor(loggingInterceptor),
		grpc.StatsHandler(otelgrpc.NewServerHandler()),
	}
	s := grpc.NewServer(opts...)
	pb.RegisterAnalyticsServiceServer(s, &server{db: db})

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(s, healthServer)
	updateHealthStatus(context.Background(), db, healthServer, logger)

	// Register reflection service on gRPC server.
	reflection.Register(s)

	slog.Info("server listening", "address", lis.Addr())

	// Handle graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := s.Serve(lis); err != nil {
			slog.Error("failed to serve", "error", err)
			os.Exit(1)
		}
	}()

	healthCtx, healthCancel := context.WithCancel(context.Background())
	healthTicker := time.NewTicker(10 * time.Second)
	go func() {
		defer healthTicker.Stop()
		for {
			select {
			case <-healthCtx.Done():
				return
			case <-healthTicker.C:
				updateHealthStatus(context.Background(), db, healthServer, logger)
			}
		}
	}()

	<-stop
	healthCancel()
	slog.Info("shutting down gRPC server...")
	s.GracefulStop()
}

func updateHealthStatus(ctx context.Context, db *sql.DB, healthServer *health.Server, logger *slog.Logger) error {
	if err := db.PingContext(ctx); err != nil {
		if logger != nil {
			logger.Warn("database health check failed", "error", err)
		}
		healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		return err
	}
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	return nil
}

func initDB(db *sql.DB) error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS backtest_results (
			job_id TEXT PRIMARY KEY,
			rule_id TEXT,
			ruleset_version TEXT NOT NULL,
			start_date TIMESTAMP NOT NULL,
			end_date TIMESTAMP NOT NULL,
			metrics JSONB NOT NULL,
			completed_at TIMESTAMP NOT NULL,
			error TEXT
		)`,
		`CREATE INDEX IF NOT EXISTS idx_backtest_results_rule_id ON backtest_results(rule_id)`,
		`CREATE INDEX IF NOT EXISTS idx_backtest_results_completed_at ON backtest_results(completed_at)`,
	}

	for _, q := range queries {
		if _, err := db.Exec(q); err != nil {
			return err
		}
	}
	return nil
}

func resolveDatabaseURL(getenv func(string) string) (string, error) {
	if value := strings.TrimSpace(getenv("DATABASE_URL")); value != "" {
		return value, nil
	}
	allowDefaults := strings.EqualFold(getenv("ANALYTICS_CRUD_ALLOW_INSECURE_DEFAULTS"), "true") ||
		strings.EqualFold(getenv("ANALYTICS_CRUD_ALLOW_INSECURE_DEFAULTS"), "1")
	if allowDefaults {
		return defaultDatabaseURL, nil
	}
	return "", fmt.Errorf("DATABASE_URL is required")
}
