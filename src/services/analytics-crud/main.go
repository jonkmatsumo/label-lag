package main

import (
	"context"
	"database/sql"
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
	schemadb "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/internal/db"
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

func (s *server) GetDailyStats(ctx context.Context, req *pb.GetDailyStatsRequest) (*pb.GetDailyStatsResponse, error) {
	days := req.Days
	if days == 0 {
		days = 30
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
	days := req.Days
	if days == 0 {
		days = 7
	}
	limit := req.Limit
	if limit == 0 {
		limit = 1000
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

func (s *server) GetRecentAlerts(ctx context.Context, req *pb.GetRecentAlertsRequest) (*pb.GetRecentAlertsResponse, error) {
	limit := req.Limit
	if limit == 0 {
		limit = 50
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

	// Prepare query: convert slice to postgres array string, e.g. '{t1,t2}'
	// Or use ANY operator with pq.Array.
	// Since we are using standard sql, we will build a param list or use pq.Array if imported.
	// We imported lib/pq as _, so we can use pq.Array if we change import or just build the IN clause.
	// Simpler to use ANY($1) with a literal string array format or multiple params.
	// Let's use ANY($1) and format the array manually to avoid explicit pq dep dependency in main code if possible,
	// but using lib/pq directly is cleaner. We only did _ import, so let's change that if needed.
	// Actually, just formatting '{a,b}' works for text arrays in Postgres.

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

func (s *server) GetFeatureSample(ctx context.Context, req *pb.GetFeatureSampleRequest) (*pb.GetFeatureSampleResponse, error) {
	sampleSize := req.SampleSize
	if sampleSize <= 0 {
		sampleSize = 100
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

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgresql://synthetic:synthetic_dev_password@localhost:5542/synthetic_data?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		slog.Error("failed to open database connection", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	// Configure connection pool
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(time.Hour)

	// Ping is essential to verify connection
	pCtx, pCancel := context.WithTimeout(ctx, 5*time.Second)
	defer pCancel()
	if err := db.PingContext(pCtx); err != nil {
		slog.Error("failed to ping database", "error", err)
		os.Exit(1)
	}
	slog.Info("database connection established")

	// Perform schema validation on startup
	expectedSchema := []schemadb.TableSchema{
		{Name: "evaluation_metadata", Columns: []string{"record_id", "user_id", "created_at", "is_train_eligible", "is_pre_fraud"}},
		{Name: "generated_records", Columns: []string{"record_id", "user_id", "amount", "is_fraudulent", "is_off_hours_txn", "merchant_risk_score"}},
		{Name: "feature_snapshots", Columns: []string{"record_id", "velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score"}},
	}

	if err := schemadb.ValidateSchema(ctx, db, expectedSchema); err != nil {
		slog.Error("database schema validation failed", "error", err)
		os.Exit(1)
	}
	slog.Info("database schema validated")

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
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

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

	<-stop
	slog.Info("shutting down gRPC server...")
	s.GracefulStop()
}
