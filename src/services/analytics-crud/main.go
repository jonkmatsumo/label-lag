package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	pb "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	_ "github.com/lib/pq"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
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

func (s *server) GetFeatureSample(ctx context.Context, req *pb.GetFeatureSampleRequest) (*pb.GetFeatureSampleResponse, error) {
	sampleSize := req.SampleSize
	if sampleSize == 0 {
		sampleSize = 100
	}

	// Simplified sampling logic for now
	query := `
		SELECT
			gr.record_id,
			gr.is_fraudulent,
			fs.velocity_24h,
			fs.amount_to_avg_ratio_30d,
			fs.balance_volatility_z_score
		FROM generated_records gr
		INNER JOIN feature_snapshots fs ON gr.record_id = fs.record_id
		ORDER BY RANDOM()
		LIMIT $1
	`

	rows, err := s.db.QueryContext(ctx, query, sampleSize)
	if err != nil {
		return nil, fmt.Errorf("failed to query feature sample: %v", err)
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
			return nil, fmt.Errorf("failed to scan feature sample: %v", err)
		}
		samples = append(samples, &sample)
	}

	return &pb.GetFeatureSampleResponse{Samples: samples}, nil
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgresql://synthetic:synthetic_dev_password@localhost:5432/synthetic_data?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatalf("failed to connect to database: %v", err)
	}
	defer db.Close()

	// Configure connection pool
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(time.Hour)

	if err := db.Ping(); err != nil {
		log.Printf("warning: failed to ping database: %v", err)
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterAnalyticsServiceServer(s, &server{db: db})

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(s, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

	// Register reflection service on gRPC server.
	reflection.Register(s)

	log.Printf("server listening at %v", lis.Addr())

	// Handle graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := s.Serve(lis); err != nil {
			log.Fatalf("failed to serve: %v", err)
		}
	}()

	<-stop
	log.Println("shutting down gRPC server...")
	s.GracefulStop()
}
