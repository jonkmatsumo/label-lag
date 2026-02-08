package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func (s *SQLStore) GetDailyStats(ctx context.Context, cutoffDate time.Time) ([]*pb.DailyStat, error) {
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
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, cutoffDate)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var stats []*pb.DailyStat
	for rows.Next() {
		var stat pb.DailyStat
		var date time.Time
		if err := rows.Scan(
			&date,
			&stat.TotalTransactions,
			&stat.FraudCount,
			&stat.FraudRate,
			&stat.TotalAmount,
			&stat.AvgZScore,
		); err != nil {
			return nil, fmt.Errorf("failed to scan daily stat: %v", err)
		}
		stat.Date = date.Format("2006-01-02")
		stats = append(stats, &stat)
	}
	return stats, nil
}

func (s *SQLStore) GetTransactionDetails(ctx context.Context, cutoffDate time.Time, limit int32) ([]*pb.TransactionDetail, error) {
	query := `
		SELECT
			em.record_id,
			em.user_id,
			em.created_at,
			em.is_train_eligible,
			em.is_pre_fraud,
			gr.amount,
			gr.is_off_hours_txn,
			gr.merchant_risk_score,
			fs.velocity_24h,
			fs.amount_to_avg_ratio_30d,
			fs.balance_volatility_z_score,
			gr.numerical_features,
			gr.categorical_features
		FROM evaluation_metadata em
		LEFT JOIN generated_records gr ON em.record_id = gr.record_id
		LEFT JOIN feature_snapshots fs ON em.record_id = fs.record_id
		WHERE em.created_at >= $1
		ORDER BY em.created_at DESC
		LIMIT $2
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, cutoffDate, limit)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var details []*pb.TransactionDetail
	for rows.Next() {
		var d pb.TransactionDetail
		var createdAt time.Time
		var numFeaturesJSON, catFeaturesJSON []byte
		if err := rows.Scan(
			&d.RecordId,
			&d.UserId,
			&createdAt,
			&d.IsTrainEligible,
			&d.IsPreFraud,
			&d.Amount,
			&d.IsOffHoursTxn,
			&d.MerchantRiskScore,
			&d.Velocity_24H,
			&d.AmountToAvgRatio_30D,
			&d.BalanceVolatilityZScore,
			&numFeaturesJSON,
			&catFeaturesJSON,
		); err != nil {
			return nil, fmt.Errorf("failed to scan transaction detail: %v", err)
		}
		d.CreatedAt = timestamppb.New(createdAt)

		if len(numFeaturesJSON) > 0 {
			json.Unmarshal(numFeaturesJSON, &d.NumericalFeatures)
		}
		if len(catFeaturesJSON) > 0 {
			json.Unmarshal(catFeaturesJSON, &d.CategoricalFeatures)
		}

		details = append(details, &d)
	}
	return details, nil
}

func (s *SQLStore) SearchTransactions(ctx context.Context, req *pb.SearchTransactionsRequest, limit, offset int32) ([]*pb.TransactionDetail, int64, error) {
	queryBuilder := db.NewQueryBuilder(`
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
			fs.balance_volatility_z_score,
			gr.numerical_features,
			gr.categorical_features
		FROM evaluation_metadata em
		LEFT JOIN generated_records gr ON em.record_id = gr.record_id
		LEFT JOIN feature_snapshots fs ON em.record_id = fs.record_id
	`)

	if req.UserId != "" {
		queryBuilder.AddCondition("em.user_id = ?", req.UserId)
	}
	if req.TransactionId != "" {
		queryBuilder.AddCondition("em.record_id = ?", req.TransactionId)
	}
	if req.MinAmount != nil {
		queryBuilder.AddCondition("gr.amount >= ?", *req.MinAmount)
	}
	if req.MaxAmount != nil {
		queryBuilder.AddCondition("gr.amount <= ?", *req.MaxAmount)
	}
	if req.StartDate != "" {
		if t, ok := parseISODate(req.StartDate); ok {
			queryBuilder.AddCondition("em.created_at >= ?", t)
		}
	}
	if req.EndDate != "" {
		if t, ok := parseISODate(req.EndDate); ok {
			queryBuilder.AddCondition("em.created_at <= ?", t)
		}
	}
	if req.IsFraudulent != nil {
		queryBuilder.AddCondition("gr.is_fraudulent = ?", *req.IsFraudulent)
	}
	if req.MinScore != nil {
		queryBuilder.AddCondition("gr.merchant_risk_score >= ?", *req.MinScore)
	}
	if req.MaxScore != nil {
		queryBuilder.AddCondition("gr.merchant_risk_score <= ?", *req.MaxScore)
	}

	// Get total count
	countQuery, countArgs := queryBuilder.BuildCount()
	var total int64
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, countQuery, countArgs...).Scan(&total)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}

	// Get results
	queryBuilder.AddOrderBy("em.created_at DESC")
	queryBuilder.SetLimit(limit)
	queryBuilder.SetOffset(offset)
	selectQuery, selectArgs := queryBuilder.BuildSelect()

	rows, err := s.db.QueryContext(queryCtx, selectQuery, selectArgs...)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}
	defer rows.Close()

	var details []*pb.TransactionDetail
	for rows.Next() {
		var d pb.TransactionDetail
		var createdAt time.Time
		var numFeaturesJSON, catFeaturesJSON []byte
		if err := rows.Scan(
			&d.RecordId,
			&d.UserId,
			&createdAt,
			&d.IsTrainEligible,
			&d.IsPreFraud,
			&d.Amount,
			&d.IsFraudulent,
			&d.FraudType,
			&d.IsOffHoursTxn,
			&d.MerchantRiskScore,
			&d.Velocity_24H,
			&d.AmountToAvgRatio_30D,
			&d.BalanceVolatilityZScore,
			&numFeaturesJSON,
			&catFeaturesJSON,
		); err != nil {
			return nil, 0, fmt.Errorf("failed to scan search result: %v", err)
		}
		d.CreatedAt = timestamppb.New(createdAt)

		if len(numFeaturesJSON) > 0 {
			json.Unmarshal(numFeaturesJSON, &d.NumericalFeatures)
		}
		if len(catFeaturesJSON) > 0 {
			json.Unmarshal(catFeaturesJSON, &d.CategoricalFeatures)
		}

		details = append(details, &d)
	}
	return details, total, nil
}

func (s *SQLStore) GetShadowComparison(ctx context.Context, hours int32) (*pb.ShadowModeMetrics, error) {
	// Logic taken from main.go GetShadowComparison
	// Currently it's a stub in main.go, so I'll copy the stub implementation.
	return &pb.ShadowModeMetrics{
		TotalEvaluations:     100,
		DivergentScoresCount: 5,
		DivergentRate:        0.05,
		ActiveScoreMean:      50.0,
		ShadowScoreMean:      55.0,
		ActiveScoreDistribution: map[string]int32{
			"0-20":   10,
			"20-40":  20,
			"40-60":  40,
			"60-80":  20,
			"80-100": 10,
		},
		ShadowScoreDistribution: map[string]int32{
			"0-20":   5,
			"20-40":  15,
			"40-60":  45,
			"60-80":  25,
			"80-100": 10,
		},
	}, nil
}

func (s *SQLStore) GetRecentAlerts(ctx context.Context, limit int32) ([]*pb.Alert, error) {
	query := `
		SELECT
			fs.record_id,
			fs.user_id,
			em.created_at,
			gr.amount,
			gr.is_fraudulent,
			COALESCE(gr.fraud_type, ''),
			gr.merchant_risk_score,
			fs.velocity_24h,
			fs.amount_to_avg_ratio_30d,
			fs.balance_volatility_z_score
		FROM feature_snapshots fs
		INNER JOIN evaluation_metadata em ON fs.record_id = em.record_id
		INNER JOIN generated_records gr ON fs.record_id = gr.record_id
		WHERE gr.is_fraudulent = TRUE
		ORDER BY em.created_at DESC
		LIMIT $1
	`

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, limit)
	if err != nil {
		return nil, db.MapDBError(err)
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
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan alert: %v", err)
		}
		alert.CreatedAt = timestamppb.New(createdAt)
		// Computed risk score is not in DB, calculated on fly or defaulted
		// In previous implementation it was just 0 or not set if not in query
		alerts = append(alerts, &alert)
	}

	return alerts, nil
}

func (s *SQLStore) GetOverviewMetrics(ctx context.Context) (*pb.GetOverviewMetricsResponse, error) {
	query := `
		SELECT
			(SELECT COUNT(*) FROM generated_records) as total_records,
			(SELECT COUNT(*) FROM generated_records WHERE is_fraudulent = TRUE) as fraud_records,
			(SELECT COUNT(DISTINCT user_id) FROM generated_records) as unique_users,
			(SELECT COALESCE(MIN(transaction_timestamp), NOW()) FROM generated_records) as min_txn_ts,
			(SELECT COALESCE(MAX(transaction_timestamp), NOW()) FROM generated_records) as max_txn_ts,
			(SELECT COALESCE(MIN(created_at), NOW()) FROM generated_records) as min_created,
			(SELECT COALESCE(MAX(created_at), NOW()) FROM generated_records) as max_created,
			(SELECT COALESCE(SUM(amount), 0) FROM generated_records) as total_amount,
			(SELECT COALESCE(SUM(amount), 0) FROM generated_records WHERE is_fraudulent = TRUE) as fraud_amount
	`

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var resp pb.GetOverviewMetricsResponse
	var minTxn, maxTxn, minCreated, maxCreated time.Time

	err := s.db.QueryRowContext(queryCtx, query).Scan(
		&resp.TotalRecords,
		&resp.FraudRecords,
		&resp.UniqueUsers,
		&minTxn,
		&maxTxn,
		&minCreated,
		&maxCreated,
		&resp.TotalAmount,
		&resp.FraudAmount,
	)
	if err != nil {
		return nil, db.MapDBError(err)
	}

	if resp.TotalRecords > 0 {
		resp.FraudRate = float64(resp.FraudRecords) / float64(resp.TotalRecords) * 100.0
	}

	// Helper to convert time to proto timestamp safely
	toProto := func(t time.Time) *timestamppb.Timestamp {
		if t.IsZero() {
			return nil
		}
		return timestamppb.New(t)
	}

	resp.MinTransactionTimestamp = toProto(minTxn)
	resp.MaxTransactionTimestamp = toProto(maxTxn)
	resp.MinCreatedAt = toProto(minCreated)
	resp.MaxCreatedAt = toProto(maxCreated)

	return &resp, nil
}

func (s *SQLStore) GetSchemaSummary(ctx context.Context) (*pb.GetSchemaSummaryResponse, error) {
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

	tableNames := []string{"generated_records", "feature_snapshots"}
	arrStr := "{" + strings.Join(tableNames, ",") + "}"

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, arrStr)
	if err != nil {
		return nil, db.MapDBError(err)
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
		col.ColumnName = strings.ToLower(col.ColumnName)
		columns = append(columns, &col)
	}

	return &pb.GetSchemaSummaryResponse{Columns: columns}, nil
}

const (
	MaxNumericKeysProfiled     = 25
	MaxCategoricalKeysProfiled = 25
	DefaultTopK                = 10
	MaxHistogramBuckets        = 50
)

func (s *SQLStore) GetDatasetProfile(ctx context.Context, datasetID string, limitFeatures, numBuckets int32) (*pb.GetDatasetProfileResponse, error) {
	// 1. Get total records
	var totalRecords int64
	err := s.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM generated_records").Scan(&totalRecords)
	if err != nil {
		return nil, db.MapDBError(err)
	}

	if totalRecords == 0 {
		return &pb.GetDatasetProfileResponse{TotalRecords: 0}, nil
	}

	if numBuckets > MaxHistogramBuckets {
		numBuckets = MaxHistogramBuckets
	}

	resp := &pb.GetDatasetProfileResponse{
		TotalRecords: totalRecords,
	}

	// 2. Profile static numeric features
	staticNumericFeatures := []string{
		"amount", "available_balance", "balance_to_transaction_ratio",
		"avg_available_balance_30d", "balance_volatility_z_score",
		"merchant_risk_score",
	}

	for _, feat := range staticNumericFeatures {
		if int32(len(resp.FeatureProfiles)) >= limitFeatures {
			resp.IsPartial = true
			resp.TruncatedKeys++
			continue
		}
		profile, err := s.profileNumericFeature(ctx, "generated_records", feat, totalRecords, numBuckets)
		if err != nil {
			continue
		}
		resp.FeatureProfiles = append(resp.FeatureProfiles, profile)
	}

	// 3. Profile dynamic numeric features
	dynamicNumericKeys, err := s.discoverJSONBKeys(ctx, "generated_records", "numerical_features", MaxNumericKeysProfiled)
	if err == nil {
		for _, key := range dynamicNumericKeys {
			if int32(len(resp.FeatureProfiles)) >= limitFeatures {
				resp.IsPartial = true
				resp.TruncatedKeys++
				continue
			}
			profile, err := s.profileNumericJSONBKey(ctx, "generated_records", "numerical_features", key, totalRecords, numBuckets)
			if err != nil {
				continue
			}
			resp.FeatureProfiles = append(resp.FeatureProfiles, profile)
		}
	}

	// 4. Profile dynamic categorical features
	dynamicCategoricalKeys, err := s.discoverJSONBKeys(ctx, "generated_records", "categorical_features", MaxCategoricalKeysProfiled)
	if err == nil {
		for _, key := range dynamicCategoricalKeys {
			if int32(len(resp.FeatureProfiles)) >= limitFeatures {
				resp.IsPartial = true
				resp.TruncatedKeys++
				continue
			}
			profile, err := s.profileCategoricalJSONBKey(ctx, "generated_records", "categorical_features", key, totalRecords, DefaultTopK)
			if err != nil {
				continue
			}
			resp.FeatureProfiles = append(resp.FeatureProfiles, profile)
		}
	}

	return resp, nil
}

func (s *SQLStore) discoverJSONBKeys(ctx context.Context, table, column string, limit int) ([]string, error) {
	// Use a subquery to avoid full table expansion before distinct/limit
	query := fmt.Sprintf(`
		SELECT DISTINCT key
		FROM (
			SELECT jsonb_object_keys(%[2]s) as key
			FROM (SELECT %[2]s FROM %[1]s WHERE %[2]s IS NOT NULL AND %[2]s != '{}'::jsonb LIMIT 1000) as sub
		) as keys
		LIMIT %[3]d
	`, table, column, limit)

	rows, err := s.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var keys []string
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err == nil {
			keys = append(keys)
			keys = append(keys, key)
		}
	}
	// Deduplicate just in case, though SQL does it
	return keys, nil
}

func (s *SQLStore) profileNumericJSONBKey(ctx context.Context, table, column, key string, totalRecords int64, numBuckets int32) (*pb.FeatureProfile, error) {
	columnExpr := fmt.Sprintf("(%s->>'%s')::numeric", column, key)
	return s.profileNumericFeatureExpr(ctx, table, columnExpr, key, totalRecords, numBuckets)
}

func (s *SQLStore) profileNumericFeature(ctx context.Context, table, column string, totalRecords int64, numBuckets int32) (*pb.FeatureProfile, error) {
	return s.profileNumericFeatureExpr(ctx, table, column, column, totalRecords, numBuckets)
}

func (s *SQLStore) profileNumericFeatureExpr(ctx context.Context, table, expr, name string, totalRecords int64, numBuckets int32) (*pb.FeatureProfile, error) {
	query := fmt.Sprintf(`
		SELECT
			AVG(%[1]s) as mean,
			STDDEV(%[1]s) as stddev,
			COUNT(*) FILTER (WHERE %[1]s IS NULL) as null_count,
			MIN(%[1]s) as min_val,
			MAX(%[1]s) as max_val
		FROM %[2]s
	`, expr, table)

	var mean, stddev, minVal, maxVal sql.NullFloat64
	var nullCount int64
	err := s.db.QueryRowContext(ctx, query).Scan(&mean, &stddev, &nullCount, &minVal, &maxVal)
	if err != nil {
		return nil, err
	}

	profile := &pb.FeatureProfile{
		Name:     name,
		Type:     "numeric",
		NullRate: float64(nullCount) / float64(totalRecords),
		Mean:     mean.Float64,
		StdDev:   stddev.Float64,
	}

	if minVal.Valid && maxVal.Valid && maxVal.Float64 >= minVal.Float64 {
		// Equi-width histogram
		var bucketSize float64
		if maxVal.Float64 > minVal.Float64 {
			bucketSize = (maxVal.Float64 - minVal.Float64) / float64(numBuckets)
		} else {
			bucketSize = 1.0 // Single value case
		}

		// Adjust max for WIDTH_BUCKET exclusive upper bound
		upperBound := maxVal.Float64 + 0.000001

		histQuery := fmt.Sprintf(`
			SELECT
				WIDTH_BUCKET(%[1]s, %[2]f, %[3]f, %[4]d) as bucket,
				COUNT(*) as count
			FROM %[5]s
			WHERE %[1]s IS NOT NULL
			GROUP BY bucket
			ORDER BY bucket
		`, expr, minVal.Float64, upperBound, numBuckets, table)

		rows, err := s.db.QueryContext(ctx, histQuery)
		if err == nil {
			defer rows.Close()
			buckets := make(map[int]int64)
			for rows.Next() {
				var b int
				var c int64
				if err := rows.Scan(&b, &c); err == nil {
					buckets[b] = c
				}
			}

			for i := 1; i <= int(numBuckets); i++ {
				lower := minVal.Float64 + float64(i-1)*bucketSize
				upper := minVal.Float64 + float64(i)*bucketSize
				profile.Histogram = append(profile.Histogram, &pb.Bucket{
					Lower: lower,
					Upper: upper,
					Count: buckets[i],
				})
			}
		}
	}

	return profile, nil
}

func (s *SQLStore) profileCategoricalJSONBKey(ctx context.Context, table, column, key string, totalRecords int64, topK int) (*pb.FeatureProfile, error) {
	expr := fmt.Sprintf("%s->>'%s'", column, key)

	// Get null rate
	var nullCount int64
	nullQuery := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE %s IS NULL", table, expr)
	err := s.db.QueryRowContext(ctx, nullQuery).Scan(&nullCount)
	if err != nil {
		return nil, err
	}

	profile := &pb.FeatureProfile{
		Name:     key,
		Type:     "categorical",
		NullRate: float64(nullCount) / float64(totalRecords),
	}

	// Get top-K frequencies
	topQuery := fmt.Sprintf(`
		SELECT %[1]s as value, COUNT(*) as count
		FROM %[2]s
		WHERE %[1]s IS NOT NULL
		GROUP BY value
		ORDER BY count DESC, value
		LIMIT %[3]d
	`, expr, table, topK)

	rows, err := s.db.QueryContext(ctx, topQuery)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var topValues []*pb.ValueCount
	var topTotalCount int64
	for rows.Next() {
		var vc pb.ValueCount
		if err := rows.Scan(&vc.Value, &vc.Count); err == nil {
			topValues = append(topValues, &vc)
			topTotalCount += vc.Count
		}
	}
	profile.TopValues = topValues

	// Add "other" if applicable
	if totalRecords-nullCount > topTotalCount {
		profile.TopValues = append(profile.TopValues, &pb.ValueCount{
			Value: "_other",
			Count: totalRecords - nullCount - topTotalCount,
		})
	}

	return profile, nil
}
