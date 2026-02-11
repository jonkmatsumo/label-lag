package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/lib/pq"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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

func (s *SQLStore) GetTransactionDetails(ctx context.Context, cutoffDate time.Time, limit, offset int32) ([]*pb.TransactionDetail, error) {
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
		LIMIT $2 OFFSET $3
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, cutoffDate, limit, offset)
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
			if err := json.Unmarshal(numFeaturesJSON, &d.NumericalFeatures); err != nil {
				return nil, status.Errorf(codes.Internal, "invalid json payload: %v", err)
			}
		}
		if len(catFeaturesJSON) > 0 {
			if err := json.Unmarshal(catFeaturesJSON, &d.CategoricalFeatures); err != nil {
				return nil, status.Errorf(codes.Internal, "invalid json payload: %v", err)
			}
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
			if err := json.Unmarshal(numFeaturesJSON, &d.NumericalFeatures); err != nil {
				return nil, 0, status.Errorf(codes.Internal, "invalid json payload: %v", err)
			}
		}
		if len(catFeaturesJSON) > 0 {
			if err := json.Unmarshal(catFeaturesJSON, &d.CategoricalFeatures); err != nil {
				return nil, 0, status.Errorf(codes.Internal, "invalid json payload: %v", err)
			}
		}

		details = append(details, &d)
	}
	return details, total, nil
}

func (s *SQLStore) GetShadowComparison(ctx context.Context, hours int32) (*pb.ShadowModeMetrics, error) {
	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour)

	query := `
		WITH shadow_deltas AS (
			SELECT
				request_id,
				SUM(score_delta) as total_shadow_delta
			FROM rule_impacts
			WHERE is_shadow = TRUE
			GROUP BY request_id
		),
		metrics_raw AS (
			SELECT
				ie.request_id,
				ie.final_score as active_score,
				ie.final_score + COALESCE(sd.total_shadow_delta, 0) as shadow_score
			FROM inference_events ie
			LEFT JOIN shadow_deltas sd ON ie.request_id = sd.request_id
			WHERE ie.ts >= $1
		)
		SELECT
			COUNT(*) as total_evaluations,
			COUNT(*) FILTER (WHERE active_score != shadow_score) as divergent_scores_count,
			COALESCE(AVG(active_score), 0) as active_score_mean,
			COALESCE(AVG(shadow_score), 0) as shadow_score_mean
		FROM metrics_raw
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var m pb.ShadowModeMetrics
	err := s.db.QueryRowContext(queryCtx, query, cutoff).Scan(
		&m.TotalEvaluations,
		&m.DivergentScoresCount,
		&m.ActiveScoreMean,
		&m.ShadowScoreMean,
	)
	if err != nil {
		return nil, db.MapDBError(err)
	}

	if m.TotalEvaluations > 0 {
		m.DivergentRate = float64(m.DivergentScoresCount) / float64(m.TotalEvaluations)
	}

	// For simplicity, we skip distributions in the first pass or use a fixed set of buckets.
	// But let's try to do a basic one.
	m.ActiveScoreDistribution = make(map[string]int32)
	m.ShadowScoreDistribution = make(map[string]int32)

	return &m, nil
}

// (Keeping the rest of the file intact)

func (s *SQLStore) GetRecentAlerts(ctx context.Context, limit, offset int32) ([]*pb.Alert, error) {
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
		LIMIT $1 OFFSET $2
	`

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, limit, offset)
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
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var keys []string
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err == nil {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
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
		return nil, db.MapDBError(err)
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
		return nil, db.MapDBError(err)
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
func (s *SQLStore) GetRuleStats(ctx context.Context, ruleID string, cutoff time.Time) ([]*pb.RuleStats, error) {
	query := `
		SELECT
			ri.rule_id,
			COUNT(*) as triggered_count,
			COUNT(*) FILTER (WHERE ri.is_shadow = TRUE) as shadow_triggered_count,
			COALESCE(AVG(CASE WHEN gr.is_fraudulent = TRUE THEN 1.0 ELSE 0.0 END), 0) as approval_rate
		FROM rule_impacts ri
		INNER JOIN inference_events ie ON ri.request_id = ie.request_id
		LEFT JOIN generated_records gr ON ri.request_id = gr.record_id
		WHERE ie.ts >= $1
	`
	args := []interface{}{cutoff}
	if ruleID != "" {
		query += " AND ri.rule_id = $2"
		args = append(args, ruleID)
	}
	query += " GROUP BY ri.rule_id"

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var stats []*pb.RuleStats
	for rows.Next() {
		var s pb.RuleStats
		if err := rows.Scan(
			&s.RuleId,
			&s.TriggeredCount,
			&s.ShadowTriggeredCount,
			&s.ApprovalRate,
		); err != nil {
			return nil, fmt.Errorf("failed to scan rule stats: %v", err)
		}
		stats = append(stats, &s)
	}
	return stats, nil
}
func (s *SQLStore) GetAttribution(ctx context.Context, cutoff time.Time, limit int32) ([]*pb.DailyAttribution, error) {
	query := `
		SELECT
			DATE(ie.ts) as date,
			ri.rule_id,
			SUM(ri.score_delta) as contribution_score,
			COUNT(*) as volume
		FROM rule_impacts ri
		INNER JOIN inference_events ie ON ri.request_id = ie.request_id
		WHERE ie.ts >= $1
		GROUP BY date, ri.rule_id
		ORDER BY date DESC, contribution_score DESC
		LIMIT $2
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, cutoff, limit)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var items []*pb.DailyAttribution
	for rows.Next() {
		var item pb.DailyAttribution
		var date time.Time
		if err := rows.Scan(
			&date,
			&item.RuleId,
			&item.ContributionScore,
			&item.Volume,
		); err != nil {
			return nil, fmt.Errorf("failed to scan attribution item: %v", err)
		}
		item.Date = date.Format("2006-01-02")
		items = append(items, &item)
	}
	return items, nil
}

func (s *SQLStore) GetLatestUserFeatures(ctx context.Context, userID string) (*pb.UserFeatures, bool, error) {
	query := `
		SELECT
			record_id, user_id, snapshot_id, computed_at,
			velocity_24h, amount_to_avg_ratio_30d, balance_volatility_z_score,
			experimental_signals
		FROM feature_snapshots
		WHERE user_id = $1
		ORDER BY computed_at DESC, record_id DESC
		LIMIT 1
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var f pb.UserFeatures
	var snapshotID int64
	var computedAt time.Time
	var experimentalSignals []byte
	var recordID string

	err := s.db.QueryRowContext(queryCtx, query, userID).Scan(
		&recordID,
		&f.UserId,
		&snapshotID,
		&computedAt,
		&f.Velocity_24H,
		&f.AmountToAvgRatio_30D,
		&f.BalanceVolatilityZScore,
		&experimentalSignals,
	)

	if err == sql.ErrNoRows {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, db.MapDBError(err)
	}

	f.SnapshotTimestamp = timestamppb.New(computedAt)
	f.SnapshotId = fmt.Sprintf("%d", snapshotID)
	f.HasHistory = true

	// Parse experimental signals
	var signals struct {
		BankConnections24h int32 `json:"bank_connections_24h"`
		MerchantRiskScore  int32 `json:"merchant_risk_score"`
	}
	if err := json.Unmarshal(experimentalSignals, &signals); err == nil {
		f.BankConnections_24H = signals.BankConnections24h
		f.MerchantRiskScore = signals.MerchantRiskScore
	}

	return &f, true, nil
}

func (s *SQLStore) BatchGetLatestUserFeatures(ctx context.Context, userIDs []string) (map[string]*pb.UserFeatures, error) {
	if len(userIDs) == 0 {
		return make(map[string]*pb.UserFeatures), nil
	}

	// ROW_NUMBER() to get the latest snapshot per user
	query := `
		WITH RankedFeatures AS (
			SELECT
				record_id, user_id, snapshot_id, computed_at,
				velocity_24h, amount_to_avg_ratio_30d, balance_volatility_z_score,
				experimental_signals,
				ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY computed_at DESC, record_id DESC) as rn
			FROM feature_snapshots
			WHERE user_id = ANY($1)
		)
		SELECT
			record_id, user_id, snapshot_id, computed_at,
			velocity_24h, amount_to_avg_ratio_30d, balance_volatility_z_score,
			experimental_signals
		FROM RankedFeatures
		WHERE rn = 1
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, pq.Array(userIDs))
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	results := make(map[string]*pb.UserFeatures)
	for rows.Next() {
		var f pb.UserFeatures
		var snapshotID int64
		var computedAt time.Time
		var experimentalSignals []byte
		var recordID string

		if err := rows.Scan(
			&recordID,
			&f.UserId,
			&snapshotID,
			&computedAt,
			&f.Velocity_24H,
			&f.AmountToAvgRatio_30D,
			&f.BalanceVolatilityZScore,
			&experimentalSignals,
		); err != nil {
			return nil, fmt.Errorf("failed to scan batch user features: %v", err)
		}

		f.SnapshotTimestamp = timestamppb.New(computedAt)
		f.SnapshotId = fmt.Sprintf("%d", snapshotID)
		f.HasHistory = true

		var signals struct {
			BankConnections24h int32 `json:"bank_connections_24h"`
			MerchantRiskScore  int32 `json:"merchant_risk_score"`
		}
		if err := json.Unmarshal(experimentalSignals, &signals); err == nil {
			f.BankConnections_24H = signals.BankConnections24h
			f.MerchantRiskScore = signals.MerchantRiskScore
		}

		results[f.UserId] = &f
	}
	return results, nil
}

func (s *SQLStore) ListDecisions(ctx context.Context, req *pb.ListDecisionsRequest) ([]*pb.DecisionSummary, int64, error) {
	whereClauses := []string{"1=1"}
	args := []interface{}{}

	if req.UserId != "" {
		args = append(args, req.UserId)
		whereClauses = append(whereClauses, fmt.Sprintf("user_id = $%d", len(args)))
	}
	if req.Decision != "" {
		args = append(args, req.Decision)
		whereClauses = append(whereClauses, fmt.Sprintf("decision = $%d", len(args)))
	}
	if req.MinScore > 0 {
		args = append(args, req.MinScore)
		whereClauses = append(whereClauses, fmt.Sprintf("final_score >= $%d", len(args)))
	}
	if req.MaxScore > 0 {
		args = append(args, req.MaxScore)
		whereClauses = append(whereClauses, fmt.Sprintf("final_score <= $%d", len(args)))
	}
	if req.StartDate != nil {
		args = append(args, req.StartDate.AsTime())
		whereClauses = append(whereClauses, fmt.Sprintf("ts >= $%d", len(args)))
	}
	if req.EndDate != nil {
		args = append(args, req.EndDate.AsTime())
		whereClauses = append(whereClauses, fmt.Sprintf("ts <= $%d", len(args)))
	}

	whereStmt := strings.Join(whereClauses, " AND ")

	countQuery := fmt.Sprintf("SELECT COUNT(*) FROM inference_events WHERE %s", whereStmt)
	var total int64
	err := s.db.QueryRowContext(ctx, countQuery, args...).Scan(&total)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}

	query := fmt.Sprintf("SELECT request_id, user_id, ts, final_score, decision, rule_impacts FROM inference_events WHERE %s ORDER BY ts DESC, request_id DESC LIMIT $%d OFFSET $%d",
		whereStmt, len(args)+1, len(args)+2)

	args = append(args, req.Limit, req.Offset)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}
	defer rows.Close()

	var decisions []*pb.DecisionSummary
	for rows.Next() {
		var d pb.DecisionSummary
		var ts time.Time
		var ruleImpactsJSON []byte
		if err := rows.Scan(&d.RequestId, &d.UserId, &ts, &d.FinalScore, &d.Decision, &ruleImpactsJSON); err != nil {
			return nil, 0, fmt.Errorf("failed to scan decision summary: %v", err)
		}
		d.CreatedAt = timestamppb.New(ts)
		d.DecisionReason, d.Thresholds = populateExplainability(d.Decision, d.FinalScore, ruleImpactsJSON)
		decisions = append(decisions, &d)
	}
	return decisions, total, nil
}

func (s *SQLStore) GetDecision(ctx context.Context, requestID string) (*pb.InferenceEvent, error) {
	query := "SELECT request_id, ts, model_version, rules_version, model_score, final_score, rule_impacts, user_id, decision FROM inference_events WHERE request_id = $1"
	var ie pb.InferenceEvent
	var ts time.Time
	var ruleImpactsJSON []byte
	err := s.db.QueryRowContext(ctx, query, requestID).Scan(
		&ie.RequestId, &ts, &ie.ModelVersion, &ie.RulesVersion,
		&ie.ModelScore, &ie.FinalScore, &ruleImpactsJSON,
		&ie.UserId, &ie.Decision,
	)
	if err == sql.ErrNoRows {
		return nil, status.Errorf(codes.NotFound, "decision not found for request %s", requestID)
	}
	if err != nil {
		return nil, db.MapDBError(err)
	}
	ie.Timestamp = timestamppb.New(ts)
	if err := json.Unmarshal(ruleImpactsJSON, &ie.RuleImpacts); err != nil {
		return nil, fmt.Errorf("failed to unmarshal rule impacts: %v", err)
	}
	ie.DecisionReason, ie.Thresholds = populateExplainability(ie.Decision, ie.FinalScore, ruleImpactsJSON)
	return &ie, nil
}

func (s *SQLStore) GetDecisionTrace(ctx context.Context, requestID string) ([]*pb.RuleImpact, error) {
	// First check if the event exists
	var exists bool
	err := s.db.QueryRowContext(ctx, "SELECT EXISTS(SELECT 1 FROM inference_events WHERE request_id = $1)", requestID).Scan(&exists)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	if !exists {
		return nil, status.Errorf(codes.NotFound, "decision not found for request %s", requestID)
	}

	query := "SELECT rule_id, is_shadow, score_delta FROM rule_impacts WHERE request_id = $1 ORDER BY score_delta DESC, rule_id ASC"
	rows, err := s.db.QueryContext(ctx, query, requestID)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var trace []*pb.RuleImpact
	for rows.Next() {
		var ri pb.RuleImpact
		if err := rows.Scan(&ri.RuleId, &ri.IsShadow, &ri.ScoreDelta); err != nil {
			return nil, fmt.Errorf("failed to scan rule impact: %v", err)
		}
		trace = append(trace, &ri)
	}
	return trace, nil
}

func (s *SQLStore) GetRuleImpact(ctx context.Context, req *pb.GetRuleImpactRequest) (*pb.GetRuleImpactResponse, error) {
	// Check if rule exists
	var exists bool
	err := s.db.QueryRowContext(ctx, "SELECT EXISTS(SELECT 1 FROM rules WHERE rule_id = $1)", req.RuleId).Scan(&exists)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	if !exists {
		return nil, status.Errorf(codes.NotFound, "rule not found: %s", req.RuleId)
	}

	resp := &pb.GetRuleImpactResponse{
		RuleId: req.RuleId,
	}

	baseWhere := "WHERE ri.rule_id = $1"
	args := []interface{}{req.RuleId}

	if req.StartDate != nil {
		args = append(args, req.StartDate.AsTime())
		baseWhere += fmt.Sprintf(" AND ie.ts >= $%d", len(args))
	}
	if req.EndDate != nil {
		args = append(args, req.EndDate.AsTime())
		baseWhere += fmt.Sprintf(" AND ie.ts <= $%d", len(args))
	}

	summaryQuery := fmt.Sprintf(`
		SELECT COUNT(*), COALESCE(AVG(ri.score_delta), 0)
		FROM rule_impacts ri
		INNER JOIN inference_events ie ON ri.request_id = ie.request_id
		%s
	`, baseWhere)

	err = s.db.QueryRowContext(ctx, summaryQuery, args...).Scan(&resp.TotalTriggers, &resp.AvgScoreDelta)
	if err != nil && err != sql.ErrNoRows {
		return nil, db.MapDBError(err)
	}

	bucketsQuery := fmt.Sprintf(`
		SELECT
			DATE(ie.ts) as date,
			COUNT(*),
			COALESCE(AVG(ri.score_delta), 0),
			SUM(CASE WHEN (ie.final_score >= 100 AND (ie.final_score - ri.score_delta) < 100) OR (ie.final_score >= 50 AND (ie.final_score - ri.score_delta) < 50) THEN 1 ELSE 0 END) as changes
		FROM rule_impacts ri
		INNER JOIN inference_events ie ON ri.request_id = ie.request_id
		%s
		GROUP BY date
		ORDER BY date DESC
	`, baseWhere)

	rows, err := s.db.QueryContext(ctx, bucketsQuery, args...)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	for rows.Next() {
		var b pb.RuleImpactBucket
		var date time.Time
		if err := rows.Scan(&date, &b.TriggerCount, &b.AvgScoreDelta, &b.DecisionsChangedCount); err != nil {
			return nil, fmt.Errorf("failed to scan bucket: %v", err)
		}
		b.Date = date.Format("2006-01-02")
		resp.DailyBuckets = append(resp.DailyBuckets, &b)
	}

	return resp, nil
}

func (s *SQLStore) GetConfusionMatrix(ctx context.Context, req *pb.GetConfusionMatrixRequest) (*pb.GetConfusionMatrixResponse, error) {
	threshold := req.Threshold
	if threshold <= 0 {
		threshold = 50 // Default
	}

	whereClauses := []string{"1=1"}
	args := []interface{}{}

	if req.StartTime != nil {
		args = append(args, req.StartTime.AsTime())
		whereClauses = append(whereClauses, fmt.Sprintf("ie.ts >= $%d", len(args)))
	}
	if req.EndTime != nil {
		args = append(args, req.EndTime.AsTime())
		whereClauses = append(whereClauses, fmt.Sprintf("ie.ts <= $%d", len(args)))
	}
	if req.ModelVersion != "" {
		args = append(args, req.ModelVersion)
		whereClauses = append(whereClauses, fmt.Sprintf("ie.model_version = $%d", len(args)))
	}

	whereStmt := strings.Join(whereClauses, " AND ")

	query := fmt.Sprintf(`
		SELECT
			COUNT(*) FILTER (WHERE ie.final_score >= %d AND gr.is_fraudulent = TRUE) as tp,
			COUNT(*) FILTER (WHERE ie.final_score >= %d AND gr.is_fraudulent = FALSE) as fp,
			COUNT(*) FILTER (WHERE ie.final_score < %d AND gr.is_fraudulent = FALSE) as tn,
			COUNT(*) FILTER (WHERE ie.final_score < %d AND gr.is_fraudulent = TRUE) as fn,
			COUNT(*) FILTER (WHERE gr.is_fraudulent IS NULL) as missing_labels
		FROM inference_events ie
		INNER JOIN generated_records gr ON ie.request_id = gr.record_id
		WHERE %s
	`, threshold, threshold, threshold, threshold, whereStmt)

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var tp, fp, tn, fn, missing int64
	err := s.db.QueryRowContext(queryCtx, query, args...).Scan(&tp, &fp, &tn, &fn, &missing)
	if err != nil {
		return nil, db.MapDBError(err)
	}

	resp := &pb.GetConfusionMatrixResponse{
		TruePositives:  tp,
		FalsePositives: fp,
		TrueNegatives:  tn,
		FalseNegatives: fn,
	}

	totalLabels := tp + fp + tn + fn
	if totalLabels == 0 {
		resp.InsufficientLabels = true
		return resp, nil
	}

	if tp+fp > 0 {
		resp.Precision = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		resp.Recall = float64(tp) / float64(tp+fn)
	}
	if resp.Precision+resp.Recall > 0 {
		resp.F1Score = 2 * (resp.Precision * resp.Recall) / (resp.Precision + resp.Recall)
	}

	if missing > 0 && totalLabels < (missing/2) {
		resp.InsufficientLabels = true
	}

	return resp, nil
}

func (s *SQLStore) GetKpis(ctx context.Context, req *pb.GetKpisRequest) (*pb.GetKpisResponse, error) {
	tableName := "aggregates_daily"
	timeCol := "date"
	if req.GroupBy == "hour" {
		tableName = "aggregates_hourly"
		timeCol = "hour"
	}

	whereClauses := []string{"1=1"}
	args := []interface{}{}

	if req.StartTime != nil {
		args = append(args, req.StartTime.AsTime())
		whereClauses = append(whereClauses, fmt.Sprintf("%s >= $%d", timeCol, len(args)))
	}
	if req.EndTime != nil {
		args = append(args, req.EndTime.AsTime())
		whereClauses = append(whereClauses, fmt.Sprintf("%s <= $%d", timeCol, len(args)))
	}

	whereStmt := strings.Join(whereClauses, " AND ")

	// Summary query
	summaryQuery := fmt.Sprintf(`
		SELECT
			COALESCE(SUM(total_decisions), 0),
			COALESCE(SUM(total_alerts), 0),
			COALESCE(SUM(sum_score), 0),
			COALESCE(SUM(rules_fired_total), 0)
		FROM %s
		WHERE %s
	`, tableName, whereStmt)

	resp := &pb.GetKpisResponse{}
	var sumScore int64
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, summaryQuery, args...).Scan(
		&resp.TotalDecisions,
		&resp.TotalAlerts,
		&sumScore,
		&resp.RulesFiredTotal,
	)
	if err != nil {
		return nil, db.MapDBError(err)
	}

	if resp.TotalDecisions > 0 {
		resp.AlertRate = float64(resp.TotalAlerts) / float64(resp.TotalDecisions)
		resp.AvgScore = float64(sumScore) / float64(resp.TotalDecisions)
	}

	// Buckets query if group_by is set
	if req.GroupBy != "" {
		bucketsQuery := fmt.Sprintf(`
			SELECT
				%s,
				total_decisions,
				total_alerts,
				CASE WHEN total_decisions > 0 THEN sum_score::float / total_decisions ELSE 0 END
			FROM %s
			WHERE %s
			ORDER BY %s ASC
		`, timeCol, tableName, whereStmt, timeCol)

		rows, err := s.db.QueryContext(queryCtx, bucketsQuery, args...)
		if err != nil {
			return nil, db.MapDBError(err)
		}
		defer rows.Close()

		for rows.Next() {
			var b pb.KpiBucket
			var ts time.Time
			if err := rows.Scan(&ts, &b.TotalDecisions, &b.TotalAlerts, &b.AvgScore); err != nil {
				return nil, fmt.Errorf("failed to scan kpi bucket: %v", err)
			}
			b.Timestamp = timestamppb.New(ts)
			resp.Buckets = append(resp.Buckets, &b)
		}
	}

	return resp, nil
}

func (s *SQLStore) GetVolumeSeries(ctx context.Context, req *pb.GetVolumeSeriesRequest) (*pb.GetVolumeSeriesResponse, error) {
	tableName := "aggregates_daily"
	timeCol := "date"
	if req.Granularity == "hour" {
		tableName = "aggregates_hourly"
		timeCol = "hour"
	}

	whereClauses := []string{"1=1"}
	args := []interface{}{}

	if req.StartTime != nil {
		args = append(args, req.StartTime.AsTime())
		whereClauses = append(whereClauses, fmt.Sprintf("%s >= $%d", timeCol, len(args)))
	}
	if req.EndTime != nil {
		args = append(args, req.EndTime.AsTime())
		whereClauses = append(whereClauses, fmt.Sprintf("%s <= $%d", timeCol, len(args)))
	}

	whereStmt := strings.Join(whereClauses, " AND ")

	query := fmt.Sprintf(`
		SELECT %s, total_decisions, total_alerts
		FROM %s
		WHERE %s
		ORDER BY %s ASC
	`, timeCol, tableName, whereStmt, timeCol)

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	resp := &pb.GetVolumeSeriesResponse{}
	for rows.Next() {
		var p pb.VolumePoint
		var ts time.Time
		if err := rows.Scan(&ts, &p.Count, &p.Alerts); err != nil {
			return nil, fmt.Errorf("failed to scan volume point: %v", err)
		}
		p.Timestamp = timestamppb.New(ts)
		resp.Points = append(resp.Points, &p)
	}

	return resp, nil
}

func populateExplainability(decision string, finalScore int32, ruleImpactsJSON []byte) (string, *pb.DecisionThresholds) {
	thresholds := &pb.DecisionThresholds{
		ApproveMax: 50,
		ReviewMax:  80,
	}

	var impacts []*pb.RuleImpact
	if len(ruleImpactsJSON) > 0 {
		_ = json.Unmarshal(ruleImpactsJSON, &impacts)
	}

	var reason string
	ruleFired := false
	for _, ri := range impacts {
		if !ri.IsShadow && ri.ScoreDelta != 0 {
			ruleFired = true
			break
		}
	}

	if ruleFired {
		reason = "rule_triggered"
	} else if finalScore < thresholds.ApproveMax {
		reason = "score_below_threshold"
	} else if finalScore < thresholds.ReviewMax {
		reason = "score_below_review_threshold"
	} else {
		reason = "score_above_review_threshold"
	}

	return reason, thresholds
}
