package httpserver

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// TODO(phase4): Map core analytics read routes.
// - GET /analytics/overview
// - GET /analytics/daily-stats
// - GET /analytics/transactions
// - GET /analytics/recent-alerts
// - GET /analytics/fingerprint
// - GET /analytics/feature-sample
// - GET /analytics/schema
// - GET /analytics/attribution (if supported upstream)

func (h *Handler) handleAnalyticsOverview(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	resp, err := h.analyticsClient.GetOverviewMetrics(r.Context(), &crudv1.GetOverviewMetricsRequest{})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	response := overviewMetricsResponse{
		TotalRecords:            resp.GetTotalRecords(),
		FraudRecords:            resp.GetFraudRecords(),
		FraudRate:               resp.GetFraudRate(),
		UniqueUsers:             resp.GetUniqueUsers(),
		MinTransactionTimestamp: formatTimestamp(resp.GetMinTransactionTimestamp()),
		MaxTransactionTimestamp: formatTimestamp(resp.GetMaxTransactionTimestamp()),
		MinCreatedAt:            formatTimestamp(resp.GetMinCreatedAt()),
		MaxCreatedAt:            formatTimestamp(resp.GetMaxCreatedAt()),
		TotalAmount:             resp.GetTotalAmount(),
		FraudAmount:             resp.GetFraudAmount(),
	}

	writeAnalyticsJSON(w, response)
}

func (h *Handler) handleAnalyticsDailyStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	days, err := parseIntQuery(r, "days", 30, 1, 90)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	resp, err := h.analyticsClient.GetDailyStats(r.Context(), &crudv1.GetDailyStatsRequest{Days: days})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	stats := make([]dailyStatResponse, 0, len(resp.GetStats()))
	for _, stat := range resp.GetStats() {
		stats = append(stats, dailyStatResponse{
			Date:              stat.GetDate(),
			TotalTransactions: stat.GetTotalTransactions(),
			FraudCount:        stat.GetFraudCount(),
			FraudRate:         stat.GetFraudRate(),
			TotalAmount:       stat.GetTotalAmount(),
			AvgZScore:         stat.GetAvgZScore(),
		})
	}

	writeAnalyticsJSON(w, dailyStatsResponse{Stats: stats})
}

func (h *Handler) handleAnalyticsTransactions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	days, err := parseIntQuery(r, "days", 7, 1, 30)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	limit, err := parseIntQuery(r, "limit", 1000, 1, 5000)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	resp, err := h.analyticsClient.GetTransactionDetails(r.Context(), &crudv1.GetTransactionDetailsRequest{
		Days:  days,
		Limit: limit,
	})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	transactions := make([]transactionDetailResponse, 0, len(resp.GetTransactions()))
	for _, tx := range resp.GetTransactions() {
		createdAt := ""
		if tx.GetCreatedAt() != nil {
			createdAt = tx.GetCreatedAt().AsTime().UTC().Format(time.RFC3339)
		}
		transactions = append(transactions, transactionDetailResponse{
			RecordID:                tx.GetRecordId(),
			UserID:                  tx.GetUserId(),
			CreatedAt:               createdAt,
			IsTrainEligible:         tx.GetIsTrainEligible(),
			IsPreFraud:              tx.GetIsPreFraud(),
			Amount:                  tx.GetAmount(),
			IsFraudulent:            tx.GetIsFraudulent(),
			FraudType:               tx.GetFraudType(),
			IsOffHoursTxn:           tx.GetIsOffHoursTxn(),
			MerchantRiskScore:       tx.GetMerchantRiskScore(),
			Velocity24H:             tx.GetVelocity_24H(),
			AmountToAvgRatio30D:     tx.GetAmountToAvgRatio_30D(),
			BalanceVolatilityZScore: tx.GetBalanceVolatilityZScore(),
		})
	}

	writeAnalyticsJSON(w, transactionDetailsResponse{Transactions: transactions})
}

func (h *Handler) handleAnalyticsRecentAlerts(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	limit, err := parseIntQuery(r, "limit", 50, 1, 200)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	resp, err := h.analyticsClient.GetRecentAlerts(r.Context(), &crudv1.GetRecentAlertsRequest{Limit: limit})
	if err != nil {
		writeAnalyticsRPCError(w, err)
		return
	}

	alerts := make([]alertResponse, 0, len(resp.GetAlerts()))
	for _, alert := range resp.GetAlerts() {
		createdAt := ""
		if alert.GetCreatedAt() != nil {
			createdAt = alert.GetCreatedAt().AsTime().UTC().Format(time.RFC3339)
		}
		alerts = append(alerts, alertResponse{
			RecordID:                alert.GetRecordId(),
			UserID:                  alert.GetUserId(),
			CreatedAt:               createdAt,
			Amount:                  alert.GetAmount(),
			IsFraudulent:            alert.GetIsFraudulent(),
			FraudType:               alert.GetFraudType(),
			MerchantRiskScore:       alert.GetMerchantRiskScore(),
			Velocity24H:             alert.GetVelocity_24H(),
			AmountToAvgRatio30D:     alert.GetAmountToAvgRatio_30D(),
			BalanceVolatilityZScore: alert.GetBalanceVolatilityZScore(),
			ComputedRiskScore:       alert.GetComputedRiskScore(),
		})
	}

	writeAnalyticsJSON(w, recentAlertsResponse{Alerts: alerts})
}

func (h *Handler) handleAnalyticsFingerprint(w http.ResponseWriter, r *http.Request) {}

func (h *Handler) handleAnalyticsFeatureSample(w http.ResponseWriter, r *http.Request) {}

func (h *Handler) handleAnalyticsSchema(w http.ResponseWriter, r *http.Request) {}

func (h *Handler) handleAnalyticsAttribution(w http.ResponseWriter, r *http.Request) {}

func parseIntQuery(r *http.Request, name string, defaultValue, minValue, maxValue int32) (int32, error) {
	raw := r.URL.Query().Get(name)
	if raw == "" {
		return defaultValue, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return 0, errors.New("invalid query parameter")
	}
	if value < int(minValue) || value > int(maxValue) {
		return 0, errors.New("query parameter out of range")
	}
	return int32(value), nil
}

func writeAnalyticsJSON(w http.ResponseWriter, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(payload)
}

func formatTimestamp(ts *timestamppb.Timestamp) string {
	if ts == nil {
		return ""
	}
	return ts.AsTime().UTC().Format(time.RFC3339)
}

type overviewMetricsResponse struct {
	TotalRecords            int64   `json:"total_records"`
	FraudRecords            int64   `json:"fraud_records"`
	FraudRate               float64 `json:"fraud_rate"`
	UniqueUsers             int64   `json:"unique_users"`
	MinTransactionTimestamp string  `json:"min_transaction_timestamp"`
	MaxTransactionTimestamp string  `json:"max_transaction_timestamp"`
	MinCreatedAt            string  `json:"min_created_at"`
	MaxCreatedAt            string  `json:"max_created_at"`
	TotalAmount             float64 `json:"total_amount"`
	FraudAmount             float64 `json:"fraud_amount"`
}

type dailyStatResponse struct {
	Date              string  `json:"date"`
	TotalTransactions int64   `json:"total_transactions"`
	FraudCount        int64   `json:"fraud_count"`
	FraudRate         float64 `json:"fraud_rate"`
	TotalAmount       float64 `json:"total_amount"`
	AvgZScore         float64 `json:"avg_z_score"`
}

type dailyStatsResponse struct {
	Stats []dailyStatResponse `json:"stats"`
}

type transactionDetailsResponse struct {
	Transactions []transactionDetailResponse `json:"transactions"`
}

type alertResponse struct {
	RecordID                string  `json:"record_id"`
	UserID                  string  `json:"user_id"`
	CreatedAt               string  `json:"created_at"`
	Amount                  float64 `json:"amount"`
	IsFraudulent            bool    `json:"is_fraudulent"`
	FraudType               string  `json:"fraud_type"`
	MerchantRiskScore       int32   `json:"merchant_risk_score"`
	Velocity24H             int32   `json:"velocity_24h"`
	AmountToAvgRatio30D     float64 `json:"amount_to_avg_ratio_30d"`
	BalanceVolatilityZScore float64 `json:"balance_volatility_z_score"`
	ComputedRiskScore       int32   `json:"computed_risk_score"`
}

type recentAlertsResponse struct {
	Alerts []alertResponse `json:"alerts"`
}
