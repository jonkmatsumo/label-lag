package httpserver

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	crudv1 "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	grpcclient "github.com/jonkmatsumo/label-lag/go/orchestrator/internal/grpc"
	"google.golang.org/grpc/codes"
)

type AnalyticsClient interface {
	SearchTransactions(ctx context.Context, req *crudv1.SearchTransactionsRequest) (*crudv1.SearchTransactionsResponse, error)
	GetDailyStats(ctx context.Context, req *crudv1.GetDailyStatsRequest) (*crudv1.GetDailyStatsResponse, error)
	GetOverviewMetrics(ctx context.Context, req *crudv1.GetOverviewMetricsRequest) (*crudv1.GetOverviewMetricsResponse, error)
	GetTransactionDetails(ctx context.Context, req *crudv1.GetTransactionDetailsRequest) (*crudv1.GetTransactionDetailsResponse, error)
	GetRecentAlerts(ctx context.Context, req *crudv1.GetRecentAlertsRequest) (*crudv1.GetRecentAlertsResponse, error)
	GetDatasetFingerprint(ctx context.Context, req *crudv1.GetDatasetFingerprintRequest) (*crudv1.GetDatasetFingerprintResponse, error)
	GetFeatureSample(ctx context.Context, req *crudv1.GetFeatureSampleRequest) (*crudv1.GetFeatureSampleResponse, error)
	GetSchemaSummary(ctx context.Context, req *crudv1.GetSchemaSummaryRequest) (*crudv1.GetSchemaSummaryResponse, error)
	ListBacktestResults(ctx context.Context, req *crudv1.ListBacktestResultsRequest) (*crudv1.ListBacktestResultsResponse, error)
	GetFeatures(ctx context.Context, userID string) (map[string]any, error)
	ClearAllData(ctx context.Context, req *crudv1.ClearAllDataRequest) (*crudv1.ClearAllDataResponse, error)
	ListRuleVersions(ctx context.Context, req *crudv1.ListRuleVersionsRequest) (*crudv1.ListRuleVersionsResponse, error)
	GetRuleVersion(ctx context.Context, req *crudv1.GetRuleVersionRequest) (*crudv1.GetRuleVersionResponse, error)
	PublishRuleVersion(ctx context.Context, req *crudv1.PublishRuleVersionRequest) (*crudv1.PublishRuleVersionResponse, error)
	GetRuleReadiness(ctx context.Context, req *crudv1.GetRuleReadinessRequest) (*crudv1.GetRuleReadinessResponse, error)
	DiffRuleVersions(ctx context.Context, req *crudv1.DiffRuleVersionsRequest) (*crudv1.DiffRuleVersionsResponse, error)
	SaveRule(ctx context.Context, req *crudv1.SaveRuleRequest) (*crudv1.SaveRuleResponse, error)
	GetRule(ctx context.Context, req *crudv1.GetRuleRequest) (*crudv1.GetRuleResponse, error)
	ListRules(ctx context.Context, req *crudv1.ListRulesRequest) (*crudv1.ListRulesResponse, error)
	DeleteRule(ctx context.Context, req *crudv1.DeleteRuleRequest) (*crudv1.DeleteRuleResponse, error)
	GetRuleStats(ctx context.Context, req *crudv1.GetRuleStatsRequest) (*crudv1.GetRuleStatsResponse, error)
	GetAttribution(ctx context.Context, req *crudv1.GetAttributionRequest) (*crudv1.GetAttributionResponse, error)
	LogInferenceEvent(ctx context.Context, req *crudv1.LogInferenceEventRequest) (*crudv1.LogInferenceEventResponse, error)
	CompareBacktests(ctx context.Context, req *crudv1.CompareBacktestsRequest) (*crudv1.CompareBacktestsResponse, error)
	GenerateData(ctx context.Context, req *crudv1.GenerateDataRequest) (*crudv1.GenerateDataResponse, error)

	// Decision Explorer (Phase 2)
	ListDecisions(ctx context.Context, req *crudv1.ListDecisionsRequest) (*crudv1.ListDecisionsResponse, error)
	GetDecision(ctx context.Context, req *crudv1.GetDecisionRequest) (*crudv1.GetDecisionResponse, error)
	GetDecisionTrace(ctx context.Context, req *crudv1.GetDecisionTraceRequest) (*crudv1.GetDecisionTraceResponse, error)
	GetRuleImpact(ctx context.Context, req *crudv1.GetRuleImpactRequest) (*crudv1.GetRuleImpactResponse, error)

	// Dashboard Aggregates (Phase 3)
	GetKpis(ctx context.Context, req *crudv1.GetKpisRequest) (*crudv1.GetKpisResponse, error)
	GetVolumeSeries(ctx context.Context, req *crudv1.GetVolumeSeriesRequest) (*crudv1.GetVolumeSeriesResponse, error)
	GetConfusionMatrix(ctx context.Context, req *crudv1.GetConfusionMatrixRequest) (*crudv1.GetConfusionMatrixResponse, error)

	// Shadow Comparison (Phase 9)
	GetShadowComparison(ctx context.Context, req *crudv1.GetShadowComparisonRequest) (*crudv1.GetShadowComparisonResponse, error)
}

type searchTransactionsRequest struct {
	UserID        string   `json:"user_id"`
	TransactionID string   `json:"transaction_id"`
	MinAmount     *float64 `json:"min_amount"`
	MaxAmount     *float64 `json:"max_amount"`
	StartDate     string   `json:"start_date"`
	EndDate       string   `json:"end_date"`
	IsFraudulent  *bool    `json:"is_fraudulent"`
	MinScore      *int32   `json:"min_score"`
	MaxScore      *int32   `json:"max_score"`
	Limit         *int32   `json:"limit"`
	Offset        *int32   `json:"offset"`
}

type transactionDetailResponse struct {
	RecordID                string  `json:"record_id"`
	UserID                  string  `json:"user_id"`
	CreatedAt               string  `json:"created_at"`
	IsTrainEligible         bool    `json:"is_train_eligible"`
	IsPreFraud              bool    `json:"is_pre_fraud"`
	Amount                  float64 `json:"amount"`
	IsFraudulent            bool    `json:"is_fraudulent"`
	FraudType               string  `json:"fraud_type"`
	IsOffHoursTxn           bool    `json:"is_off_hours_txn"`
	MerchantRiskScore       int32   `json:"merchant_risk_score"`
	Velocity24H             int32   `json:"velocity_24h"`
	AmountToAvgRatio30D     float64 `json:"amount_to_avg_ratio_30d"`
	BalanceVolatilityZScore float64 `json:"balance_volatility_z_score"`
}

type searchTransactionsResponse struct {
	Transactions []transactionDetailResponse `json:"transactions"`
	Total        int64                       `json:"total"`
}

func (h *Handler) handleSearchTransactions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if h.analyticsClient == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "analytics backend unavailable")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeJSONError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	defer r.Body.Close()

	var req searchTransactionsRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid json payload")
		return
	}

	grpcReq := &crudv1.SearchTransactionsRequest{
		UserId:        req.UserID,
		TransactionId: req.TransactionID,
		StartDate:     req.StartDate,
		EndDate:       req.EndDate,
		MinAmount:     req.MinAmount,
		MaxAmount:     req.MaxAmount,
		IsFraudulent:  req.IsFraudulent,
		MinScore:      req.MinScore,
		MaxScore:      req.MaxScore,
	}
	if req.Limit != nil {
		grpcReq.Limit = *req.Limit
	}
	if req.Offset != nil {
		grpcReq.Offset = *req.Offset
	}

	resp, err := h.analyticsClient.SearchTransactions(r.Context(), grpcReq)
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

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(searchTransactionsResponse{
		Transactions: transactions,
		Total:        resp.GetTotal(),
	})
}

func writeAnalyticsRPCError(w http.ResponseWriter, err error) {
	var rpcErr *grpcclient.RPCError
	if errors.As(err, &rpcErr) {
		switch rpcErr.Code {
		case codes.InvalidArgument:
			writeJSONError(w, http.StatusBadRequest, rpcErr.Message)
		case codes.NotFound:
			writeJSONError(w, http.StatusNotFound, rpcErr.Message)
		case codes.DeadlineExceeded, codes.Unavailable:
			writeJSONError(w, http.StatusServiceUnavailable, "analytics backend timeout")
		default:
			writeJSONError(w, http.StatusBadGateway, rpcErr.Message)
		}
		return
	}
	writeJSONError(w, http.StatusBadGateway, "analytics backend error")
}
