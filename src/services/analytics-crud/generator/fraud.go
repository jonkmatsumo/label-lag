package generator

import (
	"github.com/google/uuid"
	pb "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// FraudType represents the type of fraud scenario.
type FraudType string

const (
	FraudTypeLiquidityCrunch FraudType = "liquidity_crunch"
	FraudTypeLinkBurst       FraudType = "link_burst"
	FraudTypeATO             FraudType = "ato"
)

// GenerateFraudulent creates fraudulent transaction records of the specified type.
func (g *Generator) GenerateFraudulent(fraudType FraudType, count int) []*pb.GeneratedRecord {
	records := make([]*pb.GeneratedRecord, 0, count)

	for i := 0; i < count; i++ {
		var record *pb.GeneratedRecord
		switch fraudType {
		case FraudTypeLiquidityCrunch:
			record = g.generateLiquidityCrunch()
		case FraudTypeLinkBurst:
			record = g.generateLinkBurst()
		case FraudTypeATO:
			record = g.generateATO()
		default:
			// Unknown fraud type, default to liquidity crunch
			record = g.generateLiquidityCrunch()
		}
		records = append(records, record)
	}

	return records
}

// generateLiquidityCrunch creates a liquidity crunch fraud scenario.
// Characteristics:
// - Low available_balance
// - balance_volatility_z_score < -2.5
func (g *Generator) generateLiquidityCrunch() *pb.GeneratedRecord {
	userID := g.generateUserID()
	pii := g.pii.Generate()
	timestamp := g.generateTimestamp(false)

	// Lower mean for liquidity crunch
	amount := g.rng.LogNormal(4.5, 1.0) // Lower mean ~90
	amount = float64(int(amount*100+0.5)) / 100
	if amount < 0.01 {
		amount = 0.01
	}
	if amount > 50000.0 {
		amount = 50000.0
	}

	// Low balance indicating liquidity issues [5, 100]
	availableBalance := g.rng.Float64Range(5, 100)
	availableBalance = float64(int(availableBalance*100+0.5)) / 100

	// Very low 30d average (depleting account) [50, 300]
	avgBalance30d := g.rng.Float64Range(50, 300)
	avgBalance30d = float64(int(avgBalance30d*100+0.5)) / 100

	// High negative volatility z-score (< -2.5)
	volatilityZ := g.rng.Float64Range(-4.0, -2.6)

	// Normal-ish other metrics to isolate the signal
	connections24h := g.rng.IntRange(0, 3)
	connections7d := g.rng.IntRange(0, 8)
	amountToAvg := g.rng.Float64Range(0.8, 2.0)
	merchantRisk := g.rng.IntRange(15, 50)

	// Calculate balance to transaction ratio
	balanceToTxnRatio := 0.0
	if amount > 0 {
		balanceToTxnRatio = availableBalance / amount
	}

	return &pb.GeneratedRecord{
		RecordId:                  uuid.New().String()[:8],
		UserId:                    userID,
		FullName:                  pii.FullName,
		Email:                     pii.Email,
		Phone:                     pii.Phone,
		TransactionTimestamp:      timestamppb.New(timestamp),
		IsOffHoursTxn:             false,
		AvailableBalance:          availableBalance,
		BalanceToTransactionRatio: balanceToTxnRatio,
		AvgAvailableBalance_30D:   avgBalance30d,
		BalanceVolatilityZScore:   volatilityZ,
		BankConnectionsCount_24H:  int32(connections24h),
		BankConnectionsCount_7D:   int32(connections7d),
		BankConnectionsAvg_30D:    1.0,
		Amount:                    amount,
		AmountToAvgRatio:          amountToAvg,
		MerchantRiskScore:         int32(merchantRisk),
		IsReturned:                true, // Likely to be returned due to insufficient funds
		IsFraudulent:              true,
		FraudType:                 string(FraudTypeLiquidityCrunch),
	}
}

// generateLinkBurst creates a link burst fraud scenario.
// Characteristics:
// - bank_connections_count_24h between 5 and 15 (anomaly threshold > 4)
func (g *Generator) generateLinkBurst() *pb.GeneratedRecord {
	userID := g.generateUserID()
	pii := g.pii.Generate()
	timestamp := g.generateTimestamp(false)
	amount := g.sampleLogNormalAmount()

	// Normal account state
	availableBalance := g.rng.Float64Range(500, 5000)
	availableBalance = float64(int(availableBalance*100+0.5)) / 100

	avgBalance30d := g.rng.Float64Range(400, 4000)
	avgBalance30d = float64(int(avgBalance30d*100+0.5)) / 100

	volatilityZ := g.rng.Float64Range(-1.0, 0.5)

	// Anomalous connection pattern: 5-15 connections in 24h
	connections24h := g.rng.IntRange(5, 16)
	// Also elevated 7d count
	connections7d := g.rng.IntRange(15, 50)
	connectionsAvg30d := g.rng.Float64Range(1.0, 3.0)

	amountToAvg := g.rng.Float64Range(0.5, 2.0)
	merchantRisk := g.rng.IntRange(20, 60)

	// Calculate balance to transaction ratio
	balanceToTxnRatio := 0.0
	if amount > 0 {
		balanceToTxnRatio = availableBalance / amount
	}

	return &pb.GeneratedRecord{
		RecordId:                  uuid.New().String()[:8],
		UserId:                    userID,
		FullName:                  pii.FullName,
		Email:                     pii.Email,
		Phone:                     pii.Phone,
		TransactionTimestamp:      timestamppb.New(timestamp),
		IsOffHoursTxn:             false,
		AvailableBalance:          availableBalance,
		BalanceToTransactionRatio: balanceToTxnRatio,
		AvgAvailableBalance_30D:   avgBalance30d,
		BalanceVolatilityZScore:   volatilityZ,
		BankConnectionsCount_24H:  int32(connections24h),
		BankConnectionsCount_7D:   int32(connections7d),
		BankConnectionsAvg_30D:    connectionsAvg30d,
		Amount:                    amount,
		AmountToAvgRatio:          amountToAvg,
		MerchantRiskScore:         int32(merchantRisk),
		IsReturned:                false,
		IsFraudulent:              true,
		FraudType:                 string(FraudTypeLinkBurst),
	}
}

// generateATO creates an account takeover (ATO) fraud scenario.
// Characteristics:
// - Recent identity changes (email or phone changed)
// - Off-hours transactions
// - High merchant risk score
func (g *Generator) generateATO() *pb.GeneratedRecord {
	userID := g.generateUserID()
	pii := g.pii.Generate()
	timestamp := g.generateTimestamp(true) // Off-hours
	amount := g.sampleLogNormalAmount()

	// Normal account state
	availableBalance := g.rng.Float64Range(1000, 10000)
	availableBalance = float64(int(availableBalance*100+0.5)) / 100

	avgBalance30d := g.rng.Float64Range(800, 8000)
	avgBalance30d = float64(int(avgBalance30d*100+0.5)) / 100

	volatilityZ := g.rng.Float64Range(-0.5, 1.0)

	// Normal connection patterns
	connections24h := g.rng.IntRange(0, 3)
	connections7d := g.rng.IntRange(0, 10)
	connectionsAvg30d := g.rng.Float64Range(0.5, 2.0)

	amountToAvg := g.rng.Float64Range(1.5, 4.0) // Higher than normal
	merchantRisk := g.rng.IntRange(50, 90)      // High risk merchant

	// Calculate balance to transaction ratio
	balanceToTxnRatio := 0.0
	if amount > 0 {
		balanceToTxnRatio = availableBalance / amount
	}

	// Generate identity change timestamps (recent)
	emailChangedAt := timestamp.AddDate(0, 0, -g.rng.IntRange(1, 7))
	phoneChangedAt := timestamp.AddDate(0, 0, -g.rng.IntRange(1, 14))

	return &pb.GeneratedRecord{
		RecordId:                  uuid.New().String()[:8],
		UserId:                    userID,
		FullName:                  pii.FullName,
		Email:                     pii.Email,
		Phone:                     pii.Phone,
		TransactionTimestamp:      timestamppb.New(timestamp),
		IsOffHoursTxn:             true, // ATO often happens off-hours
		AvailableBalance:          availableBalance,
		BalanceToTransactionRatio: balanceToTxnRatio,
		AvgAvailableBalance_30D:   avgBalance30d,
		BalanceVolatilityZScore:   volatilityZ,
		BankConnectionsCount_24H:  int32(connections24h),
		BankConnectionsCount_7D:   int32(connections7d),
		BankConnectionsAvg_30D:    connectionsAvg30d,
		Amount:                    amount,
		AmountToAvgRatio:          amountToAvg,
		MerchantRiskScore:         int32(merchantRisk),
		IsReturned:                false,
		EmailChangedAt:            timestamppb.New(emailChangedAt),
		PhoneChangedAt:            timestamppb.New(phoneChangedAt),
		IsFraudulent:              true,
		FraudType:                 string(FraudTypeATO),
	}
}
