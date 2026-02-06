package generator

import (
	"fmt"
	"time"

	"github.com/google/uuid"
	pb "github.com/jonkmatsumo/label-lag/src/services/analytics-crud/proto/crud/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Generator creates synthetic transaction data.
// It mirrors the Python DataGenerator's behavior for parity testing.
type Generator struct {
	rng *RNG
	pii *PIIGenerator
}

// NewGenerator creates a generator with the given seed.
// If seed is nil, uses a random seed.
func NewGenerator(seed *int64) *Generator {
	rng := NewRNG(seed)
	return &Generator{
		rng: rng,
		pii: NewPIIGenerator(rng),
	}
}

// generateRecordID creates a unique record identifier.
func (g *Generator) generateRecordID() string {
	return fmt.Sprintf("rec_%s", uuid.New().String()[:8])
}

// generateUserID creates a unique user identifier.
func (g *Generator) generateUserID() string {
	return fmt.Sprintf("user_%s", uuid.New().String()[:8])
}

// generateTimestamp creates a transaction timestamp.
// If offHours is true, generates timestamp between 11pm-5am.
func (g *Generator) generateTimestamp(offHours bool) time.Time {
	// Base: random day in past 90 days
	daysAgo := g.rng.IntN(90)
	baseDate := time.Now().AddDate(0, 0, -daysAgo)

	var hour int
	if offHours {
		// Off hours: 11pm-5am
		hour = g.rng.IntRange(23, 29) % 24 // 23, 0, 1, 2, 3, 4
		if hour >= 24 {
			hour -= 24
		}
	} else {
		// Normal hours: 6am-10pm
		hour = g.rng.IntRange(6, 22)
	}

	minute := g.rng.IntN(60)
	second := g.rng.IntN(60)

	return time.Date(baseDate.Year(), baseDate.Month(), baseDate.Day(),
		hour, minute, second, 0, time.UTC)
}

// sampleLogNormalAmount generates a log-normal distributed transaction amount.
// Uses mu=5.0, sigma=1.5, matching the Python implementation.
// Clamped to range [0.01, 50000.0].
func (g *Generator) sampleLogNormalAmount() float64 {
	mu := 5.0
	sigma := 1.5
	amount := g.rng.LogNormal(mu, sigma)

	// Clamp to reasonable range
	if amount < 0.01 {
		amount = 0.01
	}
	if amount > 50000.0 {
		amount = 50000.0
	}

	// Round to 2 decimal places
	return float64(int(amount*100+0.5)) / 100
}

// GenerateLegitimate creates legitimate (non-fraudulent) transaction records.
// Characteristics:
// - Log-normal distribution for amounts
// - balance_volatility_z_score between -1.0 and 1.0
// - merchant_risk_score averaging 20-30
func (g *Generator) GenerateLegitimate(count int) []*pb.GeneratedRecord {
	records := make([]*pb.GeneratedRecord, 0, count)

	for i := 0; i < count; i++ {
		userID := g.generateUserID()
		pii := g.pii.Generate()
		timestamp := g.generateTimestamp(false)
		amount := g.sampleLogNormalAmount()

		// Legitimate balance: reasonable range [500, 15000]
		availableBalance := g.rng.Float64Range(500, 15000)
		availableBalance = float64(int(availableBalance*100+0.5)) / 100

		avgBalance30d := g.rng.Float64Range(400, 12000)
		avgBalance30d = float64(int(avgBalance30d*100+0.5)) / 100

		// Normal volatility: z-score between -1.0 and 1.0
		volatilityZ := g.rng.Float64Range(-1.0, 1.0)

		// Normal connection patterns
		connections24h := g.rng.IntRange(0, 3)
		connections7d := g.rng.IntRange(0, 10)
		connectionsAvg30d := g.rng.Float64Range(0.5, 2.0)

		// Amount ratio close to 1.0 (normal spending)
		amountToAvg := g.rng.Float64Range(0.3, 2.5)

		// Low merchant risk score (average 20-30)
		merchantRisk := g.rng.IntRange(5, 45)

		// Calculate balance to transaction ratio
		balanceToTxnRatio := 0.0
		if amount > 0 {
			balanceToTxnRatio = availableBalance / amount
		}

		record := &pb.GeneratedRecord{
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
			IsFraudulent:              false,
			FraudType:                 "",
		}

		records = append(records, record)
	}

	return records
}

// DatasetResult contains the generated records and their evaluation metadata.
type DatasetResult struct {
	Records  []*pb.GeneratedRecord
	Metadata []*pb.EvaluationMetadata
}

// GenerateDatasetWithSequences creates a complete dataset with user sequences.
// numUsers: total number of users to generate
// fraudRate: fraction of users that are fraudulent (0.0 to 1.0)
// Returns records and evaluation metadata for training.
func (g *Generator) GenerateDatasetWithSequences(numUsers int, fraudRate float64) *DatasetResult {
	var allRecords []*pb.GeneratedRecord
	var allMetadata []*pb.EvaluationMetadata

	fraudUserCount := int(float64(numUsers) * fraudRate)
	legitUserCount := numUsers - fraudUserCount

	// Generate legitimate user sequences
	for i := 0; i < legitUserCount; i++ {
		userRecords := g.generateUserSequence(false)
		for seq, r := range userRecords {
			allRecords = append(allRecords, r)
			allMetadata = append(allMetadata, &pb.EvaluationMetadata{
				UserId:          r.UserId,
				RecordId:        r.RecordId,
				SequenceNumber:  int32(seq + 1),
				IsPreFraud:      true, // Always true for legitimate users
				IsTrainEligible: true,
			})
		}
	}

	// Generate fraudulent user sequences
	for i := 0; i < fraudUserCount; i++ {
		userRecords := g.generateUserSequence(true)
		fraudConfirmedAt := userRecords[len(userRecords)-1].TransactionTimestamp

		for seq, r := range userRecords {
			isPreFraud := seq < len(userRecords)-1 // Last record is the fraud
			daysToFraud := int32(len(userRecords) - 1 - seq)

			allRecords = append(allRecords, r)
			allMetadata = append(allMetadata, &pb.EvaluationMetadata{
				UserId:           r.UserId,
				RecordId:         r.RecordId,
				SequenceNumber:   int32(seq + 1),
				FraudConfirmedAt: fraudConfirmedAt,
				IsPreFraud:       isPreFraud,
				DaysToFraud:      daysToFraud,
				IsTrainEligible:  isPreFraud, // Only pre-fraud records are train eligible
			})
		}
	}

	return &DatasetResult{
		Records:  allRecords,
		Metadata: allMetadata,
	}
}

// generateUserSequence creates a sequence of transactions for a single user.
// If isFraudulent, the last transaction is fraudulent.
func (g *Generator) generateUserSequence(isFraudulent bool) []*pb.GeneratedRecord {
	// Each user has 5-15 transactions
	numTransactions := g.rng.IntRange(5, 16)
	records := make([]*pb.GeneratedRecord, 0, numTransactions)

	userID := g.generateUserID()
	pii := g.pii.Generate()

	if isFraudulent {
		// Generate legitimate transactions first, then fraud at end
		for i := 0; i < numTransactions-1; i++ {
			r := g.generateLegitimateForUser(userID, pii)
			records = append(records, r)
		}
		// Final transaction is fraudulent
		fraudType := g.randomFraudType()
		r := g.generateFraudForUser(userID, pii, fraudType)
		records = append(records, r)
	} else {
		// All legitimate transactions
		for i := 0; i < numTransactions; i++ {
			r := g.generateLegitimateForUser(userID, pii)
			records = append(records, r)
		}
	}

	return records
}

// randomFraudType selects a random fraud type.
func (g *Generator) randomFraudType() FraudType {
	types := []FraudType{FraudTypeLiquidityCrunch, FraudTypeLinkBurst, FraudTypeATO}
	return types[g.rng.IntN(len(types))]
}

// generateLegitimateForUser creates a legitimate record for a specific user.
func (g *Generator) generateLegitimateForUser(userID string, pii PII) *pb.GeneratedRecord {
	timestamp := g.generateTimestamp(false)
	amount := g.sampleLogNormalAmount()

	availableBalance := g.rng.Float64Range(500, 15000)
	availableBalance = float64(int(availableBalance*100+0.5)) / 100

	avgBalance30d := g.rng.Float64Range(400, 12000)
	avgBalance30d = float64(int(avgBalance30d*100+0.5)) / 100

	volatilityZ := g.rng.Float64Range(-1.0, 1.0)
	connections24h := g.rng.IntRange(0, 3)
	connections7d := g.rng.IntRange(0, 10)
	connectionsAvg30d := g.rng.Float64Range(0.5, 2.0)
	amountToAvg := g.rng.Float64Range(0.3, 2.5)
	merchantRisk := g.rng.IntRange(5, 45)

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
		IsFraudulent:              false,
		FraudType:                 "",
	}
}

// generateFraudForUser creates a fraudulent record for a specific user.
func (g *Generator) generateFraudForUser(userID string, pii PII, fraudType FraudType) *pb.GeneratedRecord {
	var record *pb.GeneratedRecord

	switch fraudType {
	case FraudTypeLiquidityCrunch:
		record = g.generateLiquidityCrunch()
	case FraudTypeLinkBurst:
		record = g.generateLinkBurst()
	case FraudTypeATO:
		record = g.generateATO()
	default:
		record = g.generateLiquidityCrunch()
	}

	// Override user ID and PII to match the user's sequence
	record.UserId = userID
	record.FullName = pii.FullName
	record.Email = pii.Email
	record.Phone = pii.Phone

	return record
}
