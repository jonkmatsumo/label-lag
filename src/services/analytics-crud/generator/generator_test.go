package generator

import (
	"testing"
)

func TestNewGenerator(t *testing.T) {
	seed := int64(42)
	gen := NewGenerator(&seed)

	if gen == nil {
		t.Fatal("NewGenerator returned nil")
	}
	if gen.rng == nil {
		t.Error("Generator rng is nil")
	}
	if gen.pii == nil {
		t.Error("Generator pii is nil")
	}
}

func TestGenerateLegitimate(t *testing.T) {
	seed := int64(42)
	gen := NewGenerator(&seed)

	records := gen.GenerateLegitimate(10)

	if len(records) != 10 {
		t.Errorf("Expected 10 records, got %d", len(records))
	}

	for i, r := range records {
		// Verify basic fields are populated
		if r.RecordId == "" {
			t.Errorf("Record %d: RecordId is empty", i)
		}
		if r.UserId == "" {
			t.Errorf("Record %d: UserId is empty", i)
		}
		if r.FullName == "" {
			t.Errorf("Record %d: FullName is empty", i)
		}
		if r.Email == "" {
			t.Errorf("Record %d: Email is empty", i)
		}

		// Verify legitimate characteristics
		if r.IsFraudulent {
			t.Errorf("Record %d: IsFraudulent should be false", i)
		}
		if r.FraudType != "" {
			t.Errorf("Record %d: FraudType should be empty, got %q", i, r.FraudType)
		}
		if r.IsOffHoursTxn {
			t.Errorf("Record %d: IsOffHoursTxn should be false", i)
		}

		// Verify z-score range for legitimate transactions
		if r.BalanceVolatilityZScore < -1.0 || r.BalanceVolatilityZScore > 1.0 {
			t.Errorf("Record %d: BalanceVolatilityZScore %f should be in [-1.0, 1.0]",
				i, r.BalanceVolatilityZScore)
		}

		// Verify merchant risk score range
		if r.MerchantRiskScore < 5 || r.MerchantRiskScore >= 45 {
			t.Errorf("Record %d: MerchantRiskScore %d should be in [5, 45)",
				i, r.MerchantRiskScore)
		}

		// Verify amount is clamped
		if r.Amount < 0.01 || r.Amount > 50000.0 {
			t.Errorf("Record %d: Amount %f should be in [0.01, 50000.0]",
				i, r.Amount)
		}
	}
}

func TestGenerateLegitimateIsDeterministic(t *testing.T) {
	seed := int64(42)

	gen1 := NewGenerator(&seed)
	gen2 := NewGenerator(&seed)

	records1 := gen1.GenerateLegitimate(5)
	records2 := gen2.GenerateLegitimate(5)

	for i := range records1 {
		// Note: UserID uses uuid.New() which is NOT deterministic
		// So we only check the deterministic fields
		if records1[i].Amount != records2[i].Amount {
			t.Errorf("Record %d: Amount mismatch %f vs %f",
				i, records1[i].Amount, records2[i].Amount)
		}
		if records1[i].BalanceVolatilityZScore != records2[i].BalanceVolatilityZScore {
			t.Errorf("Record %d: BalanceVolatilityZScore mismatch %f vs %f",
				i, records1[i].BalanceVolatilityZScore, records2[i].BalanceVolatilityZScore)
		}
		if records1[i].AvailableBalance != records2[i].AvailableBalance {
			t.Errorf("Record %d: AvailableBalance mismatch %f vs %f",
				i, records1[i].AvailableBalance, records2[i].AvailableBalance)
		}
	}
}

func TestGenerateLegitimateAmountDistribution(t *testing.T) {
	seed := int64(42)
	gen := NewGenerator(&seed)

	records := gen.GenerateLegitimate(1000)

	// Calculate statistics
	sum := 0.0
	for _, r := range records {
		sum += r.Amount
	}
	mean := sum / float64(len(records))

	// Log-normal with mu=5, sigma=1.5 should have mean around exp(5 + 1.5^2/2) ≈ 509
	// But clamped values will shift this
	// Just check it's in a reasonable range
	if mean < 50 || mean > 2000 {
		t.Errorf("Mean amount %f is outside expected range [50, 2000]", mean)
	}
}

func TestGenerateDatasetWithSequences(t *testing.T) {
	seed := int64(42)
	gen := NewGenerator(&seed)

	result := gen.GenerateDatasetWithSequences(10, 0.10) // 10 users, 10% fraud

	// Should have records
	if len(result.Records) == 0 {
		t.Fatal("Expected records, got none")
	}

	// Should have matching metadata
	if len(result.Metadata) != len(result.Records) {
		t.Errorf("Records (%d) and Metadata (%d) count mismatch",
			len(result.Records), len(result.Metadata))
	}

	// Count fraud records
	fraudCount := 0
	for _, r := range result.Records {
		if r.IsFraudulent {
			fraudCount++
		}
	}

	// Should have approximately 1 fraudulent user (10% of 10)
	// Each fraud user has 1 fraud record at end of sequence
	if fraudCount == 0 {
		t.Error("Expected at least some fraud records")
	}

	// Verify metadata has correct user IDs
	for i, m := range result.Metadata {
		if m.UserId != result.Records[i].UserId {
			t.Errorf("Metadata %d: UserId mismatch", i)
		}
		if m.RecordId != result.Records[i].RecordId {
			t.Errorf("Metadata %d: RecordId mismatch", i)
		}
		if m.SequenceNumber < 1 {
			t.Errorf("Metadata %d: SequenceNumber should be >= 1", i)
		}
	}
}
