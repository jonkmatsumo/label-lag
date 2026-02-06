package generator

import (
	"testing"
)

func TestGenerateFraudulent(t *testing.T) {
	seed := int64(42)
	gen := NewGenerator(&seed)

	fraudTypes := []FraudType{
		FraudTypeLiquidityCrunch,
		FraudTypeLinkBurst,
		FraudTypeATO,
	}

	for _, ft := range fraudTypes {
		t.Run(string(ft), func(t *testing.T) {
			records := gen.GenerateFraudulent(ft, 5)

			if len(records) != 5 {
				t.Errorf("Expected 5 records, got %d", len(records))
			}

			for i, r := range records {
				if !r.IsFraudulent {
					t.Errorf("Record %d: IsFraudulent should be true", i)
				}
				if r.FraudType != string(ft) {
					t.Errorf("Record %d: FraudType = %q, want %q", i, r.FraudType, ft)
				}
			}
		})
	}
}

func TestLiquidityCrunchCharacteristics(t *testing.T) {
	seed := int64(42)
	gen := NewGenerator(&seed)

	records := gen.GenerateFraudulent(FraudTypeLiquidityCrunch, 20)

	for i, r := range records {
		// Low available balance [5, 100]
		if r.AvailableBalance < 5 || r.AvailableBalance > 100 {
			t.Errorf("Record %d: AvailableBalance %f should be in [5, 100]",
				i, r.AvailableBalance)
		}

		// High negative volatility z-score (< -2.5)
		if r.BalanceVolatilityZScore > -2.5 {
			t.Errorf("Record %d: BalanceVolatilityZScore %f should be < -2.5",
				i, r.BalanceVolatilityZScore)
		}

		// Is returned should be true
		if !r.IsReturned {
			t.Errorf("Record %d: IsReturned should be true for liquidity crunch", i)
		}
	}
}

func TestLinkBurstCharacteristics(t *testing.T) {
	seed := int64(42)
	gen := NewGenerator(&seed)

	records := gen.GenerateFraudulent(FraudTypeLinkBurst, 20)

	for i, r := range records {
		// Anomalous connection pattern: 5-15 connections in 24h
		if r.BankConnectionsCount_24H < 5 || r.BankConnectionsCount_24H >= 16 {
			t.Errorf("Record %d: BankConnectionsCount24h %d should be in [5, 16)",
				i, r.BankConnectionsCount_24H)
		}

		// Elevated 7d count: 15-50
		if r.BankConnectionsCount_7D < 15 || r.BankConnectionsCount_7D >= 50 {
			t.Errorf("Record %d: BankConnectionsCount7d %d should be in [15, 50)",
				i, r.BankConnectionsCount_7D)
		}
	}
}

func TestATOCharacteristics(t *testing.T) {
	seed := int64(42)
	gen := NewGenerator(&seed)

	records := gen.GenerateFraudulent(FraudTypeATO, 20)

	for i, r := range records {
		// Off-hours transaction
		if !r.IsOffHoursTxn {
			t.Errorf("Record %d: IsOffHoursTxn should be true for ATO", i)
		}

		// High merchant risk score [50, 90)
		if r.MerchantRiskScore < 50 || r.MerchantRiskScore >= 90 {
			t.Errorf("Record %d: MerchantRiskScore %d should be in [50, 90)",
				i, r.MerchantRiskScore)
		}

		// Identity changes should be set
		if r.EmailChangedAt == nil && r.PhoneChangedAt == nil {
			t.Errorf("Record %d: ATO should have EmailChangedAt or PhoneChangedAt", i)
		}
	}
}
