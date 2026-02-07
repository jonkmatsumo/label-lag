package generator

import (
	"math"
	"testing"
)

func TestNewRNGWithSeed(t *testing.T) {
	seed := int64(42)
	rng1 := NewRNG(&seed)
	rng2 := NewRNG(&seed)

	// Same seed should produce same sequence
	for i := 0; i < 10; i++ {
		v1 := rng1.Float64()
		v2 := rng2.Float64()
		if v1 != v2 {
			t.Errorf("Seeded RNG not deterministic: iteration %d, got %f vs %f", i, v1, v2)
		}
	}
}

func TestNewRNGWithoutSeed(t *testing.T) {
	rng1 := NewRNG(nil)
	rng2 := NewRNG(nil)

	// Different seeds should produce different sequences (with very high probability)
	same := true
	for i := 0; i < 10; i++ {
		v1 := rng1.Float64()
		v2 := rng2.Float64()
		if v1 != v2 {
			same = false
			break
		}
	}
	if same {
		t.Error("Unseeded RNGs produced identical sequences (extremely unlikely)")
	}
}

func TestFloat64Range(t *testing.T) {
	seed := int64(42)
	rng := NewRNG(&seed)

	for i := 0; i < 100; i++ {
		v := rng.Float64Range(10.0, 20.0)
		if v < 10.0 || v >= 20.0 {
			t.Errorf("Float64Range(10, 20) = %f, want [10, 20)", v)
		}
	}
}

func TestIntN(t *testing.T) {
	seed := int64(42)
	rng := NewRNG(&seed)

	for i := 0; i < 100; i++ {
		v := rng.IntN(10)
		if v < 0 || v >= 10 {
			t.Errorf("IntN(10) = %d, want [0, 10)", v)
		}
	}
}

func TestIntRange(t *testing.T) {
	seed := int64(42)
	rng := NewRNG(&seed)

	for i := 0; i < 100; i++ {
		v := rng.IntRange(5, 15)
		if v < 5 || v >= 15 {
			t.Errorf("IntRange(5, 15) = %d, want [5, 15)", v)
		}
	}

	// Edge case: min >= max
	v := rng.IntRange(10, 10)
	if v != 10 {
		t.Errorf("IntRange(10, 10) = %d, want 10", v)
	}
}

func TestLogNormal(t *testing.T) {
	seed := int64(42)
	rng := NewRNG(&seed)

	// Log-normal values should always be positive
	for i := 0; i < 100; i++ {
		v := rng.LogNormal(0, 1)
		if v <= 0 {
			t.Errorf("LogNormal(0, 1) = %f, want > 0", v)
		}
	}

	// Test that mean is approximately exp(mu + sigma^2/2)
	mu, sigma := 5.0, 1.0
	sum := 0.0
	n := 10000
	for i := 0; i < n; i++ {
		sum += rng.LogNormal(mu, sigma)
	}
	mean := sum / float64(n)
	expectedMean := math.Exp(mu + sigma*sigma/2)
	tolerance := 0.2 * expectedMean // 20% tolerance
	if math.Abs(mean-expectedMean) > tolerance {
		t.Errorf("LogNormal mean = %f, expected ~%f (within 20%%)", mean, expectedMean)
	}
}

func TestBool(t *testing.T) {
	seed := int64(42)
	rng := NewRNG(&seed)

	// Test p=0 always returns false
	for i := 0; i < 10; i++ {
		if rng.Bool(0.0) {
			t.Error("Bool(0.0) returned true")
		}
	}

	// Test p=1 always returns true
	for i := 0; i < 10; i++ {
		if !rng.Bool(1.0) {
			t.Error("Bool(1.0) returned false")
		}
	}

	// Test p=0.5 produces roughly 50% true
	rng2 := NewRNG(&seed)
	trueCount := 0
	n := 1000
	for i := 0; i < n; i++ {
		if rng2.Bool(0.5) {
			trueCount++
		}
	}
	ratio := float64(trueCount) / float64(n)
	if ratio < 0.4 || ratio > 0.6 {
		t.Errorf("Bool(0.5) ratio = %f, expected ~0.5", ratio)
	}
}
