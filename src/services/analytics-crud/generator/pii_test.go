package generator

import (
	"regexp"
	"strings"
	"testing"
)

func TestPIIGeneratorDeterminism(t *testing.T) {
	seed1 := int64(42)
	seed2 := int64(42)

	rng1 := NewRNG(&seed1)
	rng2 := NewRNG(&seed2)

	pii1 := NewPIIGenerator(rng1)
	pii2 := NewPIIGenerator(rng2)

	// Same seed should produce same PII
	for i := 0; i < 5; i++ {
		p1 := pii1.Generate()
		p2 := pii2.Generate()

		if p1.FullName != p2.FullName {
			t.Errorf("FullName mismatch at iteration %d: %q vs %q", i, p1.FullName, p2.FullName)
		}
		if p1.Email != p2.Email {
			t.Errorf("Email mismatch at iteration %d: %q vs %q", i, p1.Email, p2.Email)
		}
		if p1.Phone != p2.Phone {
			t.Errorf("Phone mismatch at iteration %d: %q vs %q", i, p1.Phone, p2.Phone)
		}
	}
}

func TestPIIGenerateReturnsValidData(t *testing.T) {
	seed := int64(42)
	rng := NewRNG(&seed)
	piiGen := NewPIIGenerator(rng)

	for i := 0; i < 10; i++ {
		pii := piiGen.Generate()

		// FullName should have first and last name
		if !strings.Contains(pii.FullName, " ") {
			t.Errorf("FullName should contain space: %q", pii.FullName)
		}

		// Email should contain @
		if !strings.Contains(pii.Email, "@") {
			t.Errorf("Email should contain @: %q", pii.Email)
		}

		// Phone should have digits
		hasDigits := regexp.MustCompile(`\d`).MatchString(pii.Phone)
		if !hasDigits {
			t.Errorf("Phone should contain digits: %q", pii.Phone)
		}
	}
}

func TestPIIGeneratorVariety(t *testing.T) {
	seed := int64(42)
	rng := NewRNG(&seed)
	piiGen := NewPIIGenerator(rng)

	// Generate multiple PIIs and check they're not all the same
	names := make(map[string]bool)
	emails := make(map[string]bool)

	for i := 0; i < 20; i++ {
		pii := piiGen.Generate()
		names[pii.FullName] = true
		emails[pii.Email] = true
	}

	// Should have variety
	if len(names) < 10 {
		t.Errorf("Expected variety in names, got only %d unique names", len(names))
	}
	if len(emails) < 15 {
		t.Errorf("Expected variety in emails, got only %d unique emails", len(emails))
	}
}
