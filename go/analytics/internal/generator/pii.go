package generator

import (
	"fmt"

	"github.com/brianvoe/gofakeit/v6"
)

// PIIGenerator produces realistic personally identifiable information.
// It wraps gofakeit with deterministic seeding from our RNG.
type PIIGenerator struct {
	faker *gofakeit.Faker
}

// NewPIIGenerator creates a PII generator seeded from the given RNG.
// This ensures deterministic PII generation.
func NewPIIGenerator(rng *RNG) *PIIGenerator {
	// Use RNG to generate a seed for gofakeit
	seed := rng.IntN(1<<31 - 1) // gofakeit uses int64, but smaller range is fine
	faker := gofakeit.New(int64(seed))
	return &PIIGenerator{faker: faker}
}

// PII represents personally identifiable information for a synthetic user.
type PII struct {
	FullName string
	Email    string
	Phone    string
}

// Generate creates a new set of PII data.
func (p *PIIGenerator) Generate() PII {
	firstName := p.faker.FirstName()
	lastName := p.faker.LastName()

	return PII{
		FullName: fmt.Sprintf("%s %s", firstName, lastName),
		Email:    p.faker.Email(),
		Phone:    p.faker.Phone(),
	}
}

// GenerateFullName returns a random full name.
func (p *PIIGenerator) GenerateFullName() string {
	return p.faker.Name()
}

// GenerateEmail returns a random email address.
func (p *PIIGenerator) GenerateEmail() string {
	return p.faker.Email()
}

// GeneratePhone returns a random phone number.
func (p *PIIGenerator) GeneratePhone() string {
	return p.faker.Phone()
}
