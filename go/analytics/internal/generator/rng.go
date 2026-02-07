// Package generator provides synthetic data generation for testing and development.
// It mirrors the Python DataGenerator's behavior for parity testing.
package generator

import (
	"math"
	"math/rand/v2"
)

// RNG wraps a seeded random number generator for deterministic output.
// Using math/rand/v2 with PCG for consistency across runs.
type RNG struct {
	src *rand.Rand
}

// NewRNG creates a new RNG with the given seed.
// If seed is nil, uses a random seed.
func NewRNG(seed *int64) *RNG {
	var src *rand.Rand
	if seed != nil {
		// PCG seeding: use the seed directly
		src = rand.New(rand.NewPCG(uint64(*seed), uint64(*seed)>>32))
	} else {
		// Random seed from global source
		src = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return &RNG{src: src}
}

// Float64 returns a pseudo-random float64 in [0.0, 1.0).
func (r *RNG) Float64() float64 {
	return r.src.Float64()
}

// Float64Range returns a pseudo-random float64 in [min, max).
func (r *RNG) Float64Range(min, max float64) float64 {
	return min + r.src.Float64()*(max-min)
}

// IntN returns a pseudo-random int in [0, n).
func (r *RNG) IntN(n int) int {
	return r.src.IntN(n)
}

// IntRange returns a pseudo-random int in [min, max).
func (r *RNG) IntRange(min, max int) int {
	if min >= max {
		return min
	}
	return min + r.src.IntN(max-min)
}

// NormFloat64 returns a normally distributed float64 with mean 0 and stddev 1.
func (r *RNG) NormFloat64() float64 {
	return r.src.NormFloat64()
}

// LogNormal returns a log-normally distributed value.
// mu and sigma are the mean and standard deviation of the underlying normal distribution.
// This matches numpy's lognormal: exp(mu + sigma * N(0,1))
func (r *RNG) LogNormal(mu, sigma float64) float64 {
	z := r.NormFloat64()
	return math.Exp(mu + sigma*z)
}

// Bool returns true with probability p.
func (r *RNG) Bool(p float64) bool {
	return r.Float64() < p
}
