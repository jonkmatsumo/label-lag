package httpserver

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBuildRiskComponents_Parity(t *testing.T) {
	tests := []struct {
		name     string
		features map[string]any
		expected []string // keys
	}{
		{
			name: "high velocity",
			features: map[string]any{
				"velocity_24h": 6,
			},
			expected: []string{"velocity"},
		},
		{
			name: "high amount ratio",
			features: map[string]any{
				"amount_to_avg_ratio_30d": 3.1,
			},
			expected: []string{"amount_ratio"},
		},
		{
			name: "low balance volatility",
			features: map[string]any{
				"balance_volatility_z_score": -2.1,
			},
			expected: []string{"balance"},
		},
		{
			name: "connection burst",
			features: map[string]any{
				"bank_connections_24h": 5,
			},
			expected: []string{"connections"},
		},
		{
			name: "high risk merchant",
			features: map[string]any{
				"merchant_risk_score": 71,
			},
			expected: []string{"merchant"},
		},
		{
			name: "insufficient history",
			features: map[string]any{
				"has_history": false,
			},
			expected: []string{"history"},
		},
		{
			name: "multiple components",
			features: map[string]any{
				"velocity_24h":        10,
				"merchant_risk_score": 80,
			},
			expected: []string{"velocity", "merchant"},
		},
		{
			name: "no components",
			features: map[string]any{
				"velocity_24h":               1,
				"amount_to_avg_ratio_30d":    1.0,
				"balance_volatility_z_score": 0.0,
				"bank_connections_24h":       0,
				"merchant_risk_score":        10,
				"has_history":                true,
			},
			expected: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			components := buildRiskComponents(tt.features)
			var keys []string
			for _, c := range components {
				keys = append(keys, c.Key)
			}
			assert.ElementsMatch(t, tt.expected, keys)
		})
	}
}
