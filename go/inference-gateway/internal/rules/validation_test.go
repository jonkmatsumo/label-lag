package rules

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValidateRuleset(t *testing.T) {
	score70 := 70
	score80 := 80

	rs := RuleSet{
		Rules: []Rule{
			{
				ID:     "rule1",
				Field:  "velocity",
				Op:     ">",
				Value:  10,
				Action: "override_score",
				Score:  &score70,
				Status: RuleStatusActive,
			},
			{
				ID:     "rule2",
				Field:  "velocity",
				Op:     ">",
				Value:  20,
				Action: "override_score",
				Score:  &score80,
				Status: RuleStatusActive,
			},
			{
				ID:     "rule3",
				Field:  "amount",
				Op:     ">",
				Value:  100,
				Action: "reject",
				Status: RuleStatusActive,
			},
			{
				ID:     "rule4",
				Field:  "amount",
				Op:     "<",
				Value:  150,
				Action: "override_score",
				Score:  &score70,
				Status: RuleStatusActive,
			},
			{
				ID:     "rule5",
				Field:  "balance",
				Op:     "==",
				Value:  0,
				Action: "reject",
				Status: RuleStatusShadow,
			},
			{
				ID:     "rule6",
				Field:  "balance",
				Op:     "==",
				Value:  0,
				Action: "override_score",
				Score:  &score80,
				Status: RuleStatusActive,
			},
		},
	}

	conflicts := ValidateRuleset(rs)

	// Expected conflicts:
	// 1. rule3 and rule4 on "amount": range overlap + reject/override clash (Breaking)
	// 2. rule5 and rule6 on "balance": same op + shadow/active overlap (Breaking because of reject/override)
	// 3. rule1 and rule2 on "velocity": same field same op (Warning)

	assert.Len(t, conflicts, 3)

	// rule3, rule4
	assert.Equal(t, "amount", conflicts[0].Field)
	assert.Equal(t, ConflictKindActionType, conflicts[0].Kind)
	assert.Equal(t, ConflictSeverityBreaking, conflicts[0].Severity)

	// rule5, rule6
	assert.Equal(t, "balance", conflicts[1].Field)
	assert.Equal(t, ConflictKindSameOp, conflicts[1].Kind)
	assert.Equal(t, ConflictSeverityBreaking, conflicts[1].Severity)

	// rule1, rule2
	assert.Equal(t, "velocity", conflicts[2].Field)
	assert.Equal(t, ConflictKindSameOp, conflicts[2].Kind)
	assert.Equal(t, ConflictSeverityWarning, conflicts[2].Severity)
}

func TestCheckNumericOverlap(t *testing.T) {
	tests := []struct {
		op1, op2 string
		v1, v2   float64
		expected bool
	}{
		{">", "<", 10, 5, false},
		{">", "<", 10, 15, true},
		{">", ">", 10, 20, true},
		{"<", "<", 5, 10, true},
		{"==", ">", 15, 10, true},
		{"==", "<", 5, 10, true},
		{"==", "==", 10, 10, true},
		{"==", "==", 10, 20, false},
	}

	for _, tt := range tests {
		t.Run(tt.op1+tt.op2, func(t *testing.T) {
			assert.Equal(t, tt.expected, checkNumericOverlap(tt.op1, tt.v1, tt.op2, tt.v2))
		})
	}
}
