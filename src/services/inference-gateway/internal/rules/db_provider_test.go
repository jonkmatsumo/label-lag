package rules

import (
	"context"
	"database/sql"
	"encoding/json"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/stretchr/testify/assert"
)

func TestDBProvider_GetRules(t *testing.T) {
	db, mock, err := sqlmock.New()
	assert.NoError(t, err)
	defer db.Close()

	p := &DBProvider{db: db}

	ctx := context.Background()

	t.Run("success", func(t *testing.T) {
		mock.ExpectQuery("SELECT id, version_name FROM published_rulesets").
			WillReturnRows(sqlmock.NewRows([]string{"id", "version_name"}).AddRow(1, "v1"))

		val, _ := json.Marshal(map[string]any{"v": 1000})
		mock.ExpectQuery("SELECT rv.rule_id, rv.field, rv.op, rv.value, rv.action, rv.score, rv.severity, rv.reason").
			WithArgs(1).
			WillReturnRows(sqlmock.NewRows([]string{"rule_id", "field", "op", "value", "action", "score", "severity", "reason"}).
				AddRow("rule1", "amount", ">", val, "reject", nil, "high", "too high"))

		ruleset, err := p.GetRules(ctx)
		assert.NoError(t, err)
		assert.Equal(t, "v1", ruleset.Version)
		assert.Len(t, ruleset.Rules, 1)
		assert.Equal(t, "rule1", ruleset.Rules[0].ID)
		assert.Equal(t, 1000.0, ruleset.Rules[0].Value) // json.Unmarshal into any gives float64 for numbers
	})

	t.Run("no ruleset", func(t *testing.T) {
		mock.ExpectQuery("SELECT id, version_name FROM published_rulesets").
			WillReturnError(sql.ErrNoRows)

		_, err := p.GetRules(ctx)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no published ruleset found")
	})
}
