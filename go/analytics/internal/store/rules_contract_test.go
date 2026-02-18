package store

import (
	"context"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetRuleReadiness_ValidJSONSetsPassStatuses(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	mock.ExpectQuery("SELECT rule_id, value FROM rules WHERE rule_id = \\$1").
		WithArgs("rule-1").
		WillReturnRows(sqlmock.NewRows([]string{"rule_id", "value"}).
			AddRow("rule-1", `{"threshold":10}`))

	resp, err := s.GetRuleReadiness(context.Background(), "rule-1", "")
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.True(t, resp.Ready)
	assert.Equal(t, pb.ReadinessStatus_READINESS_STATUS_PASS, resp.OverallStatus)
	require.Len(t, resp.Checks, 2)
	assert.Equal(t, pb.ReadinessStatus_READINESS_STATUS_PASS, resp.Checks[0].Status)
	assert.Equal(t, pb.ReadinessStatus_READINESS_STATUS_PASS, resp.Checks[1].Status)
	assert.True(t, resp.Checks[0].Passed)
	assert.True(t, resp.Checks[1].Passed)
	assert.NoError(t, mock.ExpectationsWereMet())
}

func TestGetRuleReadiness_InvalidJSONSetsFailStatuses(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	mock.ExpectQuery("SELECT rule_id, value FROM rules WHERE rule_id = \\$1").
		WithArgs("rule-2").
		WillReturnRows(sqlmock.NewRows([]string{"rule_id", "value"}).
			AddRow("rule-2", `{"threshold":`))

	resp, err := s.GetRuleReadiness(context.Background(), "rule-2", "")
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.False(t, resp.Ready)
	assert.Equal(t, pb.ReadinessStatus_READINESS_STATUS_FAIL, resp.OverallStatus)
	require.Len(t, resp.Checks, 2)
	assert.Equal(t, pb.ReadinessStatus_READINESS_STATUS_FAIL, resp.Checks[0].Status)
	assert.Equal(t, pb.ReadinessStatus_READINESS_STATUS_PASS, resp.Checks[1].Status)
	assert.False(t, resp.Checks[0].Passed)
	assert.True(t, resp.Checks[1].Passed)
	assert.NoError(t, mock.ExpectationsWereMet())
}
