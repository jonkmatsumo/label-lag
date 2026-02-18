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

func TestDiffRuleVersions_BreakingWhenCoreBehaviorChanges(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	mock.ExpectQuery("SELECT rule_json FROM rule_versions WHERE rule_id = \\$1 AND version_id = \\$2").
		WithArgs("rule-1", "v2").
		WillReturnRows(sqlmock.NewRows([]string{"rule_json"}).
			AddRow([]byte(`{"id":"rule-1","field":"amount","op":"<","value_json":"10","action":"flag"}`)))
	mock.ExpectQuery("SELECT rule_json FROM rule_versions WHERE rule_id = \\$1 AND version_id = \\$2").
		WithArgs("rule-1", "v1").
		WillReturnRows(sqlmock.NewRows([]string{"rule_json"}).
			AddRow([]byte(`{"id":"rule-1","field":"amount","op":">","value_json":"5","action":"flag"}`)))

	resp, err := s.DiffRuleVersions(context.Background(), "rule-1", "v2", "v1", "")
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.True(t, resp.IsBreaking)
	require.NotEmpty(t, resp.Changes)
	assert.Equal(t, "op", resp.Changes[0].FieldName)
	assert.Equal(t, "modified", resp.Changes[0].ChangeType)
	assert.Equal(t, ">", resp.Changes[0].BeforeValue)
	assert.Equal(t, "<", resp.Changes[0].AfterValue)
	assert.NoError(t, mock.ExpectationsWereMet())
}

func TestDiffRuleVersions_NonBreakingForSameTypeValueChange(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	mock.ExpectQuery("SELECT rule_json FROM rule_versions WHERE rule_id = \\$1 AND version_id = \\$2").
		WithArgs("rule-1", "v2").
		WillReturnRows(sqlmock.NewRows([]string{"rule_json"}).
			AddRow([]byte(`{"id":"rule-1","field":"amount","op":">","value_json":"10","action":"flag"}`)))
	mock.ExpectQuery("SELECT rule_json FROM rule_versions WHERE rule_id = \\$1 AND version_id = \\$2").
		WithArgs("rule-1", "v1").
		WillReturnRows(sqlmock.NewRows([]string{"rule_json"}).
			AddRow([]byte(`{"id":"rule-1","field":"amount","op":">","value_json":"5","action":"flag"}`)))

	resp, err := s.DiffRuleVersions(context.Background(), "rule-1", "v2", "v1", "")
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.False(t, resp.IsBreaking)
	require.Len(t, resp.Changes, 1)
	assert.Equal(t, "value", resp.Changes[0].FieldName)
	assert.Equal(t, "modified", resp.Changes[0].ChangeType)
	assert.Equal(t, "5", resp.Changes[0].BeforeValue)
	assert.Equal(t, "10", resp.Changes[0].AfterValue)
	assert.NoError(t, mock.ExpectationsWereMet())
}
