package store

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestListDecisions(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	req := &pb.ListDecisionsRequest{
		UserId:   "user-1",
		Decision: "APPROVE",
		Limit:    10,
		Offset:   0,
	}

	mock.ExpectQuery("SELECT COUNT").
		WithArgs("user-1", "APPROVE").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(1))

	mock.ExpectQuery("SELECT request_id, user_id, ts, final_score, decision").
		WithArgs("user-1", "APPROVE", int32(10), int32(0)).
		WillReturnRows(sqlmock.NewRows([]string{"request_id", "user_id", "ts", "final_score", "decision"}).
			AddRow("req-1", "user-1", time.Now(), 45, "APPROVE"))

	decisions, total, err := s.ListDecisions(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, int64(1), total)
	assert.Len(t, decisions, 1)
	assert.Equal(t, "req-1", decisions[0].RequestId)
	assert.Equal(t, int32(45), decisions[0].FinalScore)
}

func TestGetDecision(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	ruleImpactsJSON := `[{"rule_id": "rule-1", "is_shadow": false, "score_delta": 10.5}]`
	rows := sqlmock.NewRows([]string{"request_id", "ts", "model_version", "rules_version", "model_score", "final_score", "rule_impacts", "user_id", "decision"}).
		AddRow("req-1", time.Now(), "v1", "v1", 35, 45, []byte(ruleImpactsJSON), "user-1", "APPROVE")

	mock.ExpectQuery(`SELECT .* FROM inference_events WHERE request_id = \$1`).
		WithArgs("req-1").
		WillReturnRows(rows)

	decision, err := s.GetDecision(context.Background(), "req-1")
	require.NoError(t, err)
	assert.Equal(t, "req-1", decision.RequestId)
	assert.Len(t, decision.RuleImpacts, 1)
	assert.Equal(t, "rule-1", decision.RuleImpacts[0].RuleId)
	assert.Equal(t, int32(45), decision.FinalScore)
}

func TestGetDecisionTrace(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	rows := sqlmock.NewRows([]string{"rule_id", "is_shadow", "score_delta"}).
		AddRow("rule-1", false, 10.5).
		AddRow("rule-2", true, 5.0)

	mock.ExpectQuery("SELECT rule_id, is_shadow, score_delta FROM rule_impacts").
		WithArgs("req-1").
		WillReturnRows(rows)

	trace, err := s.GetDecisionTrace(context.Background(), "req-1")
	require.NoError(t, err)
	assert.Len(t, trace, 2)
	assert.Equal(t, "rule-1", trace[0].RuleId)
	assert.False(t, trace[0].IsShadow)
	assert.Equal(t, "rule-2", trace[1].RuleId)
	assert.True(t, trace[1].IsShadow)
}

func TestGetRuleImpact(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	startDate := time.Now().AddDate(0, 0, -7).UTC()
	req := &pb.GetRuleImpactRequest{
		RuleId:    "rule-1",
		StartDate: timestamppb.New(startDate),
	}

	mock.ExpectQuery(`SELECT COUNT\(\*\), COALESCE\(AVG\(ri.score_delta\), 0\)`).
		WithArgs("rule-1", startDate).
		WillReturnRows(sqlmock.NewRows([]string{"count", "avg"}).AddRow(100, 15.5))

	mock.ExpectQuery(`SELECT DATE\(ie.ts\) as date, COUNT\(\*\), COALESCE\(AVG\(ri.score_delta\), 0\), SUM`).
		WithArgs("rule-1", startDate).
		WillReturnRows(sqlmock.NewRows([]string{"date", "count", "avg", "changes"}).
			AddRow(time.Now().UTC(), 50, 14.0, 5).
			AddRow(time.Now().AddDate(0, 0, -1).UTC(), 50, 17.0, 10))

	resp, err := s.GetRuleImpact(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "rule-1", resp.RuleId)
	assert.Equal(t, int64(100), resp.TotalTriggers)
	assert.Equal(t, 15.5, resp.AvgScoreDelta)
	assert.Len(t, resp.DailyBuckets, 2)
}
