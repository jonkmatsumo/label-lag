package service

import (
	"context"
	"testing"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/store"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestListDecisions_Validation(t *testing.T) {
	mockStore := new(store.MockStore)
	svc := NewService(mockStore, nil)

	// Test invalid limit
	req := &pb.ListDecisionsRequest{Limit: -1}
	resp, err := svc.ListDecisions(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)

	// Test invalid offset
	req = &pb.ListDecisionsRequest{Offset: -1}
	resp, err = svc.ListDecisions(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
}

func TestListDecisions_Success(t *testing.T) {
	mockStore := new(store.MockStore)
	svc := NewService(mockStore, nil)

	req := &pb.ListDecisionsRequest{Limit: 10}
	expectedDecisions := []*pb.DecisionSummary{{RequestId: "req-1"}}

	mockStore.On("ListDecisions", mock.Anything, mock.MatchedBy(func(r *pb.ListDecisionsRequest) bool {
		return r.Limit == 10 && r.Offset == 0
	})).Return(expectedDecisions, int64(1), nil)

	resp, err := svc.ListDecisions(context.Background(), req)
	assert.NoError(t, err)
	assert.Len(t, resp.Decisions, 1)
	assert.Equal(t, int64(1), resp.Total)
}

func TestGetDecision_Validation(t *testing.T) {
	mockStore := new(store.MockStore)
	svc := NewService(mockStore, nil)

	req := &pb.GetDecisionRequest{RequestId: ""}
	resp, err := svc.GetDecision(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
}

func TestGetDecisionTrace_Validation(t *testing.T) {
	mockStore := new(store.MockStore)
	svc := NewService(mockStore, nil)

	req := &pb.GetDecisionTraceRequest{RequestId: ""}
	resp, err := svc.GetDecisionTrace(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
}

func TestGetRuleImpact_Validation(t *testing.T) {
	mockStore := new(store.MockStore)
	svc := NewService(mockStore, nil)

	req := &pb.GetRuleImpactRequest{RuleId: ""}
	resp, err := svc.GetRuleImpact(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
}
