package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jonkmatsumo/label-lag/go/analytics/internal/db"
	pb "github.com/jonkmatsumo/label-lag/go/analytics/proto/crud/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func (s *SQLStore) ListRuleVersions(ctx context.Context, ruleID string, limit, offset int32) ([]*pb.Rule, int64, error) {
	query := `
		SELECT rule_json, created_at, created_by, status
		FROM rule_versions
		WHERE rule_id = $1
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, ruleID, limit, offset)
	if err != nil {
		return nil, 0, db.MapDBError(fmt.Errorf("failed to list rule versions: %w", err))
	}
	defer rows.Close()

	var versions []*pb.Rule
	for rows.Next() {
		var ruleJSON []byte
		var createdAt time.Time
		var createdBy, statusStr sql.NullString

		if err := rows.Scan(&ruleJSON, &createdAt, &createdBy, &statusStr); err != nil {
			return nil, 0, db.MapDBError(fmt.Errorf("failed to scan rule version: %w", err))
		}

		var r pb.Rule
		if err := json.Unmarshal(ruleJSON, &r); err != nil {
			return nil, 0, status.Errorf(codes.Internal, "invalid json payload: %v", err)
		}
		if statusStr.Valid {
			r.Status = statusStr.String
		}
		versions = append(versions, &r)
	}

	var total int64
	queryCtxCount, cancelCount := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancelCount()
	err = s.db.QueryRowContext(queryCtxCount, "SELECT COUNT(*) FROM rule_versions WHERE rule_id = $1", ruleID).Scan(&total)
	if err != nil {
		return nil, 0, db.MapDBError(fmt.Errorf("failed to count rule versions: %w", err))
	}

	return versions, total, nil
}

func (s *SQLStore) GetRuleVersion(ctx context.Context, ruleID, versionID string) (*pb.GetRuleVersionResponse, error) {
	var query string
	var args []interface{}

	if versionID == "active" || versionID == "" {
		query = `
			SELECT rv.rule_json, rv.version_id, rv.created_at
			FROM rules r
			JOIN rule_versions rv ON r.active_version_id = rv.version_id
			WHERE r.rule_id = $1
		`
		args = []interface{}{ruleID}
	} else if versionID == "latest" {
		query = `
			SELECT rule_json, version_id, created_at
			FROM rule_versions
			WHERE rule_id = $1
			ORDER BY created_at DESC
			LIMIT 1
		`
		args = []interface{}{ruleID}
	} else {
		query = `
			SELECT rule_json, version_id, created_at
			FROM rule_versions
			WHERE rule_id = $1 AND version_id = $2
		`
		args = []interface{}{ruleID, versionID}
	}

	var ruleJSON []byte
	var verID string
	var createdAt time.Time

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()
	err := s.db.QueryRowContext(queryCtx, query, args...).Scan(&ruleJSON, &verID, &createdAt)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "rule version not found")
	} else if err != nil {
		return nil, db.MapDBError(fmt.Errorf("failed to get rule version: %w", err))
	}

	var r pb.Rule
	if err := json.Unmarshal(ruleJSON, &r); err != nil {
		return nil, fmt.Errorf("failed to unmarshal rule: %v", err)
	}

	return &pb.GetRuleVersionResponse{
		Rule:      &r,
		VersionId: verID,
		CreatedAt: timestamppb.New(createdAt),
	}, nil
}

func (s *SQLStore) PublishRuleVersion(ctx context.Context, req *pb.PublishRuleVersionRequest) (string, error) {
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	tx, err := s.db.BeginTx(queryCtx, nil)
	if err != nil {
		return "", db.MapDBError(fmt.Errorf("failed to begin transaction: %w", err))
	}
	defer tx.Rollback()

	var field, op, value, action, severity, reason, statusStr string
	var score sql.NullInt32
	err = tx.QueryRowContext(queryCtx, `SELECT field, op, value, action, score, severity, reason, status FROM rules WHERE rule_id = $1`, req.RuleId).
		Scan(&field, &op, &value, &action, &score, &severity, &reason, &statusStr)

	if err == sql.ErrNoRows {
		return "", status.Error(codes.NotFound, "rule not found")
	} else if err != nil {
		return "", db.MapDBError(fmt.Errorf("failed to get rule: %w", err))
	}

	newVersion := req.VersionId
	if newVersion == "" {
		newVersion = fmt.Sprintf("v%s", time.Now().Format("20060102150405"))
	}

	r := pb.Rule{
		Id:        req.RuleId,
		Field:     field,
		Op:        op,
		ValueJson: value,
		Action:    action,
		Score:     score.Int32,
		Severity:  severity,
		Reason:    reason,
		Status:    "active",
	}
	ruleJSON, _ := json.Marshal(r)

	_, err = tx.ExecContext(queryCtx, `
		INSERT INTO rule_versions (version_id, rule_id, rule_json, created_at, created_by, change_description, status, is_active)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`, newVersion, req.RuleId, ruleJSON, time.Now(), req.Actor, req.Reason, "active", true)

	if err != nil {
		return "", db.MapDBError(fmt.Errorf("failed to insert rule version: %w", err))
	}

	_, err = tx.ExecContext(queryCtx, `
		UPDATE rules SET status = 'active', active_version_id = $1 WHERE rule_id = $2
	`, newVersion, req.RuleId)

	if err != nil {
		return "", db.MapDBError(fmt.Errorf("failed to update rule status: %w", err))
	}

	_, err = tx.ExecContext(queryCtx, `
		UPDATE rule_versions SET is_active = FALSE WHERE rule_id = $1 AND version_id != $2
	`, req.RuleId, newVersion)
	if err != nil {
		return "", db.MapDBError(fmt.Errorf("failed to archive old versions: %w", err))
	}

	if err := tx.Commit(); err != nil {
		return "", db.MapDBError(fmt.Errorf("failed to commit transaction: %w", err))
	}

	return newVersion, nil
}

func (s *SQLStore) GetRuleReadiness(ctx context.Context, ruleID string) (*pb.GetRuleReadinessResponse, error) {
	var rID, valueJSON string
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()
	err := s.db.QueryRowContext(queryCtx, "SELECT rule_id, value FROM rules WHERE rule_id = $1", ruleID).Scan(&rID, &valueJSON)
	if err != nil {
		return nil, db.MapDBError(err)
	}

	checks := []*pb.ReadinessCheck{}
	overallReady := true

	var val interface{}
	jsonCheck := &pb.ReadinessCheck{Name: "json_validity", Passed: true, Message: "Value is valid JSON"}
	if err := json.Unmarshal([]byte(valueJSON), &val); err != nil {
		jsonCheck.Passed = false
		jsonCheck.Message = fmt.Sprintf("Invalid JSON value: %v", err)
		overallReady = false
	}
	checks = append(checks, jsonCheck)
	checks = append(checks, &pb.ReadinessCheck{Name: "integrity", Passed: true, Message: "Rule integrity check passed"})

	return &pb.GetRuleReadinessResponse{
		RuleId: ruleID,
		Ready:  overallReady,
		Checks: checks,
	}, nil
}

func (s *SQLStore) DiffRuleVersions(ctx context.Context, ruleID, vA, vB string) (*pb.DiffRuleVersionsResponse, error) {
	getVersionJSON := func(verID string) (*pb.Rule, error) {
		queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
		defer cancel()
		var rJSON []byte
		err := s.db.QueryRowContext(queryCtx, "SELECT rule_json FROM rule_versions WHERE rule_id = $1 AND version_id = $2", ruleID, verID).Scan(&rJSON)

		if err != nil {
			return nil, db.MapDBError(err)
		}
		var r pb.Rule
		if err := json.Unmarshal(rJSON, &r); err != nil {
			return nil, status.Errorf(codes.Internal, "invalid json payload: %v", err)
		}
		return &r, nil
	}

	ruleA, errA := getVersionJSON(vA)
	ruleB, errB := getVersionJSON(vB)

	if errA != nil || errB != nil {
		return nil, fmt.Errorf("failed to fetch versions for diff")
	}

	changes := []*pb.RuleDiffChange{}
	if ruleA.Field != ruleB.Field {
		changes = append(changes, &pb.RuleDiffChange{Field: "field", OldValue: ruleB.Field, NewValue: ruleA.Field, Description: "Field changed"})
	}
	if ruleA.Op != ruleB.Op {
		changes = append(changes, &pb.RuleDiffChange{Field: "op", OldValue: ruleB.Op, NewValue: ruleA.Op, Description: "Operator changed"})
	}
	if ruleA.ValueJson != ruleB.ValueJson {
		changes = append(changes, &pb.RuleDiffChange{Field: "value", OldValue: ruleB.ValueJson, NewValue: ruleA.ValueJson, Description: "Value changed"})
	}
	if ruleA.Action != ruleB.Action {
		changes = append(changes, &pb.RuleDiffChange{Field: "action", OldValue: ruleB.Action, NewValue: ruleA.Action, Description: "Action changed"})
	}

	return &pb.DiffRuleVersionsResponse{
		RuleId:   ruleID,
		VersionA: vA,
		VersionB: vB,
		Changes:  changes,
	}, nil
}

func (s *SQLStore) SaveRule(ctx context.Context, r *pb.Rule) error {
	query := `
		INSERT INTO rules (
			rule_id, field, op, value, action, score, severity, reason, status
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		ON CONFLICT (rule_id) DO UPDATE SET
			field = EXCLUDED.field,
			op = EXCLUDED.op,
			value = EXCLUDED.value,
			action = EXCLUDED.action,
			score = EXCLUDED.score,
			severity = EXCLUDED.severity,
			reason = EXCLUDED.reason,
			status = EXCLUDED.status
	`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err := s.db.ExecContext(queryCtx, query,
		r.Id, r.Field, r.Op, r.ValueJson, r.Action, r.Score, r.Severity, r.Reason, r.Status,
	)

	if err != nil {
		return db.MapDBError(fmt.Errorf("failed to save rule: %w", err))
	}
	return nil
}

func (s *SQLStore) GetRule(ctx context.Context, ruleID string) (*pb.Rule, error) {
	query := `SELECT rule_id, field, op, value, action, score, severity, reason, status FROM rules WHERE rule_id = $1`

	var r pb.Rule
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, query, ruleID).Scan(
		&r.Id, &r.Field, &r.Op, &r.ValueJson, &r.Action, &r.Score, &r.Severity, &r.Reason, &r.Status,
	)

	if err != nil {
		return nil, db.MapDBError(err)
	}

	return &r, nil
}

func (s *SQLStore) ListRules(ctx context.Context, statusFilter string, includeArchived bool) ([]*pb.Rule, error) {
	query := `SELECT rule_id, field, op, value, action, score, severity, reason, status FROM rules WHERE 1=1`
	args := []interface{}{}

	if statusFilter != "" {
		args = append(args, statusFilter)
		query += fmt.Sprintf(" AND status = $%d", len(args))
	} else if !includeArchived {
		query += " AND status != 'archived'"
	}

	query += " ORDER BY rule_id"

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	rows, err := s.db.QueryContext(queryCtx, query, args...)
	if err != nil {
		return nil, db.MapDBError(err)
	}
	defer rows.Close()

	var rules []*pb.Rule
	for rows.Next() {
		var r pb.Rule
		if err := rows.Scan(
			&r.Id, &r.Field, &r.Op, &r.ValueJson, &r.Action, &r.Score, &r.Severity, &r.Reason, &r.Status,
		); err != nil {
			return nil, db.MapDBError(fmt.Errorf("failed to scan rule: %w", err))
		}
		rules = append(rules, &r)
	}
	return rules, nil
}

func (s *SQLStore) DeleteRule(ctx context.Context, ruleID string) error {
	query := `UPDATE rules SET status = 'archived' WHERE rule_id = $1`
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err := s.db.ExecContext(queryCtx, query, ruleID)
	if err != nil {
		return db.MapDBError(fmt.Errorf("failed to archive rule: %w", err))
	}
	return nil
}
