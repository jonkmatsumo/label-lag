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

const datasetProfileRetention = 30 * 24 * time.Hour

func (s *SQLStore) SaveDatasetProfile(ctx context.Context, profile *pb.DatasetProfile) error {
	if profile == nil || profile.ProfileId == "" {
		return status.Error(codes.InvalidArgument, "profile and profile_id required")
	}
	if profile.TenantId == "" {
		return status.Error(codes.InvalidArgument, "tenant_id required")
	}

	query := `
		INSERT INTO dataset_profiles (
			profile_id, tenant_id, computed_at, record_count, feature_profiles
		) VALUES (
			$1, $2, $3, $4, $5
		)
		ON CONFLICT (profile_id) DO UPDATE SET
			computed_at = EXCLUDED.computed_at,
			record_count = EXCLUDED.record_count,
			feature_profiles = EXCLUDED.feature_profiles
	`

	profilesJSON, err := json.Marshal(profile.FeatureProfiles)
	if err != nil {
		return status.Errorf(codes.Internal, "failed to marshal feature profiles: %v", err)
	}

	computedAt := time.Now()
	if profile.ComputedAt != nil {
		computedAt = profile.ComputedAt.AsTime()
	}

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	_, err = s.db.ExecContext(queryCtx, query,
		profile.ProfileId,
		profile.TenantId,
		computedAt,
		profile.RecordCount,
		profilesJSON,
	)
	if err != nil {
		return db.MapDBError(err)
	}

	olderThan := computedAt.Add(-datasetProfileRetention)
	_, err = s.db.ExecContext(queryCtx,
		`DELETE FROM dataset_profiles WHERE tenant_id = $1 AND computed_at < $2`,
		profile.TenantId,
		olderThan,
	)
	if err != nil {
		return db.MapDBError(err)
	}

	return nil
}

func (s *SQLStore) GetDatasetProfileCached(ctx context.Context, profileID string, tenantID string) (*pb.DatasetProfile, error) {
	query := `
		SELECT
			profile_id, tenant_id, computed_at, record_count, feature_profiles
		FROM dataset_profiles
	`
	args := []interface{}{}
	if profileID == "latest" {
		if tenantID == "" {
			return nil, status.Error(codes.InvalidArgument, "tenant_id required for latest profile lookup")
		}
		query += " WHERE tenant_id = $1 ORDER BY computed_at DESC LIMIT 1"
		args = append(args, tenantID)
	} else {
		query += " WHERE profile_id = $1"
		args = append(args, profileID)
		if tenantID != "" {
			query += " AND tenant_id = $2"
			args = append(args, tenantID)
		}
	}

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var p pb.DatasetProfile
	var computedAt time.Time
	var profilesJSON []byte

	err := s.db.QueryRowContext(queryCtx, query, args...).Scan(
		&p.ProfileId,
		&p.TenantId,
		&computedAt,
		&p.RecordCount,
		&profilesJSON,
	)
	if err == sql.ErrNoRows {
		return nil, status.Errorf(codes.NotFound, "profile not found: %s", profileID)
	}
	if err != nil {
		return nil, db.MapDBError(err)
	}

	p.ComputedAt = timestamppb.New(computedAt)
	if len(profilesJSON) > 0 {
		if err := json.Unmarshal(profilesJSON, &p.FeatureProfiles); err != nil {
			return nil, status.Errorf(codes.Internal, "failed to unmarshal feature profiles: %v", err)
		}
	}

	return &p, nil
}

func (s *SQLStore) ListDatasetProfiles(ctx context.Context, req *pb.ListDatasetProfilesRequest) ([]*pb.DatasetProfile, int64, string, error) {
	queryBuilder := db.NewQueryBuilder(`
		SELECT
			profile_id, tenant_id, computed_at, record_count, feature_profiles
		FROM dataset_profiles
	`)

	if req.StartDate != nil {
		queryBuilder.AddCondition("computed_at >= ?", req.StartDate.AsTime())
	}
	if req.EndDate != nil {
		queryBuilder.AddCondition("computed_at <= ?", req.EndDate.AsTime())
	}
	if req.TenantId != "" {
		queryBuilder.AddCondition("tenant_id = ?", req.TenantId)
	}

	limit := int32(20)
	if req.Limit > 0 {
		limit = req.Limit
	}
	if req.Pagination != nil && req.Pagination.Limit > 0 {
		limit = req.Pagination.Limit
	}

	// Cursor pagination
	var cursorObj *profileCursor
	if req.Pagination != nil && req.Pagination.Cursor != "" {
		var err error
		cursorObj, err = decodeProfileCursor(req.Pagination.Cursor)
		if err != nil {
			return nil, 0, "", status.Errorf(codes.InvalidArgument, "invalid cursor: %v", err)
		}
		queryBuilder.AddCondition("(computed_at, profile_id) < (?, ?)", cursorObj.ComputedAt, cursorObj.ProfileId)
	}

	// Count
	var total int64
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	if cursorObj == nil {
		countQuery, countArgs := queryBuilder.BuildCount()
		err := s.db.QueryRowContext(queryCtx, countQuery, countArgs...).Scan(&total)
		if err != nil {
			return nil, 0, "", db.MapDBError(err)
		}
	}

	// List
	queryBuilder.AddOrderBy("computed_at DESC, profile_id DESC")
	queryBuilder.SetLimit(limit)
	if cursorObj == nil {
		queryBuilder.SetOffset(req.Offset)
	}
	selectQuery, selectArgs := queryBuilder.BuildSelect()

	rows, err := s.db.QueryContext(queryCtx, selectQuery, selectArgs...)
	if err != nil {
		return nil, 0, "", db.MapDBError(err)
	}
	defer rows.Close()

	var profiles []*pb.DatasetProfile
	var lastComputedAt time.Time
	var lastProfileID string

	for rows.Next() {
		var p pb.DatasetProfile
		var computedAt time.Time
		var profilesJSON []byte

		if err := rows.Scan(
			&p.ProfileId,
			&p.TenantId,
			&computedAt,
			&p.RecordCount,
			&profilesJSON,
		); err != nil {
			return nil, 0, "", fmt.Errorf("failed to scan profile: %v", err)
		}

		p.ComputedAt = timestamppb.New(computedAt)
		if len(profilesJSON) > 0 {
			if err := json.Unmarshal(profilesJSON, &p.FeatureProfiles); err != nil {
				return nil, 0, "", status.Errorf(codes.Internal, "failed to unmarshal feature profiles: %v", err)
			}
		}
		profiles = append(profiles, &p)
		lastComputedAt = computedAt
		lastProfileID = p.ProfileId
	}

	var nextCursor string
	if int32(len(profiles)) == limit && limit > 0 {
		nextCursor = encodeProfileCursor(lastComputedAt, lastProfileID)
	}

	return profiles, total, nextCursor, nil
}

func (s *SQLStore) GetLatestDatasetProfile(ctx context.Context, tenantID string) (*pb.GetLatestDatasetProfileResponse, error) {
	query := `
		SELECT
			profile_id, computed_at, record_count
		FROM dataset_profiles
	`
	args := []interface{}{}
	if tenantID != "" {
		query += " WHERE tenant_id = $1"
		args = append(args, tenantID)
	}
	query += " ORDER BY computed_at DESC, profile_id DESC LIMIT 1"
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var resp pb.GetLatestDatasetProfileResponse
	var computedAt time.Time
	err := s.db.QueryRowContext(queryCtx, query, args...).Scan(
		&resp.ProfileId,
		&computedAt,
		&resp.RecordCount,
	)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "no dataset profiles found")
	}
	if err != nil {
		return nil, db.MapDBError(err)
	}

	resp.ComputedAt = timestamppb.New(computedAt)
	return &resp, nil
}

func (s *SQLStore) PruneDatasetProfiles(ctx context.Context, olderThan time.Time) (int64, error) {
	return s.PruneDatasetProfilesByTenant(ctx, olderThan, "")
}

func (s *SQLStore) PruneDatasetProfilesByTenant(ctx context.Context, olderThan time.Time, tenantID string) (int64, error) {
	query := `DELETE FROM dataset_profiles WHERE computed_at < $1`
	args := []interface{}{olderThan}
	if tenantID != "" {
		query += ` AND tenant_id = $2`
		args = append(args, tenantID)
	}

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	result, err := s.db.ExecContext(queryCtx, query, args...)
	if err != nil {
		return 0, db.MapDBError(err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to get rows affected: %v", err)
	}

	return rowsAffected, nil
}
