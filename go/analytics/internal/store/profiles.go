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

func (s *SQLStore) SaveDatasetProfile(ctx context.Context, profile *pb.DatasetProfile) error {
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

	return nil
}

func (s *SQLStore) GetDatasetProfileCached(ctx context.Context, profileID string) (*pb.DatasetProfile, error) {
	query := `
		SELECT
			profile_id, tenant_id, computed_at, record_count, feature_profiles
		FROM dataset_profiles
		WHERE profile_id = $1
	`

	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	var p pb.DatasetProfile
	var computedAt time.Time
	var profilesJSON []byte

	err := s.db.QueryRowContext(queryCtx, query, profileID).Scan(
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

func (s *SQLStore) ListDatasetProfiles(ctx context.Context, limit, offset int32) ([]*pb.DatasetProfile, int64, error) {
	// Count
	var total int64
	queryCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	err := s.db.QueryRowContext(queryCtx, "SELECT COUNT(*) FROM dataset_profiles").Scan(&total)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}

	// List
	query := `
		SELECT
			profile_id, tenant_id, computed_at, record_count, feature_profiles
		FROM dataset_profiles
		ORDER BY computed_at DESC
		LIMIT $1 OFFSET $2
	`

	rows, err := s.db.QueryContext(queryCtx, query, limit, offset)
	if err != nil {
		return nil, 0, db.MapDBError(err)
	}
	defer rows.Close()

	var profiles []*pb.DatasetProfile
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
			return nil, 0, fmt.Errorf("failed to scan profile: %v", err)
		}

		p.ComputedAt = timestamppb.New(computedAt)
		if len(profilesJSON) > 0 {
			if err := json.Unmarshal(profilesJSON, &p.FeatureProfiles); err != nil {
				return nil, 0, status.Errorf(codes.Internal, "failed to unmarshal feature profiles: %v", err)
			}
		}
		profiles = append(profiles, &p)
	}

	return profiles, total, nil
}
