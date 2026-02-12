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

func TestSaveDatasetProfile(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	profile := &pb.DatasetProfile{
		ProfileId:   "prof-1",
		TenantId:    "tenant-1",
		ComputedAt:  timestamppb.New(time.Now().UTC()),
		RecordCount: 100,
		FeatureProfiles: []*pb.FeatureProfile{
			{Name: "f1", Type: "numeric"},
		},
	}

	mock.ExpectExec("INSERT INTO dataset_profiles").
		WithArgs("prof-1", "tenant-1", sqlmock.AnyArg(), int64(100), sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	err = s.SaveDatasetProfile(context.Background(), profile)
	require.NoError(t, err)
}

func TestGetDatasetProfileCached(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	ts := time.Now().UTC()
	jsonProfiles := `[{"name":"f1","type":"numeric"}]`

	mock.ExpectQuery("SELECT profile_id").
		WithArgs("prof-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"profile_id", "tenant_id", "computed_at", "record_count", "feature_profiles",
		}).AddRow("prof-1", "tenant-1", ts, 100, []byte(jsonProfiles)))

	p, err := s.GetDatasetProfileCached(context.Background(), "prof-1")
	require.NoError(t, err)
	assert.Equal(t, "prof-1", p.ProfileId)
	assert.Equal(t, "tenant-1", p.TenantId)
	assert.Equal(t, int64(100), p.RecordCount)
	assert.Len(t, p.FeatureProfiles, 1)
	assert.Equal(t, "f1", p.FeatureProfiles[0].Name)
}

func TestListDatasetProfiles(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	mock.ExpectQuery("SELECT COUNT").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(1))

	ts := time.Now().UTC()
	jsonProfiles := `[]`

	start := time.Now().AddDate(0, 0, -30)
	end := time.Now()

	mock.ExpectQuery("SELECT profile_id").
		WithArgs(sqlmock.AnyArg(), sqlmock.AnyArg()).
		WillReturnRows(sqlmock.NewRows([]string{
			"profile_id", "tenant_id", "computed_at", "record_count", "feature_profiles",
		}).AddRow("prof-1", "tenant-1", ts, 100, []byte(jsonProfiles)))

	profiles, total, err := s.ListDatasetProfiles(context.Background(), &pb.ListDatasetProfilesRequest{
		Limit:     10,
		Offset:    0,
		StartDate: timestamppb.New(start),
		EndDate:   timestamppb.New(end),
	})
	require.NoError(t, err)
	assert.Equal(t, int64(1), total)
	assert.Len(t, profiles, 1)
	assert.Equal(t, "prof-1", profiles[0].ProfileId)
}

func TestPruneDatasetProfiles(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	cutoff := time.Now().Add(-24 * time.Hour)

	mock.ExpectExec("DELETE FROM dataset_profiles").
		WithArgs(cutoff).
		WillReturnResult(sqlmock.NewResult(0, 5))

	count, err := s.PruneDatasetProfiles(context.Background(), cutoff)
	require.NoError(t, err)
	assert.Equal(t, int64(5), count)
}
