package store

import (
	"context"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestGetDatasetProfile_DynamicFeatures(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	// 1. Total records count
	mock.ExpectQuery(`SELECT COUNT\(\*\) FROM generated_records`).
		WithArgs("tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(100))

	// 2. Static numeric features (just mock one, then mock the rest)
	for i := 0; i < 6; i++ {
		mock.ExpectQuery(`SELECT AVG`).WillReturnRows(sqlmock.NewRows([]string{"mean", "stddev", "null_count", "min_val", "max_val"}).
			AddRow(10.0, 1.0, 0, 0.0, 20.0))
		mock.ExpectQuery(`SELECT WIDTH_BUCKET`).WillReturnRows(sqlmock.NewRows([]string{"bucket", "count"}).
			AddRow(1, 100))
	}

	// 3. Discover dynamic numeric keys
	mock.ExpectQuery(`(?s)SELECT DISTINCT kv\.key.*gr\.numerical_features.*LIMIT \$2`).
		WithArgs("tenant-1", MaxNumericKeysProfiled).
		WillReturnRows(sqlmock.NewRows([]string{"key"}).AddRow("dyn_num_1"))

	// 4. Profile dynamic numeric key
	mock.ExpectQuery(`(?s)WITH scoped AS.*jsonb_each_text\(s\.feature_map\).*WHERE key = \$1.*SELECT\s+AVG\(numeric_value\)::float8`).
		WithArgs("dyn_num_1", "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"mean", "stddev", "null_count", "min_val", "max_val"}).
			AddRow(100.0, 5.0, 5, 80.0, 120.0))
	mock.ExpectQuery(`(?s)WITH scoped AS.*SELECT\s+WIDTH_BUCKET\(numeric_value,`).
		WithArgs("dyn_num_1", "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"bucket", "count"}).
			AddRow(1, 95))

	// 5. Discover dynamic categorical keys
	mock.ExpectQuery(`(?s)SELECT DISTINCT kv\.key.*gr\.categorical_features.*LIMIT \$2`).
		WithArgs("tenant-1", MaxCategoricalKeysProfiled).
		WillReturnRows(sqlmock.NewRows([]string{"key"}).AddRow("dyn_cat_1"))

	// 6. Profile dynamic categorical key
	// Null rate
	mock.ExpectQuery(`(?s)WITH scoped AS.*jsonb_each_text\(s\.feature_map\).*WHERE key = \$1.*WHERE value IS NULL`).
		WithArgs("dyn_cat_1", "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(10))
	// Top-K
	mock.ExpectQuery(`(?s)WITH scoped AS.*SELECT value, COUNT\(\*\) as count.*GROUP BY value.*LIMIT \$3`).
		WithArgs("dyn_cat_1", "tenant-1", DefaultTopK).
		WillReturnRows(sqlmock.NewRows([]string{"value", "count"}).
			AddRow("val1", 50).
			AddRow("val2", 30))

	resp, err := s.GetDatasetProfile(context.Background(), "test-dataset", 50, 5, "tenant-1")
	require.NoError(t, err)
	assert.Equal(t, int64(100), resp.TotalRecords)

	// Verify numeric dynamic feature
	foundNum := false
	for _, p := range resp.FeatureProfiles {
		if p.Name == "dyn_num_1" {
			foundNum = true
			assert.Equal(t, "numeric", p.Type)
			assert.Equal(t, 0.05, p.NullRate)
			assert.Equal(t, 100.0, p.Mean)
		}
	}
	assert.True(t, foundNum)

	// Verify categorical dynamic feature
	foundCat := false
	for _, p := range resp.FeatureProfiles {
		if p.Name == "dyn_cat_1" {
			foundCat = true
			assert.Equal(t, "categorical", p.Type)
			assert.Equal(t, 0.1, p.NullRate)
			assert.Len(t, p.TopValues, 3) // val1, val2, and _other
			assert.Equal(t, "val1", p.TopValues[0].Value)
			assert.Equal(t, int64(50), p.TopValues[0].Count)
			assert.Equal(t, "_other", p.TopValues[2].Value)
			assert.Equal(t, int64(10), p.TopValues[2].Count) // 100 total - 10 null - 50 val1 - 30 val2 = 10 other
		}
	}
	assert.True(t, foundCat)
}

func TestGetDatasetProfile_Caps(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	mock.ExpectQuery("SELECT COUNT").WithArgs("tenant-1").WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(100))

	// If limitFeatures is 2, it should stop after 2 features
	mock.ExpectQuery("SELECT AVG").WillReturnRows(sqlmock.NewRows([]string{"mean", "stddev", "null_count", "min_val", "max_val"}).
		AddRow(10.0, 1.0, 0, 0.0, 20.0))
	mock.ExpectQuery("SELECT WIDTH_BUCKET").WillReturnRows(sqlmock.NewRows([]string{"bucket", "count"}).
		AddRow(1, 100))
	mock.ExpectQuery("SELECT AVG").WillReturnRows(sqlmock.NewRows([]string{"mean", "stddev", "null_count", "min_val", "max_val"}).
		AddRow(10.0, 1.0, 0, 0.0, 20.0))
	mock.ExpectQuery("SELECT WIDTH_BUCKET").WillReturnRows(sqlmock.NewRows([]string{"bucket", "count"}).
		AddRow(1, 100))

	resp, err := s.GetDatasetProfile(context.Background(), "test-dataset", 2, 5, "tenant-1")
	require.NoError(t, err)
	assert.Len(t, resp.FeatureProfiles, 2)
	assert.True(t, resp.IsPartial)
	assert.GreaterOrEqual(t, resp.TruncatedKeys, int32(4))
}

func TestProfileCategoricalJSONBKey_UsesParameterizedKey(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	suspiciousKey := `dyn_cat_1' OR 1=1 --`

	mock.ExpectQuery(`(?s)WITH scoped AS.*jsonb_each_text\(s\.feature_map\).*WHERE key = \$1.*WHERE value IS NULL`).
		WithArgs(suspiciousKey, "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(2))
	mock.ExpectQuery(`(?s)WITH scoped AS.*SELECT value, COUNT\(\*\) as count.*GROUP BY value.*LIMIT \$3`).
		WithArgs(suspiciousKey, "tenant-1", 5).
		WillReturnRows(sqlmock.NewRows([]string{"value", "count"}).
			AddRow("x", 10).
			AddRow("y", 20))

	profile, err := s.profileCategoricalJSONBKey(context.Background(), "generated_records", "categorical_features", suspiciousKey, 40, 5, "tenant-1")
	require.NoError(t, err)
	require.NotNil(t, profile)
	assert.Equal(t, suspiciousKey, profile.Name)
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestProfileNumericJSONBKey_RejectsUnsupportedSourceIdentifiers(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	_, err = s.profileNumericJSONBKey(
		context.Background(),
		"generated_records; DROP TABLE generated_records; --",
		"numerical_features",
		"safe-key",
		100,
		5,
		"tenant-1",
	)
	require.Error(t, err)
	assert.Equal(t, codes.InvalidArgument, status.Code(err))
	require.NoError(t, mock.ExpectationsWereMet())
}
