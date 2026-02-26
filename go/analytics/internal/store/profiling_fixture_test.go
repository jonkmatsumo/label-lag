package store

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type jsonbProfileFixture struct {
	TenantID     string `json:"tenant_id"`
	TotalRecords int64  `json:"total_records"`
	Numeric      struct {
		Key              string  `json:"key"`
		Mean             float64 `json:"mean"`
		StdDev           float64 `json:"stddev"`
		NullRate         float64 `json:"null_rate"`
		Min              float64 `json:"min"`
		Max              float64 `json:"max"`
		BucketCount      int32   `json:"bucket_count"`
		BucketFirstCount int64   `json:"bucket_first_count"`
	} `json:"numeric"`
	Categorical struct {
		Key       string  `json:"key"`
		NullRate  float64 `json:"null_rate"`
		TopValues []struct {
			Value string `json:"value"`
			Count int64  `json:"count"`
		} `json:"top_values"`
	} `json:"categorical"`
}

func TestJSONBProfiling_FixtureDrivenOutput(t *testing.T) {
	fixture := loadJSONBProfileFixture(t)

	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)

	mock.ExpectQuery(`(?s)WITH scoped AS.*jsonb_each_text\(s\.feature_map\).*WHERE key = \$1.*SELECT\s+AVG\(numeric_value\)::float8`).
		WithArgs(fixture.Numeric.Key, fixture.TenantID).
		WillReturnRows(sqlmock.NewRows([]string{"mean", "stddev", "null_count", "min_val", "max_val"}).
			AddRow(fixture.Numeric.Mean, fixture.Numeric.StdDev, 5, fixture.Numeric.Min, fixture.Numeric.Max))
	mock.ExpectQuery(`(?s)WITH scoped AS.*SELECT\s+WIDTH_BUCKET\(numeric_value,`).
		WithArgs(fixture.Numeric.Key, fixture.TenantID).
		WillReturnRows(sqlmock.NewRows([]string{"bucket", "count"}).
			AddRow(1, fixture.Numeric.BucketFirstCount))

	numeric, err := s.profileNumericJSONBKey(
		context.Background(),
		"generated_records",
		"numerical_features",
		fixture.Numeric.Key,
		fixture.TotalRecords,
		fixture.Numeric.BucketCount,
		fixture.TenantID,
	)
	require.NoError(t, err)
	require.NotNil(t, numeric)
	assert.Equal(t, fixture.Numeric.Key, numeric.Name)
	assert.Equal(t, fixture.Numeric.NullRate, numeric.NullRate)
	assert.Equal(t, fixture.Numeric.Mean, numeric.Mean)
	assert.Equal(t, fixture.Numeric.StdDev, numeric.StdDev)
	require.Len(t, numeric.Histogram, int(fixture.Numeric.BucketCount))
	assert.Equal(t, fixture.Numeric.BucketFirstCount, numeric.Histogram[0].Count)

	mock.ExpectQuery(`(?s)WITH scoped AS.*jsonb_each_text\(s\.feature_map\).*WHERE key = \$1.*WHERE value IS NULL`).
		WithArgs(fixture.Categorical.Key, fixture.TenantID).
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(10))
	mock.ExpectQuery(`(?s)WITH scoped AS.*SELECT value, COUNT\(\*\) as count.*GROUP BY value.*LIMIT \$3`).
		WithArgs(fixture.Categorical.Key, fixture.TenantID, 2).
		WillReturnRows(sqlmock.NewRows([]string{"value", "count"}).
			AddRow(fixture.Categorical.TopValues[0].Value, fixture.Categorical.TopValues[0].Count).
			AddRow(fixture.Categorical.TopValues[1].Value, fixture.Categorical.TopValues[1].Count))

	categorical, err := s.profileCategoricalJSONBKey(
		context.Background(),
		"generated_records",
		"categorical_features",
		fixture.Categorical.Key,
		fixture.TotalRecords,
		2,
		fixture.TenantID,
	)
	require.NoError(t, err)
	require.NotNil(t, categorical)
	assert.Equal(t, fixture.Categorical.Key, categorical.Name)
	assert.Equal(t, fixture.Categorical.NullRate, categorical.NullRate)
	require.Len(t, categorical.TopValues, len(fixture.Categorical.TopValues))
	for i, expected := range fixture.Categorical.TopValues {
		assert.Equal(t, expected.Value, categorical.TopValues[i].Value)
		assert.Equal(t, expected.Count, categorical.TopValues[i].Count)
	}

	require.NoError(t, mock.ExpectationsWereMet())
}

func TestProfileNumericJSONBKey_UsesParameterizedKey(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	require.NoError(t, err)
	defer db.Close()

	s := NewSQLStore(db)
	suspiciousKey := `dyn_num_1')::numeric); DROP TABLE generated_records; --`

	mock.ExpectQuery(`(?s)WITH scoped AS.*jsonb_each_text\(s\.feature_map\).*WHERE key = \$1.*SELECT\s+AVG\(numeric_value\)::float8`).
		WithArgs(suspiciousKey, "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"mean", "stddev", "null_count", "min_val", "max_val"}).
			AddRow(10.0, 2.0, 1, 1.0, 20.0))
	mock.ExpectQuery(`(?s)WITH scoped AS.*SELECT\s+WIDTH_BUCKET\(numeric_value,`).
		WithArgs(suspiciousKey, "tenant-1").
		WillReturnRows(sqlmock.NewRows([]string{"bucket", "count"}).AddRow(1, 99))

	profile, err := s.profileNumericJSONBKey(context.Background(), "generated_records", "numerical_features", suspiciousKey, 100, 5, "tenant-1")
	require.NoError(t, err)
	require.NotNil(t, profile)
	assert.Equal(t, suspiciousKey, profile.Name)
	require.NoError(t, mock.ExpectationsWereMet())
}

func loadJSONBProfileFixture(t *testing.T) *jsonbProfileFixture {
	t.Helper()
	contents, err := os.ReadFile("testdata/jsonb_profile_fixture.json")
	require.NoError(t, err)

	var fixture jsonbProfileFixture
	require.NoError(t, json.Unmarshal(contents, &fixture))
	return &fixture
}
