package obs

import (
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
)

func TestRegisterDBPoolStatsCollector_AllowsDuplicateRegistration(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	reg := prometheus.NewRegistry()
	require.NoError(t, RegisterDBPoolStatsCollector(reg, db))
	require.NoError(t, RegisterDBPoolStatsCollector(reg, db))
}

func TestDBPoolStatsCollector_EmitsBoundedMetrics(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	reg := prometheus.NewRegistry()
	require.NoError(t, RegisterDBPoolStatsCollector(reg, db))

	metricFamilies, err := reg.Gather()
	require.NoError(t, err)

	expected := map[string]bool{
		"analytics_db_pool_open_connections":            false,
		"analytics_db_pool_in_use_connections":          false,
		"analytics_db_pool_idle_connections":            false,
		"analytics_db_pool_max_open_connections":        false,
		"analytics_db_pool_wait_count_total":            false,
		"analytics_db_pool_wait_duration_seconds_total": false,
		"analytics_db_pool_max_idle_closed_total":       false,
		"analytics_db_pool_max_lifetime_closed_total":   false,
	}

	for _, family := range metricFamilies {
		name := family.GetName()
		if _, ok := expected[name]; ok {
			expected[name] = true
		}
		if strings.HasPrefix(name, "analytics_db_pool_") {
			for _, metric := range family.GetMetric() {
				require.Len(t, metric.GetLabel(), 0, "db pool metrics must not use labels")
			}
		}
	}

	for name, seen := range expected {
		require.Truef(t, seen, "expected metric family %s to be exported", name)
	}
}
