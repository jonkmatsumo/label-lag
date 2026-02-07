package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveDatabaseURLUsesEnv(t *testing.T) {
	value, err := ResolveDatabaseURL(func(key string) string {
		if key == "DATABASE_URL" {
			return "postgresql://user:pass@localhost:5432/db"
		}
		return ""
	})
	require.NoError(t, err)
	assert.Equal(t, "postgresql://user:pass@localhost:5432/db", value)
}

func TestResolveDatabaseURLAllowsDefaultsWhenEnabled(t *testing.T) {
	value, err := ResolveDatabaseURL(func(key string) string {
		if key == "ANALYTICS_CRUD_ALLOW_INSECURE_DEFAULTS" {
			return "true"
		}
		return ""
	})
	require.NoError(t, err)
	assert.Equal(t, DefaultDatabaseURL, value)
}

func TestResolveDatabaseURLRequiresExplicitValue(t *testing.T) {
	_, err := ResolveDatabaseURL(func(string) string { return "" })
	require.Error(t, err)
}
