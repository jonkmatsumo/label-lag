package config

import (
	"fmt"
	"strings"
)

const (
	DefaultDatabaseURL = "postgresql://synthetic:synthetic_dev_password@localhost:5542/synthetic_data?sslmode=disable"
)

func ResolveDatabaseURL(getenv func(string) string) (string, error) {
	if value := strings.TrimSpace(getenv("DATABASE_URL")); value != "" {
		return value, nil
	}
	allowDefaults := strings.EqualFold(getenv("ANALYTICS_CRUD_ALLOW_INSECURE_DEFAULTS"), "true") ||
		strings.EqualFold(getenv("ANALYTICS_CRUD_ALLOW_INSECURE_DEFAULTS"), "1")
	if allowDefaults {
		return DefaultDatabaseURL, nil
	}
	return "", fmt.Errorf("DATABASE_URL is required")
}
