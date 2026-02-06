package rules

import (
	"context"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestFileProvider_HotReload(t *testing.T) {
	// Skip if watch is not enabled or fsnotify is not supported in this env
	// But actually we want to test it.
	os.Setenv("INFERENCE_GATEWAY_RULES_WATCH", "true")
	defer os.Unsetenv("INFERENCE_GATEWAY_RULES_WATCH")

	tmpDir, err := os.MkdirTemp("", "rules-test")
	assert.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	rulesPath := filepath.Join(tmpDir, "rules.json")
	
	initialRules := `{"version": "v1", "rules": [{"id": "r1", "field": "f1", "op": ">", "value": 10, "action": "reject"}]}`
	err = os.WriteFile(rulesPath, []byte(initialRules), 0644)
	assert.NoError(t, err)

	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	p, err := NewFileProvider(rulesPath, logger)
	assert.NoError(t, err)

	ctx := context.Background()
	rs, err := p.GetRules(ctx)
	assert.NoError(t, err)
	assert.Equal(t, "v1", rs.Version)
	assert.Len(t, rs.Rules, 1)

	// Update rules
	updatedRules := `{"version": "v2", "rules": [{"id": "r1", "field": "f1", "op": ">", "value": 20, "action": "reject"}]}`
	err = os.WriteFile(rulesPath, []byte(updatedRules), 0644)
	assert.NoError(t, err)

	// Wait for reload
	time.Sleep(200 * time.Millisecond)

	rs, err = p.GetRules(ctx)
	assert.NoError(t, err)
	assert.Equal(t, "v2", rs.Version)
	assert.Equal(t, 20.0, rs.Rules[0].Value)
}
