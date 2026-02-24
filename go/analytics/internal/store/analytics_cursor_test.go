package store

import (
	"encoding/base64"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTransactionCursor_RoundTrip(t *testing.T) {
	now := time.Now().Truncate(time.Microsecond).UTC()
	recordID := "txn-123"

	cursorStr := encodeTransactionCursor(now, recordID)
	require.NotEmpty(t, cursorStr)

	decoded, err := decodeTransactionCursor(cursorStr)
	require.NoError(t, err)
	require.NotNil(t, decoded)

	assert.True(t, now.Equal(decoded.CreatedAt), "Expected %v, got %v", now, decoded.CreatedAt)
	assert.Equal(t, recordID, decoded.RecordId)
}

func TestTransactionCursor_Empty(t *testing.T) {
	decoded, err := decodeTransactionCursor("")
	assert.NoError(t, err)
	assert.Nil(t, decoded)
}

func TestTransactionCursor_Invalid(t *testing.T) {
	decoded, err := decodeTransactionCursor("not-a-cursor")
	assert.Error(t, err)
	assert.Nil(t, decoded)
}

func TestTransactionCursor_InvalidPayload(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString([]byte(`{"created_at":"2024-01-01T00:00:00Z"}`))
	decoded, err := decodeTransactionCursor(raw)
	assert.Error(t, err)
	assert.Nil(t, decoded)
}
